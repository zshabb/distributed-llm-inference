"""AsyncEngine: thread hand-off, streaming, cancellation on early close, crash isolation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

import pytest
import torch

from dli.engine import FinishReason, LLMEngine, SchedulerConfig, SchedulerOutput
from dli.kv import BlockManager
from dli.sampling import SamplingParams
from dli.server.async_engine import AsyncEngine, EngineDeadError
from tests.unit.engine.fake import FakeRunner, reference


@pytest.fixture
def aengine() -> Iterator[AsyncEngine]:
    bm = BlockManager(128, 4, watermark=0.0)
    eng = LLMEngine(FakeRunner(bm), bm, SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=64))
    ae = AsyncEngine(eng)
    ae.start()
    yield ae
    ae.stop()


async def _collect(
    ae: AsyncEngine, rid: str, prompt: list[int], n: int
) -> tuple[list[int], object]:
    toks: list[int] = []
    reason = None
    async for ev in ae.generate(rid, prompt, SamplingParams(max_tokens=n)):
        toks.extend(ev.token_ids)
        reason = ev.finish_reason
    return toks, reason


def _wait_until(pred: object, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():  # type: ignore[operator]
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_concurrent_streams_match_reference(aengine: AsyncEngine) -> None:
    async def main() -> list[tuple[list[int], object]]:
        return await asyncio.gather(
            *(_collect(aengine, f"r{i}", [i, i + 1, i + 2], 10 + i) for i in range(12))
        )

    results = asyncio.run(main())
    for i, (toks, reason) in enumerate(results):
        assert toks == reference([i, i + 1, i + 2], 10 + i)
        assert reason is FinishReason.LENGTH


def test_closing_the_stream_early_aborts_and_frees_kv(aengine: AsyncEngine) -> None:
    async def main() -> None:
        gen = aengine.generate("victim", [1, 2, 3], SamplingParams(max_tokens=10_000 // 4))
        got = 0
        async for ev in gen:
            got += len(ev.token_ids)
            if got >= 3:
                break  # client disconnect: the iterator is closed before finishing
        await gen.aclose()

    asyncio.run(main())
    _wait_until(lambda: aengine.snapshot.num_running == 0 and aengine.snapshot.num_waiting == 0)
    snap = aengine.snapshot
    assert snap.kv_used_blocks == 0, "aborted request still holds KV blocks"


def test_request_that_never_fits_finishes_with_capacity(aengine: AsyncEngine) -> None:
    toks, reason = asyncio.run(_collect(aengine, "huge", list(range(400)), 400))
    assert toks == []
    assert reason is FinishReason.CAPACITY


class _ExplodingRunner:
    def __init__(self, bm: BlockManager) -> None:
        self.bm = bm
        self.calls = 0

    def execute(self, sched: SchedulerOutput) -> torch.Tensor:
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("simulated CUDA error")
        for ch in sched.chunks:
            self.bm.mark_computed(ch.request.request_id, ch.start + ch.num_tokens)
        rows = sum(c.needs_logits for c in sched.chunks)
        return torch.zeros(rows, 50)


def test_engine_crash_fails_requests_loudly_and_marks_unhealthy() -> None:
    bm = BlockManager(64, 4, watermark=0.0)
    ae = AsyncEngine(LLMEngine(_ExplodingRunner(bm), bm))
    ae.start()

    async def main() -> list[str | None]:
        errors = []
        async for ev in ae.generate("a", [1, 2], SamplingParams(max_tokens=50)):
            if ev.error:
                errors.append(ev.error)
        return errors

    errors = asyncio.run(main())
    assert errors
    assert "simulated CUDA error" in errors[0]
    assert not ae.healthy
    assert not ae.snapshot.healthy

    async def after_crash() -> None:
        async for _ in ae.generate("b", [1], SamplingParams()):
            pass

    with pytest.raises(EngineDeadError):
        asyncio.run(after_crash())


def test_snapshot_tracks_progress(aengine: AsyncEngine) -> None:
    asyncio.run(_collect(aengine, "s", list(range(20)), 30))
    _wait_until(lambda: aengine.snapshot.total_generated_tokens >= 30)
    snap = aengine.snapshot
    assert snap.total_steps > 0
    assert snap.prefill_tokens_per_s > 0
    assert snap.kv_total_blocks == 128


def test_one_event_loop_wakeup_per_step_not_per_token(aengine: AsyncEngine) -> None:
    """Outputs of a step reach the loop in one call_soon_threadsafe, whatever the batch size."""

    async def main() -> tuple[list[tuple[list[int], object]], int]:
        loop = asyncio.get_running_loop()
        calls = {"n": 0}
        real = loop.call_soon_threadsafe

        def counting(*a: object, **k: object) -> object:
            calls["n"] += 1
            return real(*a, **k)  # type: ignore[arg-type]

        loop.call_soon_threadsafe = counting  # type: ignore[method-assign]
        results = await asyncio.gather(
            *(_collect(aengine, f"w{i}", [i, i + 1], 20) for i in range(8))
        )
        return results, calls["n"]

    steps_before = aengine.snapshot.total_steps
    results, wakeups = asyncio.run(main())
    steps = aengine.snapshot.total_steps - steps_before
    for i, (toks, _) in enumerate(results):
        assert toks == reference([i, i + 1], 20)
    tokens = 8 * 20
    assert wakeups <= steps + 1, (wakeups, steps)  # one per step (+ the final publish race)
    assert wakeups < tokens // 4, (wakeups, tokens)
