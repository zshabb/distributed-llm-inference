from __future__ import annotations

import random

import pytest

from dli.engine import FinishReason, LLMEngine, SchedulerConfig
from dli.kv import BlockManager
from dli.sampling import SamplingParams

from .fake import FakeRunner, reference


def make_engine(
    num_blocks: int = 256,
    block_size: int = 4,
    max_seqs: int = 8,
    budget: int = 32,
    eos: tuple[int, ...] = (),
) -> tuple[LLMEngine, FakeRunner, BlockManager]:
    bm = BlockManager(num_blocks, block_size, watermark=0.0)
    runner = FakeRunner(bm)
    eng = LLMEngine(
        runner,
        bm,
        SchedulerConfig(max_num_seqs=max_seqs, max_num_batched_tokens=budget),
        eos_token_ids=eos,
    )
    return eng, runner, bm


def run(eng: LLMEngine, limit: int = 10_000) -> None:
    for _ in range(limit):
        if not eng.has_unfinished():
            return
        eng.step()
    raise AssertionError("engine did not finish (livelock?)")


def test_concurrent_requests_match_isolated_reference() -> None:
    eng, _runner, bm = make_engine()
    rng = random.Random(1)
    prompts = [[rng.randrange(50) for _ in range(rng.randrange(1, 40))] for _ in range(20)]
    reqs = [
        eng.add_request(f"r{i}", p, SamplingParams(max_tokens=rng.randrange(1, 25)))
        for i, p in enumerate(prompts)
    ]
    run(eng)
    for r, p in zip(reqs, prompts, strict=True):
        assert r.output_ids == reference(p, r.params.max_tokens), r.request_id
        assert r.finish_reason is FinishReason.LENGTH
    bm.reset_prefix_cache()
    assert bm.stats().free_blocks == bm.num_blocks, "blocks leaked"


def test_budget_and_sequence_cap_hold_every_step() -> None:
    eng, runner, _ = make_engine(max_seqs=3, budget=10)
    for i in range(10):
        eng.add_request(f"r{i}", list(range(i, i + 17)), SamplingParams(max_tokens=5))
    run(eng)
    for sched in runner.calls:
        assert sched.num_batched_tokens <= 10
        assert len(sched.chunks) <= 3
        assert len({c.request.request_id for c in sched.chunks}) == len(sched.chunks)


def test_long_prompt_is_chunked_and_running_decodes_are_not_stalled() -> None:
    eng, runner, _ = make_engine(budget=8)
    short = eng.add_request("short", [1, 2, 3], SamplingParams(max_tokens=20))
    eng.step()  # short prefills (3 tokens) and samples its first token
    long = eng.add_request("long", list(range(40)), SamplingParams(max_tokens=2))
    for _ in range(5):
        eng.step()
    # While the 40-token prompt is chunked (7 tokens/step after the decode token), the running
    # request keeps decoding every step.
    assert len(short.output_ids) == 6
    long_chunks = [c for s in runner.calls for c in s.chunks if c.request is long]
    assert all(c.num_tokens <= 7 for c in long_chunks)
    assert not long.output_ids  # 40 prompt tokens need > 5 steps at 7 tokens/step
    run(eng)
    assert long.output_ids == reference(list(range(40)), 2)


def test_prefix_cache_hit_skips_recomputing_shared_prompt_blocks() -> None:
    eng, runner, _ = make_engine(budget=64)
    p = list(range(30))
    eng.add_request("a", p, SamplingParams(max_tokens=1))
    run(eng)
    b = eng.add_request("b", [*p, 7], SamplingParams(max_tokens=3))
    run(eng)
    first_b = next(c for s in runner.calls for c in s.chunks if c.request is b)
    assert first_b.start == 28  # 7 full 4-token blocks reused
    assert b.output_ids == reference([*p, 7], 3)


def test_stop_reasons() -> None:
    eng, _, _ = make_engine(eos=(reference([5, 6], 3)[1],))
    r_eos = eng.add_request("eos", [5, 6], SamplingParams(max_tokens=10))
    stop_tok = reference([9], 4)[2]
    r_stop = eng.add_request("stop", [9], SamplingParams(max_tokens=10, stop_token_ids=(stop_tok,)))
    r_ign = eng.add_request("ign", [5, 6], SamplingParams(max_tokens=4, ignore_eos=True))
    run(eng)
    assert (r_eos.finish_reason, len(r_eos.output_ids)) == (FinishReason.STOP, 2)
    assert (r_stop.finish_reason, len(r_stop.output_ids)) == (FinishReason.STOP, 3)
    assert (r_ign.finish_reason, len(r_ign.output_ids)) == (FinishReason.LENGTH, 4)


def test_request_that_can_never_fit_is_rejected_with_capacity() -> None:
    eng, _, _ = make_engine(num_blocks=4, block_size=4)  # 16 token slots in total
    r = eng.add_request("big", list(range(10)), SamplingParams(max_tokens=10))
    assert r.finish_reason is FinishReason.CAPACITY
    assert not eng.has_unfinished()


def test_abort_waiting_and_running_requests_frees_everything() -> None:
    eng, _, bm = make_engine(max_seqs=1)
    running = eng.add_request("run", list(range(9)), SamplingParams(max_tokens=50))
    waiting = eng.add_request("wait", [1, 2], SamplingParams(max_tokens=5))
    eng.step()
    eng.step()
    assert eng.abort("run")
    assert eng.abort("wait")
    assert not eng.abort("run"), "second cancellation is a no-op"
    assert running.finish_reason is FinishReason.ABORTED
    assert waiting.finish_reason is FinishReason.ABORTED
    assert not eng.has_unfinished()
    bm.reset_prefix_cache()
    bm.check_invariants()
    assert bm.stats().free_blocks == bm.num_blocks


def test_duplicate_ids_and_empty_prompts_rejected() -> None:
    eng, _, _ = make_engine()
    eng.add_request("x", [1])
    with pytest.raises(ValueError, match="duplicate"):
        eng.add_request("x", [2])
    with pytest.raises(ValueError, match="non-empty"):
        eng.add_request("y", [])
