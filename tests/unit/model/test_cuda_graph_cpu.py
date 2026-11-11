"""Decode CUDA graph without a GPU: fallback, scratch block, configuration."""

from __future__ import annotations

import contextlib

import pytest

from dli.engine import LLMEngine, SchedulerConfig
from dli.model import load_model
from dli.sampling import SamplingParams

from .conftest import TinyFactory


def _run(eng: LLMEngine, n: int = 6) -> list[list[int]]:
    reqs = [
        eng.add_request(
            f"r{i}", [5 + i, 9, 11 + i], SamplingParams(max_tokens=6 + i, ignore_eos=True)
        )
        for i in range(n)
    ]
    while eng.has_unfinished():
        eng.step()
    return [r.output_ids for r in reqs]


def test_cpu_disables_the_graph_and_runs_eagerly(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    cfg = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256)
    graphed = LLMEngine.for_model(
        model, num_blocks=64, block_size=4, config=cfg, cuda_graph=(16, 16)
    )
    plain = LLMEngine.for_model(model, num_blocks=64, block_size=4, config=cfg)
    assert _run(graphed) == _run(plain)
    stats = graphed.runner.decode_graph.stats  # type: ignore[attr-defined]
    assert stats.status == "disabled: not a CUDA device"
    assert stats.replays == 0
    assert sum(stats.fallbacks.values()) > 0


def test_scratch_block_exists_but_is_never_allocated(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    # Tight pool: allocation, eviction and preemption all touch the free list.
    eng = LLMEngine.for_model(
        model,
        num_blocks=12,
        block_size=4,
        cuda_graph=(16, 16),
        config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=64),
    )
    store = eng.runner.store  # type: ignore[attr-defined]
    assert store.k.shape[1] == 13  # one extra block in the store
    assert eng.bm.num_blocks == 12  # the manager knows only 0..11
    seen: set[int] = set()
    reqs = [
        eng.add_request(
            f"r{i}", list(range(3, 3 + 7 + i)), SamplingParams(max_tokens=9, ignore_eos=True)
        )
        for i in range(6)
    ]
    while eng.has_unfinished():
        for r in reqs:
            if not r.is_finished:
                with contextlib.suppress(KeyError):  # waiting requests hold no blocks
                    seen.update(eng.bm.block_table(r.request_id))
        eng.step()
    assert seen
    assert max(seen) < 12, "the scratch block (12) was handed to a sequence"


def test_cuda_graph_requires_the_batched_decode_path(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    with pytest.raises(ValueError, match="requires batched_decode"):
        LLMEngine.for_model(
            load_model(ckpt), num_blocks=16, block_size=4, cuda_graph=(16, 16), batched_decode=False
        )
