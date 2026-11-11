"""Decode CUDA graphs without a GPU: bucket choice, fallback, scratch block, configuration."""

from __future__ import annotations

import contextlib

import pytest

from dli.engine import LLMEngine, SchedulerConfig
from dli.engine.cuda_graph import BENCHMARK_SHAPES, DecodeGraphs, choose_shape
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
        model, num_blocks=64, block_size=4, config=cfg, cuda_graphs=[(16, 16)]
    )
    plain = LLMEngine.for_model(model, num_blocks=64, block_size=4, config=cfg)
    assert _run(graphed) == _run(plain)
    stats = graphed.runner.decode_graphs.stats  # type: ignore[attr-defined]
    assert stats.status == "disabled: not a CUDA device"
    assert graphed.runner.decode_graphs.summary()["replays"] == 0  # type: ignore[attr-defined]
    assert sum(stats.fallbacks.values()) > 0


def test_scratch_block_exists_but_is_never_allocated(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    # Tight pool: allocation, eviction and preemption all touch the free list.
    eng = LLMEngine.for_model(
        model,
        num_blocks=12,
        block_size=4,
        cuda_graphs=[(16, 16)],
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


def test_cuda_graphs_require_the_batched_decode_path(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    with pytest.raises(ValueError, match="require batched_decode"):
        LLMEngine.for_model(
            load_model(ckpt),
            num_blocks=16,
            block_size=4,
            cuda_graphs=[(16, 16)],
            batched_decode=False,
        )


@pytest.mark.parametrize(
    ("num_seqs", "width", "expected"),
    [
        (1, 1, (16, 16)),
        (16, 16, (16, 16)),
        (16, 17, (16, 24)),  # wider context: same batch bucket, wider graph
        (17, 3, (64, 24)),  # more rows: the smallest batch bucket that holds them
        (64, 24, (64, 24)),
        (65, 1, None),  # more rows than any bucket
        (8, 25, None),  # wider than any bucket
    ],
)
def test_choose_shape_picks_the_smallest_bucket_that_holds_the_step(
    num_seqs: int, width: int, expected: tuple[int, int] | None
) -> None:
    shapes = [(64, 24), (16, 16), (16, 24)]
    assert choose_shape(shapes, num_seqs, width) == expected


def test_benchmark_shapes_cover_the_closed_fixed_workload() -> None:
    """closed-fixed: 256-token prompts + 128 outputs at block 16, concurrency 1/16/64."""
    block, longest = 16, 256 + 128 - 1  # the last sampled token is never fed back
    width = -(-longest // block)
    for concurrency in (1, 16, 64):
        assert choose_shape(BENCHMARK_SHAPES, concurrency, width) is not None


@pytest.mark.parametrize("shapes", [[], [(0, 16)], [(16, -1)]])
def test_invalid_shapes_are_rejected(
    tiny_model: TinyFactory, shapes: list[tuple[int, int]]
) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    eng = LLMEngine.for_model(model, num_blocks=16, block_size=4)
    with pytest.raises(ValueError, match="must be positive"):
        DecodeGraphs(model, eng.runner.store, shapes=shapes, scratch_block=0)  # type: ignore[attr-defined]


def test_cpu_disables_every_bucket(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    cfg = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256)
    eng = LLMEngine.for_model(
        model, num_blocks=64, block_size=4, config=cfg, cuda_graphs=[(4, 16), (8, 16)]
    )
    _run(eng)
    graphs = eng.runner.decode_graphs  # type: ignore[attr-defined]
    assert all(g.stats.status == "disabled: not a CUDA device" for g in graphs.graphs)
    summary = graphs.summary()
    assert summary["replays"] == 0
    assert summary["fallbacks"] == {
        "disabled: not a CUDA device": sum(graphs.stats.fallbacks.values())
    }
