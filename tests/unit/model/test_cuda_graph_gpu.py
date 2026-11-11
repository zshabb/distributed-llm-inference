"""Decode CUDA graphs on a GPU (``-m gpu``): graph vs eager, changing inputs, KV safety, fallbacks.

fp32 tiny checkpoints on CUDA. Two batch buckets (16 and 64 rows) x one width bucket (16 blocks of
4 tokens = 64-token contexts here), mirroring the benchmark set's structure at a tiny scale; steps
that fit no bucket run the eager batched path.
"""

from __future__ import annotations

import random

import pytest
import torch

from dli.engine import LLMEngine, SchedulerConfig
from dli.engine.cuda_graph import DecodeGraph, DecodeGraphs
from dli.model import LlamaForCausalLM, load_model
from dli.sampling import SamplingParams

from .conftest import TinyFactory

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]
SHAPES = [(16, 16), (64, 16)]


@pytest.fixture
def model(tiny_model: TinyFactory) -> LlamaForCausalLM:
    _, ckpt = tiny_model("llama")  # GQA: 8 heads, 2 KV heads
    return load_model(ckpt, device="cuda")


def _engine(model: LlamaForCausalLM, *, graph: bool, num_blocks: int = 512) -> LLMEngine:
    return LLMEngine.for_model(
        model,
        num_blocks=num_blocks,
        block_size=4,
        config=SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=1024),
        cuda_graphs=SHAPES if graph else None,
    )


def _graphs(eng: LLMEngine) -> DecodeGraphs:
    graphs: DecodeGraphs = eng.runner.decode_graphs  # type: ignore[attr-defined]
    return graphs


def _graph(eng: LLMEngine, batch: int) -> DecodeGraph:
    return next(g for g in _graphs(eng).graphs if g.batch == batch)


class Recorder:
    def __init__(self, eng: LLMEngine) -> None:
        self.inner, self.steps = eng.runner, []
        eng.runner = self  # type: ignore[assignment]

    def execute(self, sched: object) -> torch.Tensor:
        out = self.inner.execute(sched)  # type: ignore[arg-type]
        self.steps.append(out.detach().clone())
        return out

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


def _drive(eng: LLMEngine, n: int, seed: int) -> tuple[list[list[int]], Recorder, int]:
    rec = Recorder(eng)
    rng = random.Random(seed)
    reqs = [
        eng.add_request(
            f"r{i}",
            [rng.randrange(3, 97) for _ in range(rng.randrange(2, 20))],
            SamplingParams(max_tokens=rng.randrange(5, 40), ignore_eos=True),
        )
        for i in range(n)
    ]
    while eng.has_unfinished():
        eng.step()
    eng.bm.check_invariants()
    return [r.output_ids for r in reqs], rec, sum(r.num_preemptions for r in reqs)


def _decode_slot(eng: LLMEngine, rid: str, prompt: list[int], token: int) -> int:
    eng.bm.allocate(rid, prompt)
    eng.bm.mark_computed(rid, len(prompt))
    eng.bm.append_slots(rid, [token])
    slot: int = eng.bm.slot_mapping(rid, len(prompt), len(prompt) + 1)[0]
    return slot


@pytest.mark.parametrize("batch", [16, 64])
def test_replay_equals_eager_forward_on_the_same_padded_inputs(
    model: LlamaForCausalLM, batch: int
) -> None:
    eng = _engine(model, graph=True)
    assert _graphs(eng).capture(), _graphs(eng).stats.status
    g = _graph(eng, batch)
    n = 3 if batch == 16 else 37  # padded rows in both buckets
    rids = [f"s{i}" for i in range(n)]
    slots = [_decode_slot(eng, r, list(range(3 + i, 23 + i)), 7) for i, r in enumerate(rids)]
    tables = [eng.bm.block_table(r) for r in rids]
    replayed = g.run([7] * n, [20] * n, slots, tables, [21] * n)
    eager = g._forward()[:n]  # same static buffers, no graph
    assert torch.equal(replayed, eager), "graph replay must be bitwise equal to eager"


def test_buckets_sharing_one_pool_stay_correct_when_alternated(model: LlamaForCausalLM) -> None:
    """Replaying the small graph between two large replays must not disturb the large one."""
    eng = _engine(model, graph=True)
    assert _graphs(eng).capture()
    small, large = _graph(eng, 16), _graph(eng, 64)
    rids = [f"s{i}" for i in range(40)]
    slots = [_decode_slot(eng, r, list(range(3 + i, 15 + i)), 9) for i, r in enumerate(rids)]
    tables = [eng.bm.block_table(r) for r in rids]
    args = ([9] * 40, [12] * 40, slots, tables, [13] * 40)
    first = large.run(*args)
    small_out = small.run(*(a[:5] for a in args))
    again = large.run(*args)
    assert torch.equal(first, again)
    assert torch.equal(small_out, small._forward()[:5])


@pytest.mark.parametrize(
    ("n", "num_blocks", "seed"),
    [(16, 512, 1), (11, 512, 2), (48, 512, 4), (64, 160, 5)],
)
def test_engine_with_graphs_matches_eager_over_changing_steps(
    model: LlamaForCausalLM, n: int, num_blocks: int, seed: int
) -> None:
    """Every step replays with new inputs; batches shrink as requests finish (padding rows) and
    cross from the 64-row bucket into the 16-row one; a tight pool forces preemption."""
    ref_out, ref, _ = _drive(_engine(model, graph=False, num_blocks=num_blocks), n, seed)
    eng = _engine(model, graph=True, num_blocks=num_blocks)
    out, rec, preemptions = _drive(eng, n, seed)
    graphs = _graphs(eng)
    assert graphs.stats.status == "captured"
    assert sum(g.stats.replays for g in graphs.graphs) > 10, graphs.summary()
    assert out == ref_out
    assert len(rec.steps) == len(ref.steps)
    worst = max(float((a - b).abs().max()) for a, b in zip(ref.steps, rec.steps, strict=True))
    assert worst < 1e-4, f"max logit difference {worst}"
    if n > 16:
        assert _graph(eng, 64).stats.replays > 0
        assert _graph(eng, 16).stats.replays > 0
    if num_blocks == 160:
        assert preemptions > 0, "preemption not exercised"


@pytest.mark.parametrize("batch", [16, 64])
def test_padded_replay_never_touches_other_kv_blocks(model: LlamaForCausalLM, batch: int) -> None:
    eng = _engine(model, graph=True)
    assert _graphs(eng).capture()
    g = _graph(eng, batch)
    store, bs = eng.runner.store, 4  # type: ignore[attr-defined]
    n = 5 if batch == 16 else 21  # real rows; the rest are padding
    rids = [f"s{i}" for i in range(n)]
    slots = [
        _decode_slot(eng, r, list(range(3 + i, 13 + i)), 5 + i % 7) for i, r in enumerate(rids)
    ]
    store.k.normal_()  # distinctive contents everywhere
    store.v.normal_()
    before_k, before_v = store.k.clone(), store.v.clone()
    g.run(
        [5 + i % 7 for i in range(n)],
        [10] * n,
        slots,
        [eng.bm.block_table(r) for r in rids],
        [11] * n,
    )
    torch.cuda.synchronize()
    changed = (
        ((store.k != before_k) | (store.v != before_v)).flatten(3).any(-1)
    )  # [layer, blk, off]
    written = {(s // bs, s % bs) for s in slots}
    for layer in range(changed.shape[0]):
        for blk, off in changed[layer].nonzero().tolist():
            assert (blk, off) in written or blk == g.scratch_block, (
                f"layer {layer}: block {blk}:{off} corrupted"
            )


def test_steps_that_fit_no_bucket_fall_back_to_eager(model: LlamaForCausalLM) -> None:
    outs = []
    for graph in (False, True):
        eng = LLMEngine.for_model(
            model,
            num_blocks=512,
            block_size=4,
            config=SchedulerConfig(max_num_seqs=80, max_num_batched_tokens=1024),
            cuda_graphs=SHAPES if graph else None,
        )
        reqs = [
            eng.add_request(
                f"b{i}", [5, 6, i % 90 + 3], SamplingParams(max_tokens=3, ignore_eos=True)
            )
            for i in range(70)  # 70 rows > 64
        ]
        # 87-token prompt: 22-23 blocks > 16, still decoding after the short requests finish
        reqs.append(
            eng.add_request(
                "long", list(range(3, 90)), SamplingParams(max_tokens=8, ignore_eos=True)
            )
        )
        while eng.has_unfinished():
            eng.step()
        outs.append([r.output_ids for r in reqs])
    assert outs[0] == outs[1]
    fallbacks = _graphs(eng).stats.fallbacks
    assert fallbacks.get("batch", 0) > 0, fallbacks
    assert fallbacks.get("width", 0) > 0, fallbacks


def test_a_bucket_without_memory_is_skipped_and_the_rest_still_serve(
    model: LlamaForCausalLM,
) -> None:
    ref_out, _, _ = _drive(_engine(model, graph=False), 30, 6)
    eng = _engine(model, graph=True)
    _graph(eng, 64).memory_margin = 1 << 50  # the large bucket's memory gate cannot pass
    out, _, _ = _drive(eng, 30, 6)
    graphs = _graphs(eng)
    assert out == ref_out
    assert graphs.stats.status == "captured"
    assert "MiB free" in _graph(eng, 64).stats.status
    assert _graph(eng, 64).stats.replays == 0
    assert _graph(eng, 16).stats.replays > 0  # batches <= 16 still replay
    assert graphs.stats.fallbacks.get("batch", 0) > 0  # batches > 16 ran eagerly


def test_insufficient_free_memory_disables_every_bucket(
    model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (64 << 20, 24 << 30))
    ref_out, _, _ = _drive(_engine(model, graph=False), 8, 5)
    eng = _engine(model, graph=True)
    out, _, _ = _drive(eng, 8, 5)
    graphs = _graphs(eng)
    assert graphs.stats.status.startswith("disabled:"), graphs.stats.status
    assert all("MiB free" in g.stats.status for g in graphs.graphs)
    assert sum(g.stats.replays for g in graphs.graphs) == 0
    assert out == ref_out


def test_graph_memory_overhead_is_measured(model: LlamaForCausalLM) -> None:
    eng = _engine(model, graph=True)
    graphs = _graphs(eng)
    assert graphs.capture()
    # static buffers + the shared pool: positive, and small for a tiny model
    assert 0 < graphs.stats.memory_overhead < 256 << 20, graphs.summary()
