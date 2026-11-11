"""Decode CUDA graph on a GPU (``-m gpu``): graph vs eager, changing inputs, KV safety, fallbacks.

fp32 tiny checkpoints on CUDA. The graph captures one shape (batch 16 x 16 blocks of 4 tokens =
64-token contexts here); steps that do not fit run the eager batched path.
"""

from __future__ import annotations

import random

import pytest
import torch

from dli.engine import LLMEngine, SchedulerConfig
from dli.engine.cuda_graph import DecodeGraph
from dli.model import LlamaForCausalLM, load_model
from dli.sampling import SamplingParams

from .conftest import TinyFactory

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]
GRAPH = (16, 16)


@pytest.fixture
def model(tiny_model: TinyFactory) -> LlamaForCausalLM:
    _, ckpt = tiny_model("llama")  # GQA: 8 heads, 2 KV heads
    return load_model(ckpt, device="cuda")


def _engine(model: LlamaForCausalLM, *, graph: bool, num_blocks: int = 256) -> LLMEngine:
    return LLMEngine.for_model(
        model,
        num_blocks=num_blocks,
        block_size=4,
        config=SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=512),
        cuda_graph=GRAPH if graph else None,
    )


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


def test_replay_equals_eager_forward_on_the_same_padded_inputs(model: LlamaForCausalLM) -> None:
    eng = _engine(model, graph=True)
    g: DecodeGraph = eng.runner.decode_graph  # type: ignore[attr-defined]
    assert g.capture(), g.stats.status
    eng.bm.allocate("s", list(range(3, 23)))  # 20 tokens -> 5 blocks of real KV to read
    eng.bm.mark_computed("s", 20)
    eng.bm.append_slots("s", [7])
    table = eng.bm.block_table("s")
    replayed = g.run([7], [20], [eng.bm.slot_mapping("s", 20, 21)[0]], [table], [21])
    eager = g._forward()[:1]  # same static buffers, no graph
    assert torch.equal(replayed, eager), (
        "graph replay must be bitwise equal to eager on the same inputs"
    )


@pytest.mark.parametrize(("n", "num_blocks", "seed"), [(16, 256, 1), (11, 256, 2), (16, 40, 3)])
def test_engine_with_graph_matches_eager_over_changing_steps(
    model: LlamaForCausalLM, n: int, num_blocks: int, seed: int
) -> None:
    """Every step replays with new inputs; batches shrink as requests finish (padding rows)."""
    ref_out, ref, _ = _drive(_engine(model, graph=False, num_blocks=num_blocks), n, seed)
    eng = _engine(model, graph=True, num_blocks=num_blocks)
    out, rec, preemptions = _drive(eng, n, seed)
    stats = eng.runner.decode_graph.stats  # type: ignore[attr-defined]
    assert stats.status == "captured"
    assert stats.replays > 10, stats
    assert out == ref_out
    assert len(rec.steps) == len(ref.steps)
    worst = max(float((a - b).abs().max()) for a, b in zip(ref.steps, rec.steps, strict=True))
    assert worst < 1e-4, f"max logit difference {worst}"
    if num_blocks == 40:
        assert preemptions > 0, "preemption not exercised"


def test_padded_replay_never_touches_other_kv_blocks(model: LlamaForCausalLM) -> None:
    eng = _engine(model, graph=True)
    g: DecodeGraph = eng.runner.decode_graph  # type: ignore[attr-defined]
    assert g.capture()
    store, bs = eng.runner.store, 4  # type: ignore[attr-defined]
    for i in range(5):  # 5 real sequences, 11 padding rows
        eng.bm.allocate(f"s{i}", list(range(3 + i, 13 + i)))
        eng.bm.mark_computed(f"s{i}", 10)
        eng.bm.append_slots(f"s{i}", [5 + i])
    store.k.normal_()  # distinctive contents everywhere
    store.v.normal_()
    before_k, before_v = store.k.clone(), store.v.clone()
    slots = [eng.bm.slot_mapping(f"s{i}", 10, 11)[0] for i in range(5)]
    g.run(
        [5, 6, 7, 8, 9], [10] * 5, slots, [eng.bm.block_table(f"s{i}") for i in range(5)], [11] * 5
    )
    torch.cuda.synchronize()
    changed = (
        ((store.k != before_k) | (store.v != before_v)).flatten(3).any(-1)
    )  # [layer, block, offset]
    written = {(s // bs, s % bs) for s in slots}
    for layer in range(changed.shape[0]):
        for blk, off in changed[layer].nonzero().tolist():
            assert (blk, off) in written or blk == g.scratch_block, (
                f"layer {layer}: block {blk}:{off} corrupted"
            )


def test_unsupported_shapes_fall_back_to_eager(model: LlamaForCausalLM) -> None:
    eng = _engine(model, graph=True)
    for i in range(20):  # 20 sequences > 16 rows: batch fallback while more than 16 run
        eng.add_request(f"b{i}", [5, 6, 7], SamplingParams(max_tokens=3, ignore_eos=True))
    eng.add_request(
        "long", list(range(3, 90)), SamplingParams(max_tokens=3, ignore_eos=True)
    )  # 23 blocks
    while eng.has_unfinished():
        eng.step()
    fallbacks = eng.runner.decode_graph.stats.fallbacks  # type: ignore[attr-defined]
    assert fallbacks.get("batch", 0) + fallbacks.get("width", 0) > 0, fallbacks


def test_insufficient_free_memory_disables_the_graph(
    model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (64 << 20, 24 << 30))
    ref_out, _, _ = _drive(_engine(model, graph=False), 8, 5)
    eng = _engine(model, graph=True)
    out, _, _ = _drive(eng, 8, 5)
    stats = eng.runner.decode_graph.stats  # type: ignore[attr-defined]
    assert stats.status.startswith("disabled:"), stats.status
    assert "MiB free" in stats.status, stats.status
    assert stats.replays == 0
    assert out == ref_out
