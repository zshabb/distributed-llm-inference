"""Triton paged GQA decode attention vs a float64 reference and vs the SDPA path (``-m gpu``).

Kernel level: random paged pools with scattered block tables, every grouping shape the engine can
meet (GQA 7:1 and 3:1, MHA, 4:1 with head_dim 128, a non-power-of-two head_dim), block sizes 4 and
16, fp32 and bf16, sequence lengths from 1 to many pages, every split-K factor. Table entries past a
sequence's length hold an invalid block id, so reading one would fault.

Engine level: engines with the Triton kernel and with SDPA run the same requests (decode, mixed
prefill/decode steps, CUDA graphs, preemption) and must agree token for token and logit for logit.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from dli.engine import LLMEngine, SchedulerConfig
from dli.engine.cuda_graph import LONG_CONTEXT_SHAPES
from dli.model import LlamaForCausalLM, load_model, paged_attention
from dli.sampling import SamplingParams

from .conftest import TinyFactory

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not paged_attention.available(), reason="needs CUDA and Triton"),
]
INVALID = 2**40  # a block id far outside any pool


def _case(
    bsz: int, heads: int, kv_heads: int, d: int, bs: int, dtype: torch.dtype, lens: list[int]
) -> tuple[torch.Tensor, ...]:
    g = torch.Generator(device="cuda").manual_seed(bsz * 31 + heads + d + bs)
    width = max(-(-n // bs) for n in lens) + 2  # wider than needed: padding must be ignored
    num_blocks = bsz * width + 3
    k = torch.randn(num_blocks, bs, kv_heads, d, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(num_blocks, bs, kv_heads, d, device="cuda", dtype=dtype, generator=g)
    perm = torch.randperm(num_blocks, device="cuda", generator=g)
    tables = torch.full((bsz, width), INVALID, dtype=torch.long, device="cuda")
    used = 0
    for i, n in enumerate(lens):
        nb = -(-n // bs)
        tables[i, :nb] = perm[used : used + nb]
        used += nb
    q = torch.randn(heads, bsz, d, device="cuda", dtype=dtype, generator=g).transpose(0, 1)
    seq_lens = torch.tensor(lens, dtype=torch.long, device="cuda")
    return q, k, v, tables, seq_lens


def _reference(q, k, v, tables, seq_lens) -> torch.Tensor:  # type: ignore[no-untyped-def]
    bsz, heads, d = q.shape
    kv_heads = k.shape[2]
    out = torch.empty(bsz, heads, d, dtype=torch.float64, device=q.device)
    for i in range(bsz):
        n = int(seq_lens[i])
        bs = k.shape[1]
        blocks = tables[i, : -(-n // bs)]
        keys = k[blocks].reshape(-1, kv_heads, d)[:n].double()  # [n, kvh, d]
        vals = v[blocks].reshape(-1, kv_heads, d)[:n].double()
        keys = keys.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)  # [h, n, d]
        vals = vals.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        s = (q[i].double().unsqueeze(1) @ keys.transpose(1, 2)) / math.sqrt(d)  # [h, 1, n]
        out[i] = (s.softmax(-1) @ vals).squeeze(1)
    return out


SHAPES = [  # heads, kv_heads, head_dim, block_size, dtype
    (14, 2, 64, 16, torch.bfloat16),  # Qwen2.5-0.5B
    (9, 3, 64, 4, torch.float32),  # SmolLM2-135M-like, small blocks, exact fp32 path
    (8, 8, 64, 16, torch.bfloat16),  # MHA (group 1)
    (32, 8, 128, 16, torch.bfloat16),  # Llama-3-like, head_dim 128
    (8, 2, 80, 16, torch.float32),  # head_dim not a power of two
]


@pytest.mark.parametrize(("heads", "kv_heads", "d", "bs", "dtype"), SHAPES)
@pytest.mark.parametrize("splits", [None, 1, 4, 16])
def test_kernel_matches_a_float64_reference(
    heads: int, kv_heads: int, d: int, bs: int, dtype: torch.dtype, splits: int | None
) -> None:
    rng = random.Random(heads * d + bs)
    lens = [1, bs, bs + 1, 3 * bs - 1] + [rng.randrange(1, 40 * bs) for _ in range(6)]
    q, k, v, tables, seq_lens = _case(len(lens), heads, kv_heads, d, bs, dtype, lens)
    out = paged_attention.paged_decode_attention(q, k, v, tables, seq_lens, splits=splits)
    torch.cuda.synchronize()
    ref = _reference(q, k, v, tables, seq_lens)
    err = float((out.double() - ref).abs().max())
    tol = 2e-5 if dtype is torch.float32 else 1.6e-2  # bf16: output rounding (8 mantissa bits)
    assert err < tol, f"max abs error {err}"


def test_kernel_is_capturable_and_replays_bitwise() -> None:
    q, k, v, tables, seq_lens = _case(
        16, 14, 2, 64, 16, torch.bfloat16, [300 + 7 * i for i in range(16)]
    )
    eager = paged_attention.paged_decode_attention(q, k, v, tables, seq_lens)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        paged_attention.paged_decode_attention(q, k, v, tables, seq_lens)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = paged_attention.paged_decode_attention(q, k, v, tables, seq_lens)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


def test_split_heuristic_fills_the_gpu_without_tiny_chunks() -> None:
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    assert paged_attention.choose_splits(4 * sms, 2, 4096, dev) == 1  # already enough programs
    assert paged_attention.choose_splits(1, 2, 4096, dev) > 1  # one sequence: split its keys
    assert paged_attention.choose_splits(1, 2, 64, dev) == 1  # too short to split


# ---------------------------------------------------------------------------- engine level


@pytest.fixture
def model(tiny_model: TinyFactory) -> LlamaForCausalLM:
    _, ckpt = tiny_model("llama")  # GQA: 8 heads, 2 KV heads, fp32
    return load_model(ckpt, device="cuda")


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


def _drive(
    model: LlamaForCausalLM, kernel: str, *, graphs: bool, num_blocks: int, budget: int, seed: int
) -> tuple[list[list[int]], Recorder, int, LLMEngine]:
    eng = LLMEngine.for_model(
        model,
        num_blocks=num_blocks,
        block_size=4,
        config=SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=budget),
        cuda_graphs=LONG_CONTEXT_SHAPES if graphs else None,
        decode_attention=kernel,
    )
    rec = Recorder(eng)
    rng = random.Random(seed)
    reqs = []
    for i in range(40):  # staggered arrivals: decode, mixed and prefill steps
        prompt = [rng.randrange(3, 97) for _ in range(rng.randrange(2, 60))]
        reqs.append(
            eng.add_request(
                f"r{i}", prompt, SamplingParams(max_tokens=rng.randrange(4, 60), ignore_eos=True)
            )
        )
        if i % 3 == 0 and eng.has_unfinished():
            eng.step()
    while eng.has_unfinished():
        eng.step()
    eng.bm.check_invariants()
    return [r.output_ids for r in reqs], rec, sum(r.num_preemptions for r in reqs), eng


@pytest.mark.parametrize(
    ("graphs", "num_blocks", "budget", "seed"),
    [(False, 2048, 256, 1), (True, 2048, 256, 2), (True, 300, 64, 3)],
)
def test_engine_with_triton_matches_sdpa(
    model: LlamaForCausalLM, graphs: bool, num_blocks: int, budget: int, seed: int
) -> None:
    ref_out, ref, _, _ = _drive(
        model, "sdpa", graphs=graphs, num_blocks=num_blocks, budget=budget, seed=seed
    )
    out, rec, preemptions, eng = _drive(
        model, "triton", graphs=graphs, num_blocks=num_blocks, budget=budget, seed=seed
    )
    assert eng.runner.decode_attention == "triton"  # type: ignore[attr-defined]
    assert out == ref_out
    assert len(rec.steps) == len(ref.steps)
    worst = max(float((a - b).abs().max()) for a, b in zip(ref.steps, rec.steps, strict=True))
    assert worst < 1e-4, f"max logit difference {worst}"
    if graphs:
        summary = eng.runner.decode_graphs.summary()  # type: ignore[attr-defined]
        assert summary["replays"] > 0, summary
        wide = [g for g in summary["graphs"] if g["width_blocks"] == 160]
        assert wide, summary
        assert wide[0]["status"] == "captured", summary
    if num_blocks == 300:
        assert preemptions > 0, "preemption not exercised"


def test_auto_selects_triton_on_cuda(model: LlamaForCausalLM) -> None:
    eng = LLMEngine.for_model(model, num_blocks=64, block_size=4)
    assert eng.runner.decode_attention == "triton"  # type: ignore[attr-defined]
