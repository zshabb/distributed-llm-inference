"""LLMEngine with the real ModelRunner on tiny checkpoints vs the unbatched M1 path.

The unbatched contiguous path is tied to `transformers` (M1). Equality here covers batched
attention, paged KV with prefix reuse, chunked prefill, and preemption-recompute on real tensors.
"""

from __future__ import annotations

import random

import pytest

from dli.engine import LLMEngine, SchedulerConfig
from dli.generate import greedy_generate
from dli.model import load_model
from dli.sampling import SamplingParams

from .conftest import TinyFactory


def _prompts(n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(97) for _ in range(rng.randrange(1, 40))] for _ in range(n)]


@pytest.mark.parametrize(
    ("variant", "num_blocks", "budget"),
    [
        ("llama", 256, 512),  # roomy: pure continuous batching
        ("qwen2", 256, 16),  # tiny budget: heavy chunked prefill
        ("llama", 24, 64),  # tight memory: preemption + recompute
    ],
)
def test_engine_greedy_equals_unbatched(
    tiny_model: TinyFactory, variant: str, num_blocks: int, budget: int
) -> None:
    _, ckpt = tiny_model(variant)
    model = load_model(ckpt)
    eng = LLMEngine.for_model(
        model,
        num_blocks=num_blocks,
        block_size=4,
        config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=budget),
    )
    prompts = _prompts(12, seed=num_blocks + budget)
    max_tokens = [5 + (i * 7) % 20 for i in range(12)]
    reqs = [
        eng.add_request(f"r{i}", p, SamplingParams(max_tokens=m))
        for i, (p, m) in enumerate(zip(prompts, max_tokens, strict=True))
    ]
    while eng.has_unfinished():
        eng.step()
    for r, p, m in zip(reqs, prompts, max_tokens, strict=True):
        assert r.output_ids == greedy_generate(model, p, m), r.request_id
    if num_blocks == 24:
        assert sum(r.num_preemptions for r in reqs) > 0, "preemption not exercised"
    eng.bm.check_invariants()


def test_shared_prefixes_and_cancellation_on_real_kv(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    eng = LLMEngine.for_model(model, num_blocks=128, block_size=4)
    system = [11, 22, 33, 44, 55, 66, 77, 88, 5]
    # The first request computes the system prompt; later arrivals must reuse its blocks. (Requests
    # admitted in the SAME step cannot share: blocks become shareable only once computed, rule 2.)
    keep = [eng.add_request("k0", [*system, 0], SamplingParams(max_tokens=12))]
    eng.step()
    keep += [
        eng.add_request(f"k{i}", [*system, i], SamplingParams(max_tokens=12)) for i in (1, 2, 3)
    ]
    drop = [
        eng.add_request(f"d{i}", [*system, 50 + i], SamplingParams(max_tokens=40)) for i in range(3)
    ]
    for _ in range(4):
        eng.step()
    for r in drop:
        assert eng.abort(r.request_id)
    while eng.has_unfinished():
        eng.step()
    for i, r in enumerate(keep):
        assert r.output_ids == greedy_generate(model, [*system, i], 12)
    assert eng.bm.stats().prompt_tokens_hit > 0, "shared system prompt was not reused"
    eng.bm.reset_prefix_cache()
    eng.bm.check_invariants()
    assert eng.bm.stats().free_blocks == 128


def test_seeded_sampling_is_reproducible_across_batch_compositions(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("qwen2")
    model = load_model(ckpt)
    params = SamplingParams(max_tokens=15, temperature=0.8, top_p=0.9, seed=1234)
    target = [3, 1, 4, 1, 5, 9, 2, 6]

    alone = LLMEngine.for_model(model, num_blocks=64, block_size=4)
    r_alone = alone.add_request("t", target, params)
    while alone.has_unfinished():
        alone.step()

    busy = LLMEngine.for_model(model, num_blocks=64, block_size=4)
    for i, p in enumerate(_prompts(6, seed=9)):
        busy.add_request(f"n{i}", p, SamplingParams(max_tokens=10, temperature=1.0, seed=i))
    r_busy = busy.add_request("t", target, params)
    while busy.has_unfinished():
        busy.step()
    # Same seed, same logits up to fp32 batch-reordering noise -> same tokens.
    assert r_busy.output_ids == r_alone.output_ids
