"""Paged KV end to end on tiny models: must reproduce the contiguous-cache path exactly.

The contiguous path is itself verified against `transformers` (M1), so equality here transfers
that evidence to paged storage, prefix reuse, sharing, copy-on-write, and eviction.
"""

from __future__ import annotations

import pytest
import torch

from dli.generate import PagedSequence, greedy_generate, paged_greedy_generate
from dli.kv import BlockManager
from dli.kv.paged_store import PagedKVStore
from dli.model import LlamaForCausalLM, load_model

from .conftest import TinyFactory

PROMPT = [5, 17, 42, 8, 1, 66, 23, 9, 30, 2, 11]  # 11 tokens: not a multiple of most block sizes


def _setup(
    model: LlamaForCausalLM, num_blocks: int, block_size: int, policy: str = "lru"
) -> tuple[BlockManager, PagedKVStore]:
    bm = BlockManager(num_blocks, block_size, eviction_policy=policy, watermark=0.0)
    return bm, PagedKVStore.for_model(model.config, num_blocks, block_size)


@pytest.mark.parametrize("variant", ["llama", "qwen2"])
@pytest.mark.parametrize("block_size", [1, 3, 16])
def test_paged_equals_contiguous(tiny_model: TinyFactory, variant: str, block_size: int) -> None:
    _, ckpt = tiny_model(variant)
    model = load_model(ckpt)
    bm, store = _setup(model, num_blocks=64, block_size=block_size)
    got = paged_greedy_generate(model, bm, store, "s", PROMPT, 30)
    assert got.tokens == greedy_generate(model, PROMPT, 30)
    bm.check_invariants()


def test_prefix_reuse_computes_fewer_tokens_with_identical_output(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    bm, store = _setup(model, num_blocks=64, block_size=4)
    first = paged_greedy_generate(model, bm, store, "a", PROMPT, 12)
    assert first.num_cached_prompt_tokens == 0
    longer = [*PROMPT, 3, 3, 7]
    second = paged_greedy_generate(model, bm, store, "b", longer, 12)
    assert second.num_cached_prompt_tokens == 8  # two full blocks of the shared prefix
    assert second.tokens == greedy_generate(model, longer, 12)
    bm.check_invariants()


def test_live_sequences_sharing_prefix_blocks_do_not_disturb_each_other(
    tiny_model: TinyFactory,
) -> None:
    _, ckpt = tiny_model("qwen2")
    model = load_model(ckpt)
    bm, store = _setup(model, num_blocks=64, block_size=4)
    paged_greedy_generate(model, bm, store, "warm", PROMPT, 1)  # caches PROMPT's full blocks
    prompts = {"a": [*PROMPT, 1, 2], "b": [*PROMPT, 40, 41, 42]}
    seqs, logits, outs = {}, {}, {k: [] for k in prompts}
    for sid, p in prompts.items():
        assert bm.allocate(sid, p).num_cached_tokens == 8
        seqs[sid] = PagedSequence(model, bm, store, sid)
        logits[sid] = seqs[sid].prefill(p)
    assert bm.block_table("a")[:2] == bm.block_table("b")[:2]  # physically shared blocks
    for _ in range(15):  # interleaved decoding, like a batched engine step
        for sid in prompts:
            nxt = int(torch.argmax(logits[sid]))
            outs[sid].append(nxt)
            logits[sid] = seqs[sid].feed(nxt)
    for sid, p in prompts.items():
        assert outs[sid] == greedy_generate(model, p, 15)
    bm.check_invariants()


def test_fork_with_copy_on_write_matches_independent_runs(tiny_model: TinyFactory) -> None:
    """Teacher-force different continuations into a parent and its fork; compare every logit."""
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    bm, store = _setup(model, num_blocks=64, block_size=4)
    bm.allocate("p", PROMPT)  # 11 tokens: the last block is partial, so the fork shares it
    parent = PagedSequence(model, bm, store, "p")
    parent.prefill(PROMPT)
    bm.fork("p", "c")
    child = PagedSequence(model, bm, store, "c")
    cont = {"p": [7, 7, 7, 7, 7, 7], "c": [50, 51, 52, 53, 54, 55]}
    for step in range(6):
        for sid, seq in (("p", parent), ("c", child)):
            got = seq.feed(cont[sid][step])
            full = PROMPT + cont[sid][: step + 1]
            with torch.no_grad():
                ref = model(torch.tensor(full), torch.arange(len(full)), last_token_only=True)[-1]
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=0)
    assert bm.stats().cow_copies == 1
    bm.check_invariants()


@pytest.mark.parametrize("policy", ["lru", "cost"])
def test_correct_output_under_eviction_pressure(tiny_model: TinyFactory, policy: str) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    bm, store = _setup(model, num_blocks=12, block_size=4, policy=policy)  # tiny pool: forces reuse
    for i in range(8):
        prompt = [i % 3, *PROMPT[: 4 + i % 5], 80 + i]  # ids < vocab_size (97)
        got = paged_greedy_generate(model, bm, store, f"r{i}", prompt, 20)
        assert got.tokens == greedy_generate(model, prompt, 20), f"request {i} wrong after reuse"
    assert bm.stats().evictions > 0
    bm.check_invariants()
