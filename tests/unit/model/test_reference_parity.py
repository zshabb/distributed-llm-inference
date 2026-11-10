"""Numerical parity with Hugging Face `transformers` on tiny random models (fp32, CPU).

Tolerances: fp32 implementations that perform the same operations in the same order agree to
~1e-6. We allow 1e-4 (absolute, on logits of magnitude ~1) to absorb kernel-level reordering
inside SDPA while still catching any real bug (wrong RoPE layout, GQA grouping, norm placement,
which all produce errors of order 1e-1 or larger).
"""

from __future__ import annotations

import pytest
import torch

from dli.generate import greedy_generate
from dli.model import load_model
from dli.model.kv_cache import ContiguousKVCache

from .conftest import VARIANTS, TinyFactory

ATOL = 1e-4


def _hf_logits(ref: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out: torch.Tensor = ref(ids[None], use_cache=False).logits[0]
    return out


def _hf_greedy(ref: torch.nn.Module, prompt: list[int], steps: int) -> list[int]:
    """Reference greedy loop using the HF model's own KV cache (no EOS stopping)."""
    with torch.no_grad():
        out = ref(torch.tensor([prompt]), use_cache=True)
        past, tokens = out.past_key_values, []
        for _ in range(steps):
            nxt = int(out.logits[0, -1].argmax())
            tokens.append(nxt)
            out = ref(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
    return tokens


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_full_sequence_logits_match_reference(tiny_model: TinyFactory, variant: str) -> None:
    ref, ckpt = tiny_model(variant)
    ours = load_model(ckpt)
    ids = torch.randint(0, 97, (40,), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        got = ours(ids, torch.arange(40))
    torch.testing.assert_close(got, _hf_logits(ref, ids), atol=ATOL, rtol=0)


@pytest.mark.parametrize("variant", ["llama", "llama3-rope", "qwen2"])
def test_positions_beyond_original_context_match(tiny_model: TinyFactory, variant: str) -> None:
    # 200 tokens > original_max_position_embeddings=32 in the llama3 variant: exercises all
    # three frequency bands of the scaled RoPE.
    ref, ckpt = tiny_model(variant)
    ours = load_model(ckpt)
    ids = torch.randint(0, 97, (200,), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        got = ours(ids, torch.arange(200))
    torch.testing.assert_close(got, _hf_logits(ref, ids), atol=ATOL, rtol=0)


@pytest.mark.parametrize("variant", ["llama", "llama-tied", "qwen2"])
def test_greedy_generation_matches_reference_tokens(tiny_model: TinyFactory, variant: str) -> None:
    ref, ckpt = tiny_model(variant)
    ours = load_model(ckpt)
    prompt = [5, 17, 42, 8, 1, 66, 23]
    assert greedy_generate(ours, prompt, 48) == _hf_greedy(ref, prompt, 48)


@pytest.mark.parametrize("variant", ["llama", "qwen2"])
def test_kv_cache_path_equals_recompute_path(tiny_model: TinyFactory, variant: str) -> None:
    _, ckpt = tiny_model(variant)
    ours = load_model(ckpt)
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    assert greedy_generate(ours, prompt, 40) == greedy_generate(ours, prompt, 40, use_cache=False)


def test_chunked_prefill_into_nonempty_cache_matches_single_pass(tiny_model: TinyFactory) -> None:
    """Appending a multi-token chunk to a non-empty cache uses the explicit-mask branch."""
    _, ckpt = tiny_model("llama")
    ours = load_model(ckpt)
    cfg = ours.config
    ids = torch.randint(0, 97, (30,), generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        full = ours(ids, torch.arange(30))
        cache = ContiguousKVCache(
            cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, 30, torch.float32, torch.device("cpu")
        )
        first = ours(ids[:12], torch.arange(12), cache)
        second = ours(ids[12:], torch.arange(12, 30), cache)
    assert cache.length == 30
    torch.testing.assert_close(torch.cat([first, second]), full, atol=ATOL, rtol=0)


def test_eos_stops_generation_and_is_included(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    ours = load_model(ckpt)
    free_run = greedy_generate(ours, [1, 2, 3], 20)
    eos = free_run[4]
    stopped = greedy_generate(ours, [1, 2, 3], 20, eos_token_ids={eos})
    assert stopped == free_run[: free_run.index(eos) + 1]


def test_bfloat16_load_runs_and_tracks_fp32(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    fp32, bf16 = load_model(ckpt), load_model(ckpt, dtype=torch.bfloat16)
    ids = torch.arange(1, 25)
    with torch.no_grad():
        a = fp32(ids, torch.arange(24))
        b = bf16(ids, torch.arange(24)).float()
    assert b.dtype == torch.float32
    # bf16 has an 8-bit mantissa: loose bound, only catches gross dtype bugs.
    assert (a - b).abs().max().item() < 0.1
