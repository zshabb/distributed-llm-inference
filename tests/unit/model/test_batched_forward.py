"""One forward pass carrying several sequences must equal separate per-sequence passes."""

from __future__ import annotations

import pytest
import torch

from dli.model import load_model
from dli.model.kv_cache import BatchedKV, ContiguousKVCache, Segment

from .conftest import TinyFactory

ATOL = 1e-4  # same bound as the M1 tiny-model parity: batched matmuls may reorder fp32 sums


def _cache(model: torch.nn.Module) -> ContiguousKVCache:
    cfg = model.config
    return ContiguousKVCache(
        cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, 64, torch.float32, torch.device("cpu")
    )


@pytest.mark.parametrize("variant", ["llama", "qwen2", "llama-tied"])
def test_mixed_prefill_chunk_and_decode_batch_equals_separate_passes(
    tiny_model: TinyFactory, variant: str
) -> None:
    _, ckpt = tiny_model(variant)
    model = load_model(ckpt)
    g = torch.Generator().manual_seed(5)
    a = torch.randint(0, 97, (9,), generator=g)  # fresh prompt
    b = torch.randint(0, 97, (14,), generator=g)  # 8 cached + 6-token chunk
    c = torch.randint(0, 97, (5,), generator=g)  # 4 cached + 1 decode token

    # Reference: each sequence on its own (prefixes computed first where needed).
    refs, caches = [], []
    with torch.no_grad():
        refs.append(model(a, torch.arange(9), _cache(model)))
        cb, cc = _cache(model), _cache(model)
        model(b[:8], torch.arange(8), cb)
        refs.append(model(b[8:], torch.arange(8, 14), cb))
        model(c[:4], torch.arange(4), cc)
        refs.append(model(c[4:], torch.arange(4, 5), cc))

        # Batched: same three chunks in one flattened pass, against fresh copies of the prefixes.
        ca, cb2, cc2 = _cache(model), _cache(model), _cache(model)
        model(b[:8], torch.arange(8), cb2)
        model(c[:4], torch.arange(4), cc2)
        caches = [ca, cb2, cc2]
        ids = torch.cat([a, b[8:], c[4:]])
        pos = torch.cat([torch.arange(9), torch.arange(8, 14), torch.arange(4, 5)])
        batch = BatchedKV((Segment(0, 9, ca), Segment(9, 6, cb2), Segment(15, 1, cc2)))
        got = model(ids, pos, batch)

    torch.testing.assert_close(got, torch.cat(refs), atol=ATOL, rtol=0)
    assert [c.length for c in caches] == [9, 14, 5], "every segment's cache must advance"


def test_logits_indices_select_requested_rows(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    ids = torch.arange(1, 13)
    with torch.no_grad():
        full = model(ids, torch.arange(12))
        picked = model(ids, torch.arange(12), logits_indices=torch.tensor([3, 11]))
    torch.testing.assert_close(picked, full[[3, 11]], atol=ATOL, rtol=0)


def test_segments_must_tile_the_batch(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    ids = torch.arange(1, 7)
    with pytest.raises(ValueError, match="tile"):
        model(ids, torch.arange(6), BatchedKV((Segment(0, 2, None), Segment(3, 3, None))))
    with pytest.raises(ValueError, match="cover"):
        model(ids, torch.arange(6), BatchedKV((Segment(0, 2, None),)))
