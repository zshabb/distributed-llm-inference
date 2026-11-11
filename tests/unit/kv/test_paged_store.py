from __future__ import annotations

import pytest
import torch

from dli.kv import BlockManager, CopyOp
from dli.kv.paged_store import PagedKVStore, PagedSequenceView
from dli.model.config import ModelConfig, RopeConfig


def _cfg() -> ModelConfig:
    return ModelConfig(
        model_type="llama",
        vocab_size=10,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        rope=RopeConfig(theta=10000.0),
        tie_word_embeddings=False,
        qkv_bias=False,
        o_bias=False,
        mlp_bias=False,
    )


def test_write_then_gather_round_trips_through_scattered_blocks() -> None:
    store = PagedKVStore(1, num_blocks=8, block_size=3, num_kv_heads=2, head_dim=4)
    table = torch.tensor([5, 0, 6])  # deliberately non-contiguous, unordered
    n = 7
    k = torch.randn(n, 2, 4)
    v = torch.randn(n, 2, 4)
    slots = torch.tensor([int(table[p // 3]) * 3 + p % 3 for p in range(n)])
    store.write(0, slots, k, v)
    gk, gv = store.gather(0, table, n)
    torch.testing.assert_close(gk, k.transpose(0, 1))
    torch.testing.assert_close(gv, v.transpose(0, 1))


def test_copy_blocks_copies_all_layers() -> None:
    store = PagedKVStore(3, num_blocks=4, block_size=2, num_kv_heads=1, head_dim=2)
    store.k.normal_()
    store.v.normal_()
    store.copy_blocks([CopyOp(src=1, dst=3)])
    torch.testing.assert_close(store.k[:, 3], store.k[:, 1])
    torch.testing.assert_close(store.v[:, 3], store.v[:, 1])
    store.copy_blocks([])  # no-op


def test_memory_sizing() -> None:
    cfg = _cfg()
    per_block = PagedKVStore.bytes_per_block(cfg, block_size=16, dtype=torch.bfloat16)
    assert per_block == 2 * 2 * 16 * 2 * 4 * 2
    assert PagedKVStore.num_blocks_for_budget(cfg, 16, torch.bfloat16, 10 * per_block + 1) == 10


def test_view_rejects_out_of_order_writes_and_mismatched_block_size() -> None:
    cfg = _cfg()
    bm = BlockManager(num_blocks=4, block_size=2, watermark=0.0)
    bm.allocate("s", [1, 2, 3])
    view = PagedSequenceView(PagedKVStore.for_model(cfg, 4, 2), bm, "s")
    kv = torch.zeros(2, 1, 4)
    with pytest.raises(ValueError, match="write at"):
        view.write(0, 1, kv, kv)
    with pytest.raises(ValueError, match="block sizes"):
        PagedSequenceView(PagedKVStore.for_model(cfg, 4, 3), bm, "s")
