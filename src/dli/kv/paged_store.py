"""Physical paged KV storage and the per-sequence view the model writes through.

Layout per layer: ``K[layer]`` and ``V[layer]`` are
``[num_blocks, block_size, kv_heads, head_dim]``, one contiguous tensor each, viewed flat as
``[num_blocks * block_size, kv_heads, head_dim]`` for slot-indexed writes.
Slot ``s = block_id * block_size + offset`` comes from
:meth:`BlockManager.slot_mapping`, so the block manager decides *where* KV lives and this module
only moves bytes.

Reading (M2 reference path): attention needs the sequence's K/V in position order, so
:meth:`PagedKVStore.gather` indexes the blocks of the block table and trims to the sequence length.
This is correct but copies the whole prefix for every layer and every decode step
(O(context) memory traffic per token). The M6 Triton paged-attention kernel removes the copy by
reading blocks in place. Until then, paged decoding is *slower* than the contiguous cache, by
design and as recorded in docs/known-issues.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from dli.kv.block_manager import BlockManager, CopyOp, SeqId
from dli.model.config import ModelConfig


class PagedKVStore:
    """Preallocated K/V block pool for all layers.

    Args:
        num_layers, num_blocks, block_size, num_kv_heads, head_dim: Pool geometry.
        dtype, device: Storage dtype/device (must match the model's activations).

    Memory: ``2 * num_layers * num_blocks * block_size * num_kv_heads * head_dim * itemsize`` bytes,
    allocated once at construction so serving never allocates KV memory on the hot path.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    @classmethod
    def for_model(
        cls,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> PagedKVStore:
        return cls(
            cfg.num_layers,
            num_blocks,
            block_size,
            cfg.num_kv_heads,
            cfg.head_dim,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def bytes_per_block(cfg: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
        """KV bytes one block occupies across all layers (K and V)."""
        itemsize = torch.empty((), dtype=dtype).element_size()
        return 2 * cfg.num_layers * block_size * cfg.num_kv_heads * cfg.head_dim * itemsize

    @classmethod
    def num_blocks_for_budget(
        cls, cfg: ModelConfig, block_size: int, dtype: torch.dtype, budget_bytes: int
    ) -> int:
        """Largest pool that fits in ``budget_bytes`` (e.g. free GPU memory after weights)."""
        return budget_bytes // cls.bytes_per_block(cfg, block_size, dtype)

    def _flat(self, t: torch.Tensor, layer: int) -> torch.Tensor:
        return t[layer].view(-1, self.num_kv_heads, self.head_dim)

    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter ``[n, kv_heads, head_dim]`` K/V rows into flat ``slots`` (int64 ``[n]``)."""
        self._flat(self.k, layer).index_copy_(0, slots, k)
        self._flat(self.v, layer).index_copy_(0, slots, v)

    def gather(
        self, layer: int, block_table: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K/V of one sequence in position order, ``[kv_heads, num_tokens, head_dim]`` each."""

        def take(t: torch.Tensor) -> torch.Tensor:
            pages = t[layer].index_select(0, block_table)  # [nb, bs, kvh, d]
            return pages.reshape(-1, self.num_kv_heads, self.head_dim)[:num_tokens].transpose(0, 1)

        return take(self.k), take(self.v)

    def copy_blocks(self, ops: Iterable[CopyOp]) -> None:
        """Apply copy-on-write operations returned by the block manager (all layers at once)."""
        ops = list(ops)
        if not ops:
            return
        src = torch.tensor([o.src for o in ops], device=self.k.device)
        dst = torch.tensor([o.dst for o in ops], device=self.k.device)
        self.k[:, dst] = self.k[:, src]
        self.v[:, dst] = self.v[:, src]


class PagedSequenceView:
    """Adapter giving the model a :class:`~dli.model.kv_cache.KVCacheView` over paged storage.

    The block manager must already have reserved slots (``allocate`` / ``append_slots``) for every
    position the model writes. :meth:`advance` reports progress to the manager
    (``mark_computed``), which is what makes newly completed blocks eligible for prefix sharing.
    """

    def __init__(self, store: PagedKVStore, manager: BlockManager, seq_id: SeqId) -> None:
        if store.block_size != manager.block_size:
            raise ValueError("store and manager block sizes differ")
        self._store = store
        self._manager = manager
        self._seq_id = seq_id
        self._length = manager.num_computed_tokens(seq_id)
        self._slots_key: tuple[int, int] | None = None
        self._slots: torch.Tensor | None = None
        self._table: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return self._length

    def write(
        self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = k.shape[1]
        if start != self._length:
            raise ValueError(f"write at {start}, but {self._length} tokens are computed")
        if self._slots_key != (start, n):
            # Same slots and table for every layer of one forward pass: compute once per pass.
            dev = self._store.k.device
            self._slots = torch.tensor(
                self._manager.slot_mapping(self._seq_id, start, start + n), device=dev
            )
            table: Sequence[int] = self._manager.block_table(self._seq_id)
            self._table = torch.tensor(table, device=dev)
            self._slots_key = (start, n)
        assert self._slots is not None
        assert self._table is not None
        self._store.write(layer, self._slots, k.transpose(0, 1), v.transpose(0, 1))
        return self._store.gather(layer, self._table, start + n)

    def advance(self, n: int) -> None:
        self._length += n
        self._manager.mark_computed(self._seq_id, self._length)
        self._slots_key = None
