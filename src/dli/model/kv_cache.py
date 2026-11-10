"""Contiguous per-sequence KV cache (M1 bridge).

This is intentionally the simplest correct cache: one preallocated
``[kv_heads, max_len, head_dim]`` buffer per layer for a single sequence. It exists so M1 can
validate the model's incremental-decoding math against a reference. It is NOT the serving design:
it reserves memory for ``max_len`` tokens up front (the fragmentation problem the paged block
manager in ``dli.kv`` solves in M2) and cannot be shared between sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class KVCacheView(Protocol):
    """What the model needs from a KV cache for ONE sequence.

    Implemented by :class:`ContiguousKVCache` (M1) and ``dli.kv.paged_store.PagedSequenceView``
    (M2). The model only appends at ``length`` and reads the whole prefix back, so storage layout,
    paging, and sharing stay invisible to the attention math.
    """

    @property
    def length(self) -> int:
        """Tokens whose K/V are written for all layers."""
        ...

    def write(
        self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store ``[kv_heads, n, head_dim]`` K/V at positions ``[start, start+n)``.

        Returns:
            K and V for positions ``[0, start+n)`` as ``[kv_heads, start+n, head_dim]``.
        """
        ...

    def advance(self, n: int) -> None:
        """Commit ``n`` tokens after every layer has written them."""
        ...


class ContiguousKVCache:
    """KV storage for one sequence across all layers.

    Thread-safety: none; owned by a single generation loop.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (num_layers, num_kv_heads, max_len, head_dim)
        self._k = torch.empty(shape, dtype=dtype, device=device)
        self._v = torch.empty(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.length = 0  # tokens whose K/V are fully written for ALL layers

    def write(
        self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store ``k``/``v`` (``[kv_heads, n, head_dim]``) at ``[start, start+n)``.

        Returns:
            Views of all cached keys and values for positions ``[0, start+n)`` of this layer.

        Raises:
            ValueError: The write would exceed ``max_len`` or leave a gap.
        """
        n = k.shape[1]
        end = start + n
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > max_len {self.max_len}")
        if start > self.length:
            raise ValueError(f"non-contiguous write at {start}; cache holds {self.length}")
        self._k[layer, :, start:end] = k
        self._v[layer, :, start:end] = v
        return self._k[layer, :, :end], self._v[layer, :, :end]

    def advance(self, n: int) -> None:
        """Commit ``n`` new tokens after every layer has written them."""
        self.length += n


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of tokens from ONE sequence inside a flattened batch.

    Attributes:
        offset: Index of the segment's first token in the flattened batch.
        length: Number of tokens in the segment.
        cache: That sequence's KV view (``None`` = no cache: attend within the segment only).
    """

    offset: int
    length: int
    cache: KVCacheView | None


@dataclass(frozen=True, slots=True)
class BatchedKV:
    """Many sequences in one forward pass (continuous batching, M3).

    Dense layers run once over all tokens; attention runs per segment against that sequence's own
    KV. Segments must tile the batch in order without gaps.
    """

    segments: tuple[Segment, ...]

    def validate(self, num_tokens: int) -> None:
        expected = 0
        for seg in self.segments:
            if seg.offset != expected or seg.length <= 0:
                raise ValueError("segments must tile the batch contiguously with positive lengths")
            expected += seg.length
        if expected != num_tokens:
            raise ValueError(f"segments cover {expected} tokens, batch has {num_tokens}")

    def advance(self) -> None:
        for seg in self.segments:
            if seg.cache is not None:
                seg.cache.advance(seg.length)
