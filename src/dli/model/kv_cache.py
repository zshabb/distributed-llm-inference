"""Contiguous per-sequence KV cache (M1 bridge).

This is intentionally the simplest correct cache: one preallocated
``[kv_heads, max_len, head_dim]`` buffer per layer for a single sequence. It exists so M1 can
validate the model's incremental-decoding math against a reference. It is NOT the serving design:
it reserves memory for ``max_len`` tokens up front (the fragmentation problem the paged block
manager in ``dli.kv`` solves in M2) and cannot be shared between sequences.
"""

from __future__ import annotations

import torch


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
