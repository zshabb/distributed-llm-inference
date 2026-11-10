"""Content-addressed identities for full KV blocks.

A block's KV values depend on *every* token before it, not just its own tokens (attention mixes
the whole prefix into each layer's hidden states). Its identity is therefore a hash chain:

    h_0 = H(namespace, ZERO, tokens[0:B])
    h_i = H(namespace, h_{i-1}, tokens[i*B:(i+1)*B])

Two blocks with equal ``h_i`` hold identical KV (same model, same prefix) and can be shared.

Why SHA-256 rather than Python's ``hash``: a collision makes one request silently attend to another
request's cached keys and values, which is both a correctness bug and a cross-request data leak.
Python's tuple hash is 64-bit and not designed against adversarial inputs, and prefix caches keyed
on such hashes are a known vulnerability class. SHA-256 costs ~1 µs per 16-token block, which is
negligible next to a forward pass.

``namespace`` scopes sharing: it must at least identify the model and weights (and adapter). It can
also include a tenant id, to disable cross-tenant reuse. Prefix-cache hits are observable through
latency (a timing side channel), so multi-tenant deployments should not share prefixes across
tenants unless they accept that leak.
"""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Sequence

BlockHash = bytes
ROOT_PARENT: BlockHash = b"\x00" * 32


def block_hash(namespace: bytes, parent: BlockHash, tokens: Sequence[int]) -> BlockHash:
    """Hash one full block given its parent block's hash.

    Args:
        namespace: Opaque sharing scope (model/weights/adapter, optionally tenant).
        parent: Hash of the previous block, or :data:`ROOT_PARENT` for the first block.
        tokens: Token ids of this block (exactly ``block_size`` of them).

    Returns:
        32-byte digest.
    """
    h = hashlib.sha256()
    # Length-prefix the namespace so (ns="ab", tokens…) can never alias (ns="a", …).
    h.update(len(namespace).to_bytes(4, "little"))
    h.update(namespace)
    h.update(parent)
    h.update(array("q", tokens).tobytes())  # int64: no truncation for any vocabulary size
    return h.digest()


def chain_hashes(
    namespace: bytes, tokens: Sequence[int], block_size: int, parent: BlockHash = ROOT_PARENT
) -> list[BlockHash]:
    """Hashes of every *full* block in ``tokens`` (a trailing partial block is not hashed)."""
    hashes: list[BlockHash] = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = block_hash(namespace, parent, tokens[start : start + block_size])
        hashes.append(parent)
    return hashes
