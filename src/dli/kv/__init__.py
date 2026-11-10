"""Paged KV-cache memory management: block allocation, prefix caching, eviction, and storage."""

from dli.kv.block_manager import (
    Allocation,
    BlockManager,
    BlockManagerStats,
    CopyOp,
    NoFreeBlocksError,
    SequenceError,
)
from dli.kv.eviction import CostAwarePolicy, EvictionPolicy, LRUPolicy, make_policy

__all__ = [
    "Allocation",
    "BlockManager",
    "BlockManagerStats",
    "CopyOp",
    "CostAwarePolicy",
    "EvictionPolicy",
    "LRUPolicy",
    "NoFreeBlocksError",
    "SequenceError",
    "make_policy",
]
