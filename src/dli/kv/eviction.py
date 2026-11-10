"""Eviction policies for cached-but-unused KV blocks.

The block manager only offers *candidates* to a policy: cached blocks with no active user
(``ref_count == 0``) and no cached child. Leaf-only eviction keeps the cache prefix-closed: a cached
block is reachable by a lookup only if all its ancestors are cached, so evicting an interior block
would strand its descendants as unreachable memory.

Two policies:

* :class:`LRUPolicy`: evict the candidate that became unused longest ago. Strong baseline and
  the common default in serving engines.
* :class:`CostAwarePolicy`: GreedyDual-Size-Frequency (Cherkasova, 1998) adapted to KV blocks.
  Priority ``H = L + freq * cost(depth) / size``, with ``size = 1`` (all blocks equal) and ``cost``
  growing with depth (a deeper block's prefill attends over a longer prefix). ``L`` is an inflation
  value set to the priority of the last victim, which ages entries that stop being hit so a
  once-popular block cannot pin memory forever. It targets workloads where a few shared prefixes
  (system prompts, few-shot templates) are reused far more often than one-off conversation tails,
  which LRU evicts as soon as a burst of unique requests passes through.

Complexity: LRU is O(1) per operation; CostAware is O(log n) with lazy heap deletion (stale entries
are skipped at pop time, bounded by the number of updates).
"""

from __future__ import annotations

import heapq
import itertools
from collections import OrderedDict
from typing import Protocol


class EvictionPolicy(Protocol):
    """Callbacks the block manager issues; the policy chooses victims among candidates."""

    name: str

    def on_cached(self, block_id: int, depth: int) -> None:
        """A block was registered in the prefix index at chain ``depth`` (0 = first block)."""

    def on_hit(self, block_id: int) -> None:
        """A cached block was reused by a prefix-cache hit."""

    def add_candidate(self, block_id: int) -> None:
        """The block became evictable (unused and a leaf)."""

    def remove_candidate(self, block_id: int) -> None:
        """The block stopped being evictable (reused, or gained a cached child)."""

    def pop_victim(self) -> int:
        """Remove and return the block to evict. Raises ``LookupError`` when there is none."""

    def on_evicted(self, block_id: int) -> None:
        """Forget all metadata for an evicted block."""

    def __len__(self) -> int:
        """Number of current candidates."""


class LRUPolicy:
    """Least-recently-released first."""

    name = "lru"

    def __init__(self) -> None:
        self._candidates: OrderedDict[int, None] = OrderedDict()

    def on_cached(self, block_id: int, depth: int) -> None:
        pass

    def on_hit(self, block_id: int) -> None:
        pass  # recency is refreshed when the block is released again (add_candidate)

    def add_candidate(self, block_id: int) -> None:
        self._candidates[block_id] = None
        self._candidates.move_to_end(block_id)

    def remove_candidate(self, block_id: int) -> None:
        self._candidates.pop(block_id, None)

    def pop_victim(self) -> int:
        if not self._candidates:
            raise LookupError("no evictable blocks")
        block_id, _ = self._candidates.popitem(last=False)
        return block_id

    def on_evicted(self, block_id: int) -> None:
        pass

    def __len__(self) -> int:
        return len(self._candidates)


class CostAwarePolicy:
    """GreedyDual-Size-Frequency over KV blocks (see module docstring).

    Args:
        depth_weight: ``cost(depth) = 1 + depth_weight * depth``. 0 makes cost uniform, leaving
            frequency + aging only. The default is a modest preference for keeping deep blocks; the
            right value depends on the attention/MLP FLOP ratio of the model and is a tuning knob,
            not a derived constant.
    """

    name = "cost"

    def __init__(self, depth_weight: float = 0.05) -> None:
        if depth_weight < 0:
            raise ValueError("depth_weight must be >= 0")
        self._depth_weight = depth_weight
        self._inflation = 0.0  # L
        self._freq: dict[int, int] = {}
        self._depth: dict[int, int] = {}
        self._heap: list[tuple[float, int, int]] = []  # (priority, tie-break, block_id)
        self._live: dict[int, tuple[float, int]] = {}  # block_id -> (priority, tie) of valid entry
        self._counter = itertools.count()

    def priority(self, block_id: int) -> float:
        """Current GDSF priority the block would get if it became a candidate now."""
        cost = 1.0 + self._depth_weight * self._depth.get(block_id, 0)
        return self._inflation + self._freq.get(block_id, 1) * cost

    def on_cached(self, block_id: int, depth: int) -> None:
        self._freq[block_id] = 1
        self._depth[block_id] = depth

    def on_hit(self, block_id: int) -> None:
        self._freq[block_id] = self._freq.get(block_id, 1) + 1

    def add_candidate(self, block_id: int) -> None:
        tie = next(self._counter)  # FIFO among equal priorities -> older first
        prio = self.priority(block_id)
        self._live[block_id] = (prio, tie)
        heapq.heappush(self._heap, (prio, tie, block_id))
        self._maybe_compact()

    def remove_candidate(self, block_id: int) -> None:
        self._live.pop(block_id, None)  # heap entry becomes stale; skipped at pop time

    def pop_victim(self) -> int:
        while self._heap:
            prio, tie, block_id = heapq.heappop(self._heap)
            live = self._live.get(block_id)
            if live is not None and live[1] == tie:
                del self._live[block_id]
                self._inflation = prio  # GDSF aging: future entries start above the last victim
                return block_id
        raise LookupError("no evictable blocks")

    def heap_size(self) -> int:
        """Heap entries including stale ones (exposed for the bounded-memory test)."""
        return len(self._heap)

    def _maybe_compact(self) -> None:
        # Hot blocks cycle between in-use and candidate without ever being popped, leaving a
        # stale entry per cycle. Rebuild when stale entries dominate so memory stays O(candidates).
        if len(self._heap) > 2 * len(self._live) + 64:
            self._heap = [(p, t, b) for b, (p, t) in self._live.items()]
            heapq.heapify(self._heap)

    def on_evicted(self, block_id: int) -> None:
        self._freq.pop(block_id, None)
        self._depth.pop(block_id, None)

    def __len__(self) -> int:
        return len(self._live)


def make_policy(name: str) -> EvictionPolicy:
    """Factory used by configuration: ``"lru"`` or ``"cost"``."""
    if name == "lru":
        return LRUPolicy()
    if name == "cost":
        return CostAwarePolicy()
    raise ValueError(f"unknown eviction policy {name!r}")
