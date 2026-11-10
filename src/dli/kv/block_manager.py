"""Paged KV-cache block manager with reference counting, copy-on-write, and prefix caching.

The GPU (or CPU) KV memory is carved into ``num_blocks`` fixed-size blocks of ``block_size`` token
slots each. Every sequence owns a *block table* (a list of block ids), so its KV cache does not
need to be contiguous. Memory is allocated a block at a time as the sequence grows. That removes the
up-front ``max_len`` reservation of a contiguous cache and bounds internal fragmentation to fewer
than ``block_size`` slots per sequence. Design doc: ``docs/algorithms/paged-kv-cache.md``.

Every block is in exactly one state:

* **FREE**: ``ref == 0``, not cached, on the free list.
* **USED**: ``ref > 0``, referenced by ``ref`` block-table entries (shared prefixes, forks).
* **CACHED-IDLE**: ``ref == 0`` but registered in the prefix index. Its KV is still valid and
  can be revived by a later prefix hit, or evicted (leaf-first, by the eviction policy) when memory
  is needed. A cached-idle block with no cached child is an eviction *candidate*.

Prefix caching rules (each one prevents a concrete bug):

1. Only **full** blocks are cached. A partial block's contents are still changing.
2. A block is registered only after its KV has been **computed** (:meth:`mark_computed`), never
   at allocation. Otherwise a concurrent request could hit the block and read unwritten memory.
3. A prompt can never be served entirely from cache: the block containing the prompt's last token is
   always recomputed, because the forward pass must run on at least one token to produce logits.
4. The index is **prefix-closed**: a block is registered only if its parent hash is registered,
   and only leaves are evicted. Every cached block is therefore reachable by a lookup walk from the
   root.
5. Shared blocks are never written. Appending into a partial block that a fork also references
   copies it first (copy-on-write), and the caller performs the returned :class:`CopyOp` on the KV
   tensors before the next forward pass.

Concurrency: **thread-confined**. The manager belongs to the engine's scheduler loop, which is the
only writer of KV metadata (ADR-0001). It is not internally locked, because a lock on every token
append would be pure overhead in that design. With ``debug_checks=True`` (default) every public call
verifies it runs on the owning thread, turning accidental cross-thread use into an immediate error
instead of silent corruption.

Failure atomicity: an operation that cannot get enough blocks raises :class:`NoFreeBlocksError`
**before** mutating any state, so the scheduler can preempt or retry without leaking blocks.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from dli.kv.eviction import EvictionPolicy, make_policy
from dli.kv.hashing import ROOT_PARENT, BlockHash, block_hash, chain_hashes

SeqId = int | str


class NoFreeBlocksError(RuntimeError):
    """Not enough free or evictable blocks; no state was changed."""


class SequenceError(KeyError):
    """Unknown or duplicate sequence id."""


@dataclass(frozen=True, slots=True)
class CopyOp:
    """Copy KV of block ``src`` into block ``dst`` (all layers) before the next forward pass."""

    src: int
    dst: int


@dataclass(frozen=True, slots=True)
class Allocation:
    """Result of admitting a prompt.

    Attributes:
        block_table: Block ids covering the prompt.
        num_cached_tokens: Leading prompt tokens whose KV is already present (prefix hit). The
            forward pass only needs to compute ``prompt[num_cached_tokens:]``.
    """

    block_table: tuple[int, ...]
    num_cached_tokens: int


@dataclass(slots=True)
class BlockManagerStats:
    """Counters since construction (snapshot via :meth:`BlockManager.stats`)."""

    num_blocks: int = 0
    free_blocks: int = 0
    cached_idle_blocks: int = 0
    used_blocks: int = 0
    prompt_tokens_queried: int = 0
    prompt_tokens_hit: int = 0
    evictions: int = 0
    cow_copies: int = 0
    duplicate_registrations: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of queried prompt tokens served from the prefix cache."""
        return (
            self.prompt_tokens_hit / self.prompt_tokens_queried
            if self.prompt_tokens_queried
            else 0.0
        )


@dataclass(slots=True)
class _Cached:
    block_hash: BlockHash
    parent: int | None
    depth: int
    children: int = 0


@dataclass(slots=True)
class _Seq:
    blocks: list[int]
    num_tokens: int
    num_computed: int
    namespace: bytes
    # Hashes of the leading full blocks known so far, and the tokens after them. Only the
    # chain tip and the unhashed tail are kept, not the whole token history.
    block_hashes: list[BlockHash] = field(default_factory=list)
    tail_tokens: list[int] = field(default_factory=list)
    num_registered: int = 0  # leading blocks already registered or obtained from the cache


class BlockManager:
    """Allocates KV blocks to sequences and maintains the prefix cache.

    Args:
        num_blocks: Total blocks in the pool.
        block_size: Token slots per block.
        eviction_policy: ``"lru"``, ``"cost"``, or an :class:`EvictionPolicy` instance.
        enable_prefix_caching: If False, nothing is hashed or cached and freed blocks go straight
            to the free list.
        watermark: Fraction of the pool kept in reserve when admitting *new* prompts, so running
            sequences can still grow by a block without immediate preemption.
        debug_checks: Enforce thread confinement on every public call.

    Thread-safety: not thread-safe; confined to the creating thread (see module docstring).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        eviction_policy: str | EvictionPolicy = "lru",
        enable_prefix_caching: bool = True,
        watermark: float = 0.01,
        debug_checks: bool = True,
    ) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        if not 0.0 <= watermark < 1.0:
            raise ValueError("watermark must be in [0, 1)")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.policy: EvictionPolicy = (
            make_policy(eviction_policy) if isinstance(eviction_policy, str) else eviction_policy
        )
        self._watermark_blocks = int(watermark * num_blocks)
        self._debug = debug_checks
        self._owner = threading.get_ident()

        self._ref = [0] * num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._cached: dict[int, _Cached] = {}
        self._by_hash: dict[BlockHash, int] = {}
        self._num_cached_idle = 0
        self._seqs: dict[SeqId, _Seq] = {}
        self._stats = BlockManagerStats(num_blocks=num_blocks)

    # ------------------------------------------------------------------ queries

    @property
    def num_available_blocks(self) -> int:
        """Blocks obtainable right now: free plus evictable cached-idle blocks."""
        return len(self._free) + self._num_cached_idle

    def has_sequence(self, seq_id: SeqId) -> bool:
        return seq_id in self._seqs

    def block_table(self, seq_id: SeqId) -> tuple[int, ...]:
        return tuple(self._seq(seq_id).blocks)

    def num_tokens(self, seq_id: SeqId) -> int:
        return self._seq(seq_id).num_tokens

    def num_computed_tokens(self, seq_id: SeqId) -> int:
        return self._seq(seq_id).num_computed

    def slot_mapping(self, seq_id: SeqId, start: int, end: int) -> list[int]:
        """Flat KV slot (``block_id * block_size + offset``) for positions ``[start, end)``."""
        seq = self._seq(seq_id)
        if not 0 <= start <= end <= seq.num_tokens:
            raise ValueError(f"positions [{start}, {end}) outside reserved [0, {seq.num_tokens})")
        bs = self.block_size
        return [seq.blocks[p // bs] * bs + p % bs for p in range(start, end)]

    def blocks_needed_for_prompt(self, token_ids: Sequence[int], namespace: bytes = b"") -> int:
        """New blocks :meth:`allocate` would take for this prompt, after prefix hits."""
        hits, _ = self._lookup(token_ids, namespace)
        return self._ceil_div(len(token_ids)) - len(hits)

    def can_allocate(self, token_ids: Sequence[int], namespace: bytes = b"") -> bool:
        """Whether :meth:`allocate` would succeed while honoring the watermark."""
        hits, _ = self._lookup(token_ids, namespace)
        return self._admissible(len(token_ids), hits, respect_watermark=True)

    def stats(self) -> BlockManagerStats:
        s = self._stats
        return BlockManagerStats(
            num_blocks=self.num_blocks,
            free_blocks=len(self._free),
            cached_idle_blocks=self._num_cached_idle,
            used_blocks=self.num_blocks - len(self._free) - self._num_cached_idle,
            prompt_tokens_queried=s.prompt_tokens_queried,
            prompt_tokens_hit=s.prompt_tokens_hit,
            evictions=s.evictions,
            cow_copies=s.cow_copies,
            duplicate_registrations=s.duplicate_registrations,
        )

    # ------------------------------------------------------------------ lifecycle

    def allocate(
        self,
        seq_id: SeqId,
        token_ids: Sequence[int],
        *,
        namespace: bytes = b"",
        respect_watermark: bool = True,
    ) -> Allocation:
        """Admit a prompt: reuse cached prefix blocks, allocate the rest.

        Raises:
            SequenceError: ``seq_id`` already exists.
            NoFreeBlocksError: Not enough blocks (state unchanged).
            ValueError: Empty prompt.
        """
        self._check_thread()
        if seq_id in self._seqs:
            raise SequenceError(f"sequence {seq_id!r} already allocated")
        if not token_ids:
            raise ValueError("cannot allocate an empty prompt")
        hits, hashes = self._lookup(token_ids, namespace)
        if not self._admissible(len(token_ids), hits, respect_watermark=respect_watermark):
            raise NoFreeBlocksError(
                f"need {self._ceil_div(len(token_ids)) - len(hits)} blocks, "
                f"{self.num_available_blocks} available (watermark {self._watermark_blocks})"
            )

        # Revive hits BEFORE taking new blocks so eviction can never pick a block we are reusing.
        for b in hits:
            self._incref(b)
            self.policy.on_hit(b)
        blocks = list(hits)
        for _ in range(self._ceil_div(len(token_ids)) - len(hits)):
            blocks.append(self._take_block())

        n_cached = len(hits) * self.block_size
        self._stats.prompt_tokens_queried += len(token_ids)
        self._stats.prompt_tokens_hit += n_cached
        seq = _Seq(
            blocks=blocks, num_tokens=len(token_ids), num_computed=n_cached, namespace=namespace
        )
        if self.enable_prefix_caching:
            seq.block_hashes = hashes
            seq.tail_tokens = list(token_ids[len(hashes) * self.block_size :])
            seq.num_registered = len(hits)
        self._seqs[seq_id] = seq
        return Allocation(tuple(blocks), n_cached)

    def append_slots(self, seq_id: SeqId, token_ids: Sequence[int]) -> list[CopyOp]:
        """Reserve slots for ``token_ids`` appended to the sequence (e.g. one sampled token).

        Returns:
            Copy-on-write operations the caller must apply to the KV tensors before writing.

        Raises:
            NoFreeBlocksError: Not enough blocks (state unchanged). The scheduler should preempt.
        """
        self._check_thread()
        seq = self._seq(seq_id)
        bs = self.block_size
        pos = seq.num_tokens
        needed = sum(1 for p in range(pos, pos + len(token_ids)) if p % bs == 0)
        cow = pos % bs != 0 and self._ref[seq.blocks[-1]] > 1
        if needed + cow > self.num_available_blocks:
            raise NoFreeBlocksError(f"append needs {needed + cow} blocks")

        ops: list[CopyOp] = []
        if cow:
            # The partial last block is shared with a fork: writing into it would change the other
            # sequence's KV. Give this sequence a private copy first.
            old = seq.blocks[-1]
            new = self._take_block()
            ops.append(CopyOp(src=old, dst=new))
            seq.blocks[-1] = new
            self._decref(old)
            self._stats.cow_copies += 1
        for p in range(pos, pos + len(token_ids)):
            if p % bs == 0:
                seq.blocks.append(self._take_block())
        seq.num_tokens += len(token_ids)
        if self.enable_prefix_caching:
            seq.tail_tokens.extend(token_ids)
        return ops

    def mark_computed(self, seq_id: SeqId, num_computed_tokens: int) -> None:
        """Record that KV for the first ``num_computed_tokens`` tokens has been written.

        Newly completed full blocks are registered in the prefix index (rule 2).
        """
        self._check_thread()
        seq = self._seq(seq_id)
        if not seq.num_computed <= num_computed_tokens <= seq.num_tokens:
            raise ValueError(
                f"computed must be in [{seq.num_computed}, {seq.num_tokens}], "
                f"got {num_computed_tokens}"
            )
        seq.num_computed = num_computed_tokens
        if self.enable_prefix_caching:
            self._register_full_blocks(seq)

    def fork(self, parent_id: SeqId, child_id: SeqId) -> None:
        """Create ``child_id`` sharing all of the parent's blocks (e.g. parallel sampling).

        Both sequences then reference every block. The first append into a shared partial block
        triggers copy-on-write.
        """
        self._check_thread()
        parent = self._seq(parent_id)
        if child_id in self._seqs:
            raise SequenceError(f"sequence {child_id!r} already allocated")
        for b in parent.blocks:
            self._incref(b)
        self._seqs[child_id] = _Seq(
            blocks=list(parent.blocks),
            num_tokens=parent.num_tokens,
            num_computed=parent.num_computed,
            namespace=parent.namespace,
            block_hashes=list(parent.block_hashes),
            tail_tokens=list(parent.tail_tokens),
            num_registered=parent.num_registered,
        )

    def free(self, seq_id: SeqId) -> None:
        """Release a finished, cancelled, or preempted sequence's blocks.

        Cached blocks stay valid as cached-idle (revivable); uncached blocks go to the free list.
        """
        self._check_thread()
        seq = self._seqs.pop(seq_id, None)
        if seq is None:
            raise SequenceError(f"unknown sequence {seq_id!r}")
        # Deepest first: children become candidates before parents, matching leaf-first eviction.
        for b in reversed(seq.blocks):
            self._decref(b)

    def reset_prefix_cache(self) -> int:
        """Evict every evictable cached block (admin/testing). Returns the number evicted."""
        self._check_thread()
        n = 0
        while len(self.policy):
            self._free.append(self._evict(self.policy.pop_victim()))
            n += 1
        return n

    # ------------------------------------------------------------------ verification

    def check_invariants(self) -> None:
        """Exhaustively verify internal consistency. O(blocks + table entries); for tests/debugging.

        Raises:
            AssertionError: Describing the first violated invariant.
        """
        expected_ref: Counter[int] = Counter()
        for seq in self._seqs.values():
            expected_ref.update(seq.blocks)
            assert len(seq.blocks) == self._ceil_div(seq.num_tokens), "table size != ceil(tokens/B)"
            assert 0 <= seq.num_computed <= seq.num_tokens, "computed outside [0, tokens]"
        for b in range(self.num_blocks):
            assert self._ref[b] == expected_ref[b], (
                f"block {b}: ref {self._ref[b]} != {expected_ref[b]}"
            )

        free = set(self._free)
        assert len(free) == len(self._free), "duplicate block on the free list"
        cached = set(self._cached)
        used = {b for b in range(self.num_blocks) if self._ref[b] > 0}
        assert not free & cached, "block both free and cached"
        assert not free & used, "block both free and used"
        idle = cached - used
        assert free | cached | used == set(range(self.num_blocks)), "block in no state (leaked)"
        assert len(idle) == self._num_cached_idle, "cached-idle counter drift"

        assert len(self._by_hash) == len(self._cached), "hash index size mismatch"
        children: Counter[int] = Counter()
        for b, entry in self._cached.items():
            assert self._by_hash.get(entry.block_hash) == b, f"block {b} not indexed by its hash"
            if entry.parent is not None:
                assert entry.parent in self._cached, (
                    f"block {b}: parent not cached (not prefix-closed)"
                )
                children[entry.parent] += 1
        for b, entry in self._cached.items():
            assert entry.children == children[b], (
                f"block {b}: children {entry.children} != {children[b]}"
            )

        for b in cached & used:
            parent = self._cached[b].parent
            # Liveness: an in-use cached block's cached ancestors are in use, so every idle block
            # has only idle descendants and is eventually evictable (available count is exact).
            assert parent is None or self._ref[parent] > 0, f"in-use block {b} has idle parent"

        candidates = {b for b in idle if self._cached[b].children == 0}
        assert len(self.policy) == len(candidates), (
            f"policy holds {len(self.policy)} candidates, expected {len(candidates)}"
        )

    # ------------------------------------------------------------------ internals

    def _ceil_div(self, n: int) -> int:
        return -(-n // self.block_size)

    def _seq(self, seq_id: SeqId) -> _Seq:
        self._check_thread()
        try:
            return self._seqs[seq_id]
        except KeyError:
            raise SequenceError(f"unknown sequence {seq_id!r}") from None

    def _check_thread(self) -> None:
        if self._debug and threading.get_ident() != self._owner:
            raise RuntimeError("BlockManager is thread-confined; called from a non-owner thread")

    def _lookup(
        self, token_ids: Sequence[int], namespace: bytes
    ) -> tuple[list[int], list[BlockHash]]:
        """Longest cached full-block prefix, excluding the block with the last token (rule 3)."""
        if not self.enable_prefix_caching:
            return [], []
        eligible = (len(token_ids) - 1) // self.block_size * self.block_size
        hashes = chain_hashes(namespace, token_ids[:eligible], self.block_size)
        hits: list[int] = []
        for h in hashes:
            b = self._by_hash.get(h)
            if b is None:
                break
            hits.append(b)
        return hits, hashes

    def _admissible(self, n_tokens: int, hits: list[int], *, respect_watermark: bool) -> bool:
        revived = sum(1 for b in hits if self._ref[b] == 0)  # idle hits stop being evictable
        available = self.num_available_blocks - revived
        reserve = self._watermark_blocks if respect_watermark else 0
        return available - (self._ceil_div(n_tokens) - len(hits)) >= reserve

    def _take_block(self) -> int:
        # Eviction cannot raise LookupError here: every caller pre-checks num_available_blocks.
        b = self._free.popleft() if self._free else self._evict(self.policy.pop_victim())
        self._ref[b] = 1
        return b

    def _incref(self, b: int) -> None:
        if self._ref[b] == 0:
            entry = self._cached.get(b)
            if entry is None:
                raise AssertionError(f"incref on free block {b}")
            self._num_cached_idle -= 1
            if entry.children == 0:
                self.policy.remove_candidate(b)
        self._ref[b] += 1

    def _decref(self, b: int) -> None:
        self._ref[b] -= 1
        if self._ref[b] > 0:
            return
        entry = self._cached.get(b)
        if entry is None:
            self._free.append(b)
            return
        self._num_cached_idle += 1
        if entry.children == 0:
            self.policy.add_candidate(b)

    def _evict(self, b: int) -> int:
        entry = self._cached.pop(b)
        del self._by_hash[entry.block_hash]
        self._num_cached_idle -= 1
        self.policy.on_evicted(b)
        self._stats.evictions += 1
        if entry.parent is not None:
            parent = self._cached[entry.parent]
            parent.children -= 1
            if parent.children == 0 and self._ref[entry.parent] == 0:
                self.policy.add_candidate(entry.parent)  # parent is now an evictable leaf
        return b

    def _register_full_blocks(self, seq: _Seq) -> None:
        bs = self.block_size
        full_computed = seq.num_computed // bs
        while seq.num_registered < full_computed:
            i = seq.num_registered
            while len(seq.block_hashes) <= i:  # hash newly completed blocks from the tail
                parent_hash = seq.block_hashes[-1] if seq.block_hashes else ROOT_PARENT
                seq.block_hashes.append(
                    block_hash(seq.namespace, parent_hash, seq.tail_tokens[:bs])
                )
                del seq.tail_tokens[:bs]
            prev_hash = seq.block_hashes[i - 1] if i else None
            prev_block = seq.blocks[i - 1] if i else None
            self._register(seq.blocks[i], seq.block_hashes[i], prev_hash, prev_block, i)
            seq.num_registered += 1

    def _register(
        self,
        b: int,
        h: BlockHash,
        parent_hash: BlockHash | None,
        parent_block: int | None,
        depth: int,
    ) -> None:
        if b in self._cached:
            return  # obtained from the cache (prefix hit) or shared via fork: already indexed
        if h in self._by_hash:
            # Same content already cached in another block (e.g. two identical prompts admitted
            # before either was computed). Keep the existing entry; ours stays private and is
            # freed normally. Re-pointing the sequence to the cached block would save memory but
            # complicates in-flight metadata. Recorded in stats for visibility.
            self._stats.duplicate_registrations += 1
            return
        parent: int | None = None
        if parent_hash is not None:
            parent = self._by_hash.get(parent_hash)
            if parent is None or parent != parent_block:
                # Either the parent is not cached (index must stay prefix-closed), or its content is
                # cached in a DIFFERENT block (we hold a private duplicate). Linking under that
                # other block would let an idle parent own an in-use child: counted as available but
                # never evictable, which would break the allocation pre-check.
                return
            pentry = self._cached[parent]
            if pentry.children == 0 and self._ref[parent] == 0:
                self.policy.remove_candidate(parent)  # an idle leaf just gained a child
            pentry.children += 1
        self._cached[b] = _Cached(block_hash=h, parent=parent, depth=depth)
        self._by_hash[h] = b
        self.policy.on_cached(b, depth)
