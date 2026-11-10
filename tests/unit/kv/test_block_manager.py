"""Behavioral unit tests for BlockManager. Every test ends by checking all invariants."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from dli.kv import BlockManager, CopyOp, NoFreeBlocksError, SequenceError

BS = 4


@pytest.fixture
def bm() -> Iterator[BlockManager]:
    m = BlockManager(num_blocks=16, block_size=BS, watermark=0.0)
    yield m
    m.check_invariants()


def prompt(n: int, base: int = 100) -> list[int]:
    return list(range(base, base + n))


def admit(m: BlockManager, seq: str, toks: list[int], **kw: object) -> int:
    """Allocate and mark the whole prompt computed (what the runner does after prefill)."""
    alloc = m.allocate(seq, toks, **kw)  # type: ignore[arg-type]
    m.mark_computed(seq, len(toks))
    return alloc.num_cached_tokens


# ---------------------------------------------------------------- allocation and release


def test_allocation_rounds_up_to_blocks_and_free_returns_them(bm: BlockManager) -> None:
    alloc = bm.allocate("a", prompt(9))
    assert len(alloc.block_table) == 3
    assert alloc.num_cached_tokens == 0
    assert bm.num_available_blocks == 13
    bm.free("a")
    assert bm.num_available_blocks == 16


def test_slot_mapping_follows_block_table(bm: BlockManager) -> None:
    table = bm.allocate("a", prompt(6)).block_table
    assert bm.slot_mapping("a", 0, 6) == [table[0] * BS + i for i in range(4)] + [
        table[1] * BS + i for i in range(2)
    ]
    with pytest.raises(ValueError, match="outside"):
        bm.slot_mapping("a", 0, 7)


def test_append_allocates_only_at_block_boundaries(bm: BlockManager) -> None:
    bm.allocate("a", prompt(3))
    assert bm.append_slots("a", [1]) == []
    assert len(bm.block_table("a")) == 1  # 4 tokens fit in one block
    bm.append_slots("a", [2])
    assert len(bm.block_table("a")) == 2
    bm.append_slots("a", [3, 4, 5, 6])  # multi-token append crossing one more boundary
    assert bm.num_tokens("a") == 9
    assert len(bm.block_table("a")) == 3


def test_errors_for_unknown_duplicate_and_empty(bm: BlockManager) -> None:
    bm.allocate("a", prompt(2))
    with pytest.raises(SequenceError):
        bm.allocate("a", prompt(2))
    with pytest.raises(SequenceError):
        bm.free("nope")
    with pytest.raises(SequenceError):
        bm.fork("nope", "x")
    with pytest.raises(ValueError, match="empty"):
        bm.allocate("b", [])


def test_mark_computed_bounds(bm: BlockManager) -> None:
    bm.allocate("a", prompt(5))
    bm.mark_computed("a", 3)
    with pytest.raises(ValueError, match="computed must be"):
        bm.mark_computed("a", 2)  # cannot go backwards
    with pytest.raises(ValueError, match="computed must be"):
        bm.mark_computed("a", 6)  # beyond reserved slots


# ---------------------------------------------------------------- exhaustion and atomicity


def test_allocation_failure_is_atomic(bm: BlockManager) -> None:
    bm.allocate("a", prompt(14 * BS))
    before = (bm.num_available_blocks, bm.block_table("a"))
    with pytest.raises(NoFreeBlocksError):
        bm.allocate("b", prompt(3 * BS))
    assert (bm.num_available_blocks, bm.block_table("a")) == before
    assert not bm.has_sequence("b")


def test_append_failure_is_atomic(bm: BlockManager) -> None:
    bm.allocate("a", prompt(16 * BS))  # whole pool
    with pytest.raises(NoFreeBlocksError):
        bm.append_slots("a", [1])
    assert bm.num_tokens("a") == 16 * BS


def test_watermark_reserves_blocks_for_running_sequences() -> None:
    m = BlockManager(num_blocks=10, block_size=BS, watermark=0.2)  # keep 2 blocks in reserve
    m.allocate("a", prompt(7 * BS))
    assert not m.can_allocate(prompt(2 * BS, base=900))
    with pytest.raises(NoFreeBlocksError):
        m.allocate("b", prompt(2 * BS, base=900))
    m.allocate("b", prompt(2 * BS, base=900), respect_watermark=False)  # e.g. a resumed sequence
    m.append_slots("a", [1])  # appends ignore the watermark: running sequences may use reserve
    m.check_invariants()


# ---------------------------------------------------------------- prefix caching


def test_prefix_hit_requires_computed_blocks(bm: BlockManager) -> None:
    p = prompt(10)
    bm.allocate("a", p)  # allocated but NOT computed
    assert bm.allocate("b", p).num_cached_tokens == 0, "must never hit unwritten KV"
    bm.mark_computed("a", 10)
    assert bm.allocate("c", p).num_cached_tokens == 8


def test_prefix_hit_shares_blocks_and_survives_owner_release(bm: BlockManager) -> None:
    p = prompt(10)
    admit(bm, "a", p)
    shared = bm.block_table("a")[:2]
    bm.free("a")  # blocks become cached-idle, not free
    assert bm.stats().cached_idle_blocks == 2
    alloc = bm.allocate("b", [*p[:8], 7, 7, 7])
    assert alloc.num_cached_tokens == 8
    assert alloc.block_table[:2] == shared


def test_last_block_is_always_recomputed_even_on_full_hit(bm: BlockManager) -> None:
    p = prompt(8)  # exactly two full blocks
    admit(bm, "a", p)
    alloc = bm.allocate("b", p)
    assert alloc.num_cached_tokens == 4  # block 2 holds the last token: must produce logits
    assert alloc.block_table[1] != bm.block_table("a")[1]


def test_divergent_suffix_shares_only_common_full_blocks(bm: BlockManager) -> None:
    admit(bm, "a", prompt(12))
    alt = prompt(12)
    alt[5] = 0  # differs inside the second block
    assert bm.allocate("b", alt).num_cached_tokens == 4


def test_namespace_prevents_cross_namespace_hits(bm: BlockManager) -> None:
    admit(bm, "a", prompt(9), namespace=b"tenant-1")
    assert bm.allocate("b", prompt(9), namespace=b"tenant-2").num_cached_tokens == 0
    assert bm.allocate("c", prompt(9), namespace=b"tenant-1").num_cached_tokens == 8


def test_prefix_caching_disabled_frees_directly() -> None:
    m = BlockManager(num_blocks=8, block_size=BS, enable_prefix_caching=False, watermark=0.0)
    admit(m, "a", prompt(9))
    m.free("a")
    assert m.stats().free_blocks == 8
    assert m.allocate("b", prompt(9)).num_cached_tokens == 0
    m.check_invariants()


def test_decode_tokens_complete_blocks_that_become_cacheable(bm: BlockManager) -> None:
    admit(bm, "a", prompt(3))
    for t in (50, 51, 52, 53, 54):  # one at a time, like decoding
        bm.append_slots("a", [t])
        bm.mark_computed("a", bm.num_tokens("a"))
    full = [*prompt(3), 50, 51, 52, 53, 54]
    assert bm.allocate("b", [*full, 99]).num_cached_tokens == 8


def test_duplicate_concurrent_prompts_are_counted_not_double_indexed(bm: BlockManager) -> None:
    p = prompt(9)
    bm.allocate("a", p)
    bm.allocate("b", p)  # admitted before "a" is computed: no hit, private copy
    bm.mark_computed("a", 9)
    bm.mark_computed("b", 9)
    assert bm.stats().duplicate_registrations == 2  # both full blocks of "b" already cached via "a"
    bm.free("a")
    bm.free("b")


# ---------------------------------------------------------------- eviction


def test_eviction_reclaims_idle_cached_blocks_leaf_first() -> None:
    m = BlockManager(num_blocks=4, block_size=BS, watermark=0.0)
    admit(m, "a", prompt(3 * BS + 1))  # 3 full cached blocks + 1 partial
    a_table = m.block_table("a")
    m.free("a")
    assert m.stats().cached_idle_blocks == 3
    m.allocate("b", prompt(2 * BS, base=900))  # needs 2: 1 free + 1 evicted
    assert m.stats().evictions == 1
    m.free("b")  # never computed, so its blocks return to the free list
    # Leaf-first: the deepest cached block went first, so the 2-block prefix is still reusable.
    assert m.allocate("c", prompt(3 * BS + 1)).block_table[:2] == a_table[:2]
    m.check_invariants()


def test_hit_revives_idle_block_and_protects_it_from_eviction() -> None:
    m = BlockManager(num_blocks=3, block_size=BS, watermark=0.0)
    admit(m, "a", prompt(BS + 1))
    m.free("a")  # block0 cached-idle, block1 free
    alloc = m.allocate("b", [*prompt(BS), 5])  # revives block0
    assert alloc.num_cached_tokens == BS
    with pytest.raises(NoFreeBlocksError):
        m.allocate("c", prompt(2 * BS, base=900))  # only 1 truly available; revived block is used
    m.check_invariants()


def test_reset_prefix_cache_evicts_everything_idle(bm: BlockManager) -> None:
    admit(bm, "a", prompt(13))
    bm.free("a")
    assert bm.reset_prefix_cache() == 3
    assert bm.stats().free_blocks == 16


# ---------------------------------------------------------------- fork and copy-on-write


def test_fork_shares_blocks_and_cow_copies_partial_block_once(bm: BlockManager) -> None:
    admit(bm, "a", prompt(6))  # block0 full, block1 has 2 tokens
    bm.fork("a", "b")
    assert bm.block_table("a") == bm.block_table("b")
    shared_partial = bm.block_table("a")[1]

    ops = bm.append_slots("b", [1])
    assert len(ops) == 1
    assert ops[0] == CopyOp(src=shared_partial, dst=bm.block_table("b")[1])
    assert bm.block_table("b")[0] == bm.block_table("a")[0]  # full block stays shared
    assert bm.append_slots("a", [2]) == []  # "a" now owns the original partial block alone
    assert bm.stats().cow_copies == 1


def test_fork_at_block_boundary_needs_no_copy(bm: BlockManager) -> None:
    admit(bm, "a", prompt(8))
    bm.fork("a", "b")
    assert bm.append_slots("b", [1]) == []  # new token starts a fresh block


def test_freeing_forks_in_any_order_releases_everything(bm: BlockManager) -> None:
    admit(bm, "a", prompt(7))
    bm.fork("a", "b")
    bm.fork("b", "c")
    bm.append_slots("c", [1])
    for s in ("b", "a", "c"):
        bm.free(s)
    bm.reset_prefix_cache()
    assert bm.stats().free_blocks == 16


# ---------------------------------------------------------------- concurrency contract


def test_cross_thread_use_is_rejected(bm: BlockManager) -> None:
    bm.allocate("a", prompt(3))
    errors: list[BaseException] = []

    def other() -> None:
        try:
            bm.append_slots("a", [1])
        except RuntimeError as exc:
            errors.append(exc)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert errors, "a non-owner thread must not be able to mutate the manager"
    assert "thread-confined" in str(errors[0])
