from __future__ import annotations

import pytest

from dli.kv.eviction import CostAwarePolicy, LRUPolicy, make_policy
from dli.kv.hashing import ROOT_PARENT, block_hash, chain_hashes


def test_chain_hash_depends_on_whole_prefix_not_just_block_tokens() -> None:
    a = chain_hashes(b"m", [1, 2, 3, 4, 5, 6], 3)
    b = chain_hashes(b"m", [9, 9, 9, 4, 5, 6], 3)
    assert len(a) == len(b) == 2
    assert a[1] != b[1], "second block hashes must differ when the first block differs"


def test_partial_trailing_block_is_not_hashed() -> None:
    assert len(chain_hashes(b"m", list(range(7)), 3)) == 2


def test_namespace_isolates_identical_tokens_and_is_length_prefixed() -> None:
    toks = [1, 2, 3]
    assert block_hash(b"model-a", ROOT_PARENT, toks) != block_hash(b"model-b", ROOT_PARENT, toks)
    # Without length-prefixing, these two could serialize to the same byte stream.
    assert block_hash(b"ab", ROOT_PARENT, toks) != block_hash(b"a", ROOT_PARENT, toks)


def test_large_token_ids_are_not_truncated() -> None:
    assert block_hash(b"", ROOT_PARENT, [2**40]) != block_hash(b"", ROOT_PARENT, [0])


def test_lru_evicts_least_recently_released() -> None:
    p = LRUPolicy()
    for b in (1, 2, 3):
        p.add_candidate(b)
    p.remove_candidate(1)  # revived
    p.add_candidate(1)  # released again: now most recent
    assert [p.pop_victim() for _ in range(3)] == [2, 3, 1]
    with pytest.raises(LookupError):
        p.pop_victim()


def test_cost_aware_keeps_frequently_hit_block_over_recent_one_off() -> None:
    p = CostAwarePolicy(depth_weight=0.0)
    p.on_cached(1, depth=0)
    for _ in range(5):
        p.on_hit(1)  # e.g. a shared system prompt
    p.add_candidate(1)
    p.on_cached(2, depth=0)
    p.add_candidate(2)  # a one-off block released more recently
    assert p.pop_victim() == 2  # LRU would evict block 1 here


def test_cost_aware_aging_eventually_evicts_stale_popular_block() -> None:
    p = CostAwarePolicy(depth_weight=0.0)
    p.on_cached(0, depth=0)
    for _ in range(3):
        p.on_hit(0)
    p.add_candidate(0)  # priority 4
    victims = []
    for b in range(1, 10):  # stream of one-offs: each victim raises the inflation value L
        p.on_cached(b, depth=0)
        p.add_candidate(b)
        victims.append(p.pop_victim())
    assert 0 in victims, "without aging, a once-popular block would pin memory forever"


def test_cost_aware_prefers_evicting_shallow_blocks_when_depth_weighted() -> None:
    p = CostAwarePolicy(depth_weight=1.0)
    p.on_cached(1, depth=10)
    p.on_cached(2, depth=0)
    p.add_candidate(1)
    p.add_candidate(2)
    assert p.pop_victim() == 2


def test_cost_aware_heap_stays_bounded_under_revive_release_churn() -> None:
    p = CostAwarePolicy()
    p.on_cached(7, depth=0)
    for _ in range(10_000):
        p.add_candidate(7)
        p.remove_candidate(7)
    assert p.heap_size() <= 2 * len(p) + 65


def test_make_policy() -> None:
    assert make_policy("lru").name == "lru"
    assert make_policy("cost").name == "cost"
    with pytest.raises(ValueError, match="unknown"):
        make_policy("fifo")
