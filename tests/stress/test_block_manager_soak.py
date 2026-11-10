"""Stress tier (``-m stress``): long randomized runs of the block manager (spec 01 AC2, M2 scope).

* ``test_one_million_random_operations``: 10^6 operations. Cheap conservation checks run after
  EVERY operation (block-state accounting must sum to the pool size and match each table);
  the exhaustive ``check_invariants`` runs every 2,000 operations and at the end. A full check
  after every operation would be O(10^6 x pool) and is what the Hypothesis state machine does
  at smaller scale.
* ``test_ten_thousand_request_lifecycles``: request-level soak with shared system prompts,
  decoding, forks (parallel sampling), and random cancellation. At the end, after freeing everything
  and clearing the prefix cache, every block must be free (zero leaks).

This exercises the manager in isolation. The engine-level soak (real forward passes) belongs to
M3/M4, once the scheduler exists.
"""

from __future__ import annotations

import random
import time

import pytest

from dli.kv import BlockManager, NoFreeBlocksError

pytestmark = pytest.mark.stress


def _conservation(bm: BlockManager) -> None:
    s = bm.stats()
    assert s.free_blocks + s.cached_idle_blocks + s.used_blocks == bm.num_blocks


@pytest.mark.parametrize("policy", ["lru", "cost"])
def test_one_million_random_operations(policy: str) -> None:
    rng = random.Random(20261110)
    bm = BlockManager(num_blocks=256, block_size=8, eviction_policy=policy, watermark=0.0)
    systems = [[rng.randrange(1000) for _ in range(rng.randrange(8, 64))] for _ in range(6)]
    live: list[int] = []
    next_id = 0
    counts = {"allocate": 0, "append": 0, "compute": 0, "fork": 0, "free": 0, "nospace": 0}
    t0 = time.perf_counter()
    for i in range(1_000_000):
        r = rng.random()
        try:
            if r < 0.2 or not live:
                prompt = rng.choice(systems)[: rng.randrange(1, 64)] + [
                    rng.randrange(1000) for _ in range(rng.randrange(0, 24))
                ] or [1]
                bm.allocate(next_id, prompt)
                live.append(next_id)
                next_id += 1
                counts["allocate"] += 1
            elif r < 0.55:
                s = rng.choice(live)
                bm.mark_computed(s, bm.num_tokens(s))
                bm.append_slots(s, [rng.randrange(1000) for _ in range(rng.randrange(1, 4))])
                counts["append"] += 1
            elif r < 0.75:
                s = rng.choice(live)
                bm.mark_computed(s, rng.randint(bm.num_computed_tokens(s), bm.num_tokens(s)))
                counts["compute"] += 1
            elif r < 0.8:
                bm.fork(rng.choice(live), next_id)
                live.append(next_id)
                next_id += 1
                counts["fork"] += 1
            else:
                bm.free(live.pop(rng.randrange(len(live))))
                counts["free"] += 1
        except NoFreeBlocksError:
            counts["nospace"] += 1
            bm.free(live.pop(rng.randrange(len(live))))  # what a scheduler would do: preempt
        _conservation(bm)
        if i % 2000 == 0:
            bm.check_invariants()
    bm.check_invariants()
    elapsed = time.perf_counter() - t0
    for s in live:
        bm.free(s)
    bm.reset_prefix_cache()
    bm.check_invariants()
    assert bm.stats().free_blocks == bm.num_blocks, "blocks leaked"
    print(f"\n[{policy}] 1e6 ops in {elapsed:.1f}s {counts} stats={bm.stats()}")


def test_ten_thousand_request_lifecycles() -> None:
    rng = random.Random(7)
    bm = BlockManager(num_blocks=512, block_size=16, eviction_policy="cost", watermark=0.02)
    systems = [[rng.randrange(32000) for _ in range(rng.choice([32, 128, 300]))] for _ in range(8)]
    active: dict[int, int] = {}  # seq -> remaining decode steps
    cancelled = finished = preempted = rejected = 0
    for rid in range(10_000):
        prompt = rng.choice(systems) + [rng.randrange(32000) for _ in range(rng.randrange(1, 200))]
        if bm.can_allocate(prompt):
            alloc = bm.allocate(rid, prompt)
            assert alloc.num_cached_tokens < len(prompt)
            bm.mark_computed(rid, len(prompt))  # prefill
            active[rid] = rng.randrange(1, 120)
            if rng.random() < 0.05:  # parallel sampling: fork a sibling
                bm.fork(rid, -rid - 1)
                active[-rid - 1] = rng.randrange(1, 60)
        else:
            rejected += 1
        for s in list(active):  # one decode step for every running sequence
            if rng.random() < 0.01:
                bm.free(s)
                del active[s]
                cancelled += 1
                continue
            try:
                bm.append_slots(s, [rng.randrange(32000)])
            except NoFreeBlocksError:
                bm.free(s)
                del active[s]
                preempted += 1
                continue
            bm.mark_computed(s, bm.num_tokens(s))
            active[s] -= 1
            if active[s] == 0:
                bm.free(s)
                del active[s]
                finished += 1
        if rid % 500 == 0:
            bm.check_invariants()
    for s in list(active):
        bm.free(s)
    bm.check_invariants()
    bm.reset_prefix_cache()
    stats = bm.stats()
    assert stats.free_blocks == bm.num_blocks, f"{bm.num_blocks - stats.free_blocks} blocks leaked"
    print(
        f"\n10k lifecycles: finished={finished} cancelled={cancelled} preempted={preempted} "
        f"rejected={rejected} hit_rate={stats.hit_rate:.3f} evictions={stats.evictions} "
        f"cow={stats.cow_copies}"
    )
    assert cancelled > 0
    assert stats.cow_copies > 0
    assert stats.evictions > 0  # the soak must actually exercise memory pressure
