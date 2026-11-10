"""Randomized and adversarial scheduler testing (Hypothesis-generated scenarios).

Each scenario mixes arrivals, priorities, tenants, and cancellations against deliberately hostile
limits (1-token budgets, single-sequence caps, pools barely larger than the largest request).
Per-step invariants plus end-state checks (liveness, exact outputs, zero leaks) must hold.
Fast profile ~60 scenarios; ``HYPOTHESIS_PROFILE=focused`` runs 2,000.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

from dli.engine import FinishReason, LLMEngine, Priority, SchedulerConfig
from dli.kv import BlockManager
from dli.sampling import SamplingParams

from .fake import FakeRunner, reference


@dataclass(frozen=True)
class Scenario:
    seed: int
    num_blocks: int
    block_size: int
    budget: int
    max_seqs: int
    n_requests: int
    cancel_rate: float


scenarios = st.builds(
    Scenario,
    seed=st.integers(0, 2**32 - 1),
    num_blocks=st.integers(6, 40),
    block_size=st.sampled_from([1, 2, 4, 8]),
    budget=st.sampled_from([1, 2, 3, 8, 32, 128]),
    max_seqs=st.integers(1, 6),
    n_requests=st.integers(1, 25),
    cancel_rate=st.sampled_from([0.0, 0.02, 0.1]),
)


@given(scenarios)
def test_random_adversarial_workloads(sc: Scenario) -> None:
    rng = random.Random(sc.seed)
    bm = BlockManager(sc.num_blocks, sc.block_size, watermark=0.0)
    runner = FakeRunner(bm)
    eng = LLMEngine(
        runner,
        bm,
        SchedulerConfig(
            max_num_seqs=sc.max_seqs,
            max_num_batched_tokens=sc.budget,
            aging_steps=rng.choice([5, 50]),
            tenant_weights={"t1": 2.0},
        ),
    )
    capacity_tokens = sc.num_blocks * sc.block_size
    pending = []
    for i in range(sc.n_requests):
        # Some requests are sized right at (or beyond) the pool's capacity on purpose.
        size = rng.choice([1, 2, 5, capacity_tokens // 2, capacity_tokens - 1, capacity_tokens + 3])
        plen = max(1, min(size, rng.randrange(1, 30)))
        max_tokens = max(1, size - plen) if rng.random() < 0.3 else rng.randrange(1, 20)
        prompt = [rng.randrange(50) for _ in range(plen)]
        arrive_at = rng.randrange(0, 40)
        pending.append((arrive_at, f"r{i}", prompt, max_tokens))
    reqs, prompts, cancelled = {}, {}, set()

    step = 0
    while pending or eng.has_unfinished():
        for item in [p for p in pending if p[0] <= step]:
            pending.remove(item)
            _, rid, prompt, mt = item
            reqs[rid] = eng.add_request(
                rid,
                prompt,
                SamplingParams(max_tokens=mt),
                priority=rng.choice(list(Priority)),
                tenant=rng.choice(["t0", "t1", "t2"]),
            )
            prompts[rid] = prompt
        for rid, r in reqs.items():
            if not r.is_finished and rng.random() < sc.cancel_rate:
                assert eng.abort(rid)
                cancelled.add(rid)

        before = {rid: r.num_computed_tokens for rid, r in reqs.items()}
        eng.step()
        if runner.calls:  # the runner is only called for non-empty batches
            sched = runner.calls[-1]
            ids = [c.request.request_id for c in sched.chunks]
            assert len(ids) == len(set(ids)), "request scheduled twice in one step"
            assert sched.num_batched_tokens <= sc.budget
            assert len(sched.chunks) <= sc.max_seqs
            for c in sched.chunks:
                # Chunks continue exactly where the request's computed KV ended (0 after preemption,
                # or the cached-prefix length on (re)admission).
                assert c.start >= 0
                assert c.start + c.num_tokens <= c.request.num_tokens
                if c.request.num_preemptions == 0 and before.get(c.request.request_id, 0) > 0:
                    assert c.start == before[c.request.request_id]
        bm.check_invariants()
        step += 1
        assert step < 20_000, f"livelock: scenario {sc}"
        runner.calls.clear()

    for rid, r in reqs.items():
        assert r.is_finished
        if rid in cancelled:
            assert r.finish_reason is FinishReason.ABORTED
        elif r.finish_reason is FinishReason.CAPACITY:
            need = -(-(len(prompts[rid]) + r.params.max_tokens) // sc.block_size)
            assert need > sc.num_blocks, "rejected a request that fits"
        else:
            assert r.output_ids == reference(prompts[rid], r.params.max_tokens), rid
    bm.reset_prefix_cache()
    assert bm.stats().free_blocks == sc.num_blocks, "blocks leaked"
