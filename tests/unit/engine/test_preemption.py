"""Memory pressure: preemption by recompute must never change outputs or leak blocks."""

from __future__ import annotations

import random

import pytest

from dli.engine import LLMEngine, SchedulerConfig
from dli.kv import BlockManager
from dli.sampling import SamplingParams

from .fake import FakeRunner, reference


def tight_engine(num_blocks: int, budget: int = 64, seqs: int = 16) -> tuple[LLMEngine, FakeRunner]:
    bm = BlockManager(num_blocks, block_size=4, watermark=0.0)
    runner = FakeRunner(bm)
    cfg = SchedulerConfig(max_num_seqs=seqs, max_num_batched_tokens=budget)
    return LLMEngine(runner, bm, cfg), runner


def drain(eng: LLMEngine, limit: int = 50_000) -> None:
    for _ in range(limit):
        if not eng.has_unfinished():
            return
        eng.step()
    raise AssertionError("did not finish: livelock")


@pytest.mark.parametrize("num_blocks", [12, 20, 40])
def test_outputs_unchanged_under_preemption(num_blocks: int) -> None:
    eng, _runner = tight_engine(num_blocks)
    rng = random.Random(num_blocks)
    cases = []
    for i in range(24):
        prompt = [rng.randrange(50) for _ in range(rng.randrange(1, 14))]
        max_tokens = rng.randrange(1, 30)
        cases.append(
            (eng.add_request(f"r{i}", prompt, SamplingParams(max_tokens=max_tokens)), prompt)
        )
    drain(eng)
    for req, prompt in cases:
        assert req.output_ids == reference(prompt, req.params.max_tokens), req.request_id
    assert sum(r.num_preemptions for r, _ in cases) > 0, "pool too large: preemption not exercised"
    eng.bm.reset_prefix_cache()
    eng.bm.check_invariants()
    assert eng.bm.stats().free_blocks == num_blocks


def test_no_admission_in_a_step_that_preempted() -> None:
    eng, runner = tight_engine(10)
    for i in range(12):
        eng.add_request(f"r{i}", [i, i + 1, i + 2], SamplingParams(max_tokens=25))
    drain(eng)
    preempting = [s for s in runner.calls if s.preempted]
    assert preempting, "test did not exercise preemption"
    for sched in preempting:
        new = [c for c in sched.chunks if c.start == 0]
        # Chunks starting at 0 in a preempting step would be fresh admissions.
        assert not new, "a request was admitted in the same step as a preemption"


def test_resumed_request_reuses_its_cached_prefix() -> None:
    # "grower" (admitted first: oldest, never preempted) and "victim" each fit alone in the
    # 12-block pool but not together, so the victim is preempted mid-decode. Its full blocks become
    # cached-idle; the grower then evicts some of them (deepest first), so the victim's resume must
    # start at a block boundary past 0: reuse of whatever prefix survived.
    eng, runner = tight_engine(12)
    grower = eng.add_request("grower", [1, 2, 3], SamplingParams(max_tokens=40))
    victim_prompt = [7, 7, 7, 7, 8, 8, 8, 8]
    victim = eng.add_request("victim", victim_prompt, SamplingParams(max_tokens=30))
    drain(eng)
    assert victim.num_preemptions >= 1
    assert grower.num_preemptions == 0
    assert victim.output_ids == reference(victim_prompt, 30)
    assert grower.output_ids == reference([1, 2, 3], 40)
    victim_chunks = [c for s in runner.calls for c in s.chunks if c.request is victim]
    first_resume = next(c for c in victim_chunks[1:] if c.num_tokens > 1)
    assert first_resume.start > 0, "resume recomputed blocks that were still cached"
    assert first_resume.start % 4 == 0


def test_oldest_request_is_never_preempted() -> None:
    eng, _runner = tight_engine(16)
    first = eng.add_request("oldest", [1, 2, 3], SamplingParams(max_tokens=45))
    for i in range(10):
        eng.add_request(f"r{i}", [4, 5, i], SamplingParams(max_tokens=20))
    drain(eng)
    assert first.num_preemptions == 0
    assert first.output_ids == reference([1, 2, 3], 45)


def test_abort_of_a_preempted_waiting_request() -> None:
    eng, _runner = tight_engine(10)
    reqs = [eng.add_request(f"r{i}", [i, 9, 9], SamplingParams(max_tokens=30)) for i in range(8)]
    while not any(r.num_preemptions for r in reqs):
        eng.step()
    preempted = next(r for r in reqs if r.num_preemptions)
    assert eng.abort(preempted.request_id)
    drain(eng)
    eng.bm.reset_prefix_cache()
    assert eng.bm.stats().free_blocks == 10
