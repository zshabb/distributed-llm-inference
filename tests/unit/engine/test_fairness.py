"""Priority classes, aging, and weighted per-tenant fair queuing at admission."""

from __future__ import annotations

from collections import defaultdict

from dli.engine import LLMEngine, Priority, SchedulerConfig
from dli.kv import BlockManager
from dli.sampling import SamplingParams

from .fake import FakeRunner, reference


def engine(seqs: int = 2, budget: int = 64, **cfg: object) -> tuple[LLMEngine, FakeRunner]:
    bm = BlockManager(512, block_size=4, watermark=0.0)
    runner = FakeRunner(bm)
    config = SchedulerConfig(max_num_seqs=seqs, max_num_batched_tokens=budget, **cfg)  # type: ignore[arg-type]
    return LLMEngine(runner, bm, config), runner


def served_by_tenant(runner: FakeRunner, upto: int) -> dict[str, int]:
    tokens: dict[str, int] = defaultdict(int)
    for sched in runner.calls[:upto]:
        for ch in sched.chunks:
            tokens[ch.request.tenant] += ch.num_tokens
    return tokens


def test_backlogged_tenants_share_equally_despite_submission_order() -> None:
    eng, runner = engine()
    # Tenant A floods the queue first; under FIFO, B would wait until all of A finished.
    for i in range(20):
        eng.add_request(f"a{i}", [1, 2, 3, i], SamplingParams(max_tokens=6), tenant="A")
    for i in range(20):
        eng.add_request(f"b{i}", [4, 5, 6, i], SamplingParams(max_tokens=6), tenant="B")
    for _ in range(60):
        eng.step()
    s = served_by_tenant(runner, 60)
    assert s["B"] > 0.4 * (s["A"] + s["B"]), f"B starved behind A's backlog: {dict(s)}"


def test_weights_skew_the_share() -> None:
    eng, runner = engine(seqs=3, tenant_weights={"A": 2.0, "B": 1.0})
    for i in range(30):
        eng.add_request(f"a{i}", [1, i], SamplingParams(max_tokens=10), tenant="A")
        eng.add_request(f"b{i}", [2, i], SamplingParams(max_tokens=10), tenant="B")
    for _ in range(80):
        eng.step()
    s = served_by_tenant(runner, 80)
    ratio = s["A"] / s["B"]
    assert 1.5 < ratio < 2.6, f"expected ~2:1 service, got {ratio:.2f} ({dict(s)})"


def test_idle_tenant_cannot_bank_credit() -> None:
    eng, runner = engine(seqs=1)
    for i in range(30):
        eng.add_request(f"a{i}", [1, i], SamplingParams(max_tokens=4), tenant="A")
    for _ in range(40):  # A runs alone for a while, accumulating virtual time
        eng.step()
    for i in range(30):
        eng.add_request(f"b{i}", [2, i], SamplingParams(max_tokens=4), tenant="B")
    start = len(runner.calls)
    for _ in range(40):
        eng.step()
    window: dict[str, int] = defaultdict(int)
    for sched in runner.calls[start:]:
        for ch in sched.chunks:
            window[ch.request.tenant] += ch.num_tokens
    # B joins at the current virtual clock: it shares the engine instead of monopolizing it to
    # "catch up" on the 40 steps it was idle.
    assert window["A"] > 0.3 * (window["A"] + window["B"]), dict(window)


def test_interactive_is_admitted_before_earlier_batch_work() -> None:
    eng, _runner = engine(seqs=1)
    batch = [
        eng.add_request(f"b{i}", [i, 1], SamplingParams(max_tokens=3), priority=Priority.BATCH)
        for i in range(5)
    ]
    inter = eng.add_request("i0", [9, 9], SamplingParams(max_tokens=3))
    eng.step()
    assert inter.output_ids, "interactive request should run first"
    assert all(not b.output_ids for b in batch)


def test_aging_bounds_starvation_of_batch_work() -> None:
    aging = 15
    eng, runner = engine(seqs=1, aging_steps=aging)
    starving = eng.add_request(
        "batch", [7, 7], SamplingParams(max_tokens=2), priority=Priority.BATCH
    )
    steps = 0
    # A continuous interactive stream that would starve BATCH forever without aging.
    while not starving.is_finished:
        eng.add_request(f"i{steps}", [steps % 50, 1], SamplingParams(max_tokens=2))
        eng.step()
        steps += 1
        assert steps < 200, "batch request starved"
    first = next(
        i for i, s in enumerate(runner.calls) if any(c.request is starving for c in s.chunks)
    )
    assert (
        aging <= first <= aging + 3
    )  # admitted right after promotion (one in-flight request ahead)
    assert starving.output_ids == reference([7, 7], 2)


def test_preempted_request_resumes_before_new_arrivals_of_same_priority() -> None:
    bm = BlockManager(10, block_size=4, watermark=0.0)
    runner = FakeRunner(bm)
    eng = LLMEngine(runner, bm, SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64))
    reqs = [eng.add_request(f"r{i}", [i, 2, 3], SamplingParams(max_tokens=25)) for i in range(4)]
    while not any(r.num_preemptions for r in reqs):
        eng.step()
    victim = next(r for r in reqs if r.num_preemptions)
    late = eng.add_request("late", [1, 1, 1], SamplingParams(max_tokens=2))
    while victim.status.value == "waiting" and late.status.value == "waiting":
        eng.step()
    assert victim.status.value != "waiting", "a new arrival overtook the preempted request"
