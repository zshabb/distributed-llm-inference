"""TTFT prediction for deadline admission (step model)."""

from __future__ import annotations

import pytest

from dli.server.admission import AdmissionConfig, AdmissionController
from dli.server.async_engine import EngineSnapshot

GATE = AdmissionController(AdmissionConfig())


def snap(**kw: object) -> EngineSnapshot:
    base: dict[str, object] = {
        "step_seconds": 0.2,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 128,
    }
    return EngineSnapshot(**{**base, **kw})  # type: ignore[arg-type]


def test_regression_decode_heavy_engine_is_not_mistaken_for_slow_prefill() -> None:
    """M5 finding: 60 decoding sequences make every step ~0.2 s. The old predictor divided queued
    prefill by a 'prefill rate' measured on such steps (~200 tok/s) and predicted 5.6 s, rejecting
    requests whose real TTFT was ~0.1-0.2 s. One budgeted step absorbs the whole queue here."""
    s = snap(
        num_running=60,
        waiting_prefill_tokens=600,
        prefill_tokens_per_s=200.0,
        running_remaining_tokens=tuple(range(10, 70)),
    )
    predicted = GATE.predicted_ttft_s(s, 512)
    assert predicted == pytest.approx(2 * 0.2)  # one prefill step + the first-token step
    assert GATE.check(s, 512, deadline_ms=2000, draining=False) is None


def test_prefill_backlog_drains_at_the_budget_left_after_decode() -> None:
    s = snap(num_running=48, waiting_prefill_tokens=10_000)
    # (10_000 + 500) / (2048 - 48) = 5.25 -> 6 steps, + 1
    assert GATE.predicted_ttft_s(s, 500) == pytest.approx(7 * 0.2)


def test_full_sequence_slots_wait_for_the_soonest_finisher() -> None:
    s = snap(num_running=128, num_waiting=2, running_remaining_tokens=(5, 9, 40) + (100,) * 125)
    # third in line: needs 3 completions -> 40 steps, + 1
    assert GATE.predicted_ttft_s(s, 10) == pytest.approx(41 * 0.2)
    rejection = GATE.check(s, 10, deadline_ms=1000, draining=False)
    assert rejection is not None
    assert rejection.code == "deadline_unattainable"


def test_more_waiters_than_running_requests_queue_for_multiple_rounds() -> None:
    s = snap(max_num_seqs=2, num_running=2, num_waiting=4, running_remaining_tokens=(3, 7))
    # need 5 completions from 2 slots: 2 full rounds of the longest (7) + the 1st soonest (3) = 17
    assert GATE.predicted_ttft_s(s, 1) == pytest.approx(18 * 0.2)


def test_cold_start_admits_without_a_prediction() -> None:
    assert GATE.predicted_ttft_s(EngineSnapshot(), 100) is None
    assert GATE.check(EngineSnapshot(), 100, deadline_ms=1, draining=False) is None
