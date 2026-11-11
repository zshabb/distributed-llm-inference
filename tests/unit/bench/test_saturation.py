"""Saturation detection for open-loop sweeps (queue growth, not a tail-biased achieved rate)."""

from __future__ import annotations

from dli.bench.campaign import saturated
from dli.bench.loadgen import RequestRecord, RunResult, run_metrics


def _run(ttfts: list[float], e2e: float, n_fail: int = 0) -> dict[str, object]:
    recs = [
        RequestRecord(
            f"r{i}",
            scheduled=float(i),
            sent=float(i),
            first_token=i + t,
            end=i + e2e,
            chunk_times=[i + t, i + e2e],
            output_tokens=10,
            status="ok",
        )
        for i, t in enumerate(ttfts)
    ]
    recs += [RequestRecord(f"f{i}", scheduled=0.0, status="timeout") for i in range(n_fail)]
    return run_metrics(RunResult("open", recs, 100.0, {"rate": 1.0}))


def test_regression_long_tails_alone_do_not_mean_saturation() -> None:
    """M5 finding: 40 arrivals at 1/s with 20 s outputs. The first rule divided completions by a
    span including the 20 s drain tail (40 / 60 s = 0.67 req/s < 0.9 x offered) and stopped the
    sweep, although TTFT was flat. Flat TTFT is a stable queue."""
    m = _run([0.1] * 40, e2e=20.0)
    assert m["completed_req_per_s"] < 0.9  # the biased quantity the old rule used
    assert not saturated(m, offered=1.0, ttft_limit=10.0)


def test_growing_ttft_is_saturation() -> None:
    m = _run([0.1 + 0.08 * i for i in range(40)], e2e=20.0)  # queue grows through the run
    assert m["ttft_last_third_p50_s"] > 2 * m["ttft_first_third_p50_s"] + 0.5
    assert saturated(m, offered=1.0, ttft_limit=10.0)


def test_incomplete_or_slow_runs_are_saturated() -> None:
    assert saturated(_run([0.1] * 40, e2e=2.0, n_fail=2), offered=1.0, ttft_limit=10.0)
    assert saturated(_run([12.0] * 40, e2e=20.0), offered=1.0, ttft_limit=10.0)
