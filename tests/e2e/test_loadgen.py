"""The benchmark load generator against the real server over TCP (tiny model, CPU)."""

from __future__ import annotations

import asyncio
import itertools

from dli.bench.loadgen import (
    RequestRecord,
    RunResult,
    Target,
    closed_loop,
    open_loop,
    poisson_arrivals,
    run_metrics,
    sessions_open_loop,
)
from dli.bench.workloads import GenRequest, Session

from .conftest import MODEL, LiveServer


def _reqs(n: int, out: int = 6) -> list[GenRequest]:
    return [GenRequest(f"r{i}", out, prompt=tuple(range(5, 5 + 8 + i % 4))) for i in range(n)]


def test_poisson_arrivals_rate_and_determinism() -> None:
    a = poisson_arrivals(4000, 20.0, seed=1)
    assert a == poisson_arrivals(4000, 20.0, seed=1)
    assert a[0] == 0.0
    assert all(x <= y for x, y in itertools.pairwise(a))
    assert 0.9 * 20 < len(a) / a[-1] < 1.1 * 20


def test_open_loop_counts_tokens_and_measures_from_schedule(server: LiveServer) -> None:
    res = asyncio.run(open_loop(Target(server.url, MODEL), _reqs(20), rate=40.0, seed=3))
    m = run_metrics(res)
    assert m["ok"] == 20
    assert m["output_tokens"] == 20 * 6  # ignore_eos: exact lengths from the server's usage
    for r in res.records:
        assert r.first_token is not None
        assert r.end is not None
        assert r.sent >= r.scheduled - 1e-6
        assert r.prompt_tokens == len(_reqs(20)[int(r.rid[1:])].prompt or ())
    assert m["ttft_s"]["n"] == 20
    assert m["client_lag_s"]["p99"] < 0.25  # the client kept to its schedule


def test_latency_includes_client_lateness() -> None:
    """Coordinated omission guard: latency starts at the scheduled arrival, not the send."""
    rec = RequestRecord(
        "x",
        scheduled=0.0,
        sent=1.0,
        first_token=1.2,
        end=2.0,
        chunk_times=[1.2, 2.0],
        output_tokens=2,
        status="ok",
    )
    m = run_metrics(RunResult("open", [rec], 2.0, {}))
    assert m["ttft_s"]["p50"] == 1.2
    assert m["e2e_s"]["p50"] == 2.0
    assert m["tpot_s"]["p50"] == 0.8


def test_closed_loop_never_exceeds_concurrency(server: LiveServer) -> None:
    res = asyncio.run(closed_loop(Target(server.url, MODEL), _reqs(24, out=10), concurrency=3))
    assert run_metrics(res)["ok"] == 24
    events = sorted(
        [(r.sent, 1) for r in res.records] + [(r.end or 0.0, -1) for r in res.records],
        key=lambda e: (e[0], e[1]),
    )
    live = peak = 0
    for _, d in events:
        live += d
        peak = max(peak, live)
    assert peak <= 3


def test_rejections_are_classified_with_their_latency(server: LiveServer) -> None:
    server.bundle.app.state.draining = True
    res = asyncio.run(open_loop(Target(server.url, MODEL), _reqs(5), rate=50.0))
    m = run_metrics(res)
    assert m["rejected"] == 5
    assert m["ok"] == 0
    assert m["completion_rate"] == 0
    assert m["rejection_latency_s"]["n"] == 5
    assert m["ttft_s"] is None  # no fabricated latency for requests that never ran


def test_non_streaming_mode_times_the_whole_response(server: LiveServer) -> None:
    res = asyncio.run(closed_loop(Target(server.url, MODEL, stream=False), _reqs(4), concurrency=2))
    for r in res.records:
        assert r.status == "ok"
        assert r.first_token == r.end
        assert r.output_tokens == 6


def test_sessions_feed_replies_back_into_the_conversation(server: LiveServer) -> None:
    sessions = [Session(f"s{i}", "be brief", ("hello", "and then?", "thanks"), 5) for i in range(2)]
    res = asyncio.run(sessions_open_loop(Target(server.url, MODEL), sessions, 5.0, think_s=0.01))
    assert len(res.records) == 6
    assert all(r.status == "ok" for r in res.records)
    for sid in ("s0", "s1"):
        turns = sorted((r for r in res.records if r.session == sid), key=lambda r: r.turn)
        lens = [r.prompt_tokens or 0 for r in turns]
        # each turn's prompt holds the previous prompt, the 5-token reply and the new user turn
        assert lens[1] > lens[0] + 5
        assert lens[2] > lens[1] + 5
        assert turns[1].scheduled >= (turns[0].end or 0.0)  # sent only after the previous reply
