"""Overload behaviour over real HTTP: fast, honest rejections and no leaked engine work."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from dli.engine import SchedulerConfig
from dli.server.admission import AdmissionConfig

from .conftest import MODEL, LiveServer, ServerFactory, wait_for

ONE_SLOT = SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64)
BIG_POOL = 1024  # blocks x 4 = 4096 token slots, so a 3000-token holder fits


def _hold_slot(live: LiveServer, started: threading.Event, stop: threading.Event) -> None:
    """Occupy the engine's only sequence slot with a long stream until told to stop."""
    with (
        live.client() as c,
        c.stream(
            "POST",
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": "hold",
                "max_tokens": 3000,
                "stream": True,
                "ignore_eos": True,
            },
        ) as r,
    ):
        for line in r.iter_lines():
            if line.startswith("data: "):
                started.set()
            if stop.is_set():
                return


def test_inflight_cap_rejects_fast_with_retry_after(make_server: ServerFactory) -> None:
    live = make_server(scheduler=ONE_SLOT, admission=AdmissionConfig(max_inflight_requests=2))

    def one(i: int) -> tuple[int, float, httpx.Response]:
        with live.client() as c:
            t0 = time.perf_counter()
            r = c.post(
                "/v1/completions", json={"model": MODEL, "prompt": f"p{i}", "max_tokens": 60}
            )
            return i, time.perf_counter() - t0, r

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(one, range(8)))
    rejected = [(dt, r) for _, dt, r in results if r.status_code == 429]
    accepted = [r for _, _, r in results if r.status_code == 200]
    assert rejected, "cap of 2 in-flight requests should reject some of 8 concurrent requests"
    assert accepted
    for dt, r in rejected:
        assert r.json()["error"]["code"] == "server_busy"
        assert int(r.headers["Retry-After"]) >= 1
        assert dt < 0.5, f"rejection took {dt:.3f}s; overload responses must be fast"


def test_queue_cap(make_server: ServerFactory) -> None:
    live = make_server(
        scheduler=ONE_SLOT, num_blocks=BIG_POOL, admission=AdmissionConfig(max_queued_requests=1)
    )
    started, stop = threading.Event(), threading.Event()
    holder = threading.Thread(target=_hold_slot, args=(live, started, stop))
    holder.start()
    try:
        assert started.wait(10)
        engine = live.bundle.engine
        waiter = threading.Thread(
            target=lambda: live.client().post(
                "/v1/completions", json={"model": MODEL, "prompt": "w", "max_tokens": 5}
            )
        )
        waiter.start()
        wait_for(lambda: engine.snapshot.num_waiting == 1)
        with live.client() as c:
            r = c.post("/v1/completions", json={"model": MODEL, "prompt": "x", "max_tokens": 5})
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "queue_full"
    finally:
        stop.set()
        holder.join(30)
    waiter.join(30)


def test_unattainable_deadline_is_rejected_up_front(make_server: ServerFactory) -> None:
    live = make_server()
    with live.client() as c:
        # Warm-up gives the admission controller a measured prefill rate.
        assert (
            c.post(
                "/v1/completions", json={"model": MODEL, "prompt": "warm up " * 4, "max_tokens": 4}
            ).status_code
            == 200
        )
        t0 = time.perf_counter()
        r = c.post(
            "/v1/completions",
            headers={"X-Deadline-Ms": "1"},
            json={"model": MODEL, "prompt": "y" * 200, "max_tokens": 4},
        )
        dt = time.perf_counter() - t0
    assert r.status_code == 429, r.text
    assert r.json()["error"]["code"] == "deadline_unattainable"
    assert dt < 0.5
    assert live.bundle.engine.snapshot.num_waiting == 0, (
        "rejected request must not reach the engine"
    )


def test_deadline_exceeded_while_queued_aborts_the_request(make_server: ServerFactory) -> None:
    live = make_server(scheduler=ONE_SLOT, num_blocks=BIG_POOL)
    started, stop = threading.Event(), threading.Event()
    holder = threading.Thread(target=_hold_slot, args=(live, started, stop))
    holder.start()
    try:
        assert started.wait(10)
        # Simulate an optimistic prediction (it is an estimate): the request is admitted, and the
        # deadline must still be enforced while it waits for the busy slot.
        live.bundle.app.state.admission.predicted_ttft_s = lambda snap, n: None
        with live.client() as c:
            t0 = time.perf_counter()
            req = {"model": MODEL, "prompt": "q", "max_tokens": 5, "stream": True}
            r = c.post("/v1/completions", headers={"X-Deadline-Ms": "300"}, json=req)
            dt = time.perf_counter() - t0
        # The stream never started: the miss is a real HTTP 429, not an in-band event.
        assert r.status_code == 429, r.text
        assert r.json()["error"]["code"] == "deadline_exceeded"
        assert 0.3 <= dt < 1.5
        engine = live.bundle.engine
        wait_for(lambda: engine.snapshot.num_waiting == 0)  # aborted, not left queued
    finally:
        stop.set()
        holder.join(30)


def test_busy_slot_is_predicted_and_rejected_up_front(make_server: ServerFactory) -> None:
    """The step model sees that the only sequence slot is held by a long request."""
    live = make_server(scheduler=ONE_SLOT, num_blocks=BIG_POOL)
    started, stop = threading.Event(), threading.Event()
    holder = threading.Thread(target=_hold_slot, args=(live, started, stop))
    holder.start()
    try:
        assert started.wait(10)
        with live.client() as c:
            t0 = time.perf_counter()
            req = {"model": MODEL, "prompt": "q", "max_tokens": 5}
            r = c.post("/v1/completions", headers={"X-Deadline-Ms": "300"}, json=req)
            dt = time.perf_counter() - t0
        assert r.status_code == 429, r.text
        assert r.json()["error"]["code"] == "deadline_unattainable"
        assert dt < 0.25  # decided at admission, without waiting for the deadline
    finally:
        stop.set()
        holder.join(30)


def test_draining_rejects_new_work_but_finishes_inflight(make_server: ServerFactory) -> None:
    live = make_server()
    with (
        live.client() as c,
        c.stream(
            "POST",
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": "z",
                "max_tokens": 40,
                "stream": True,
                "temperature": 0,
            },
        ) as r,
    ):
        it = r.iter_lines()
        next(line for line in it if line.startswith("data: "))
        live.bundle.app.state.draining = True
        with live.client() as c2:
            rej = c2.post("/v1/completions", json={"model": MODEL, "prompt": "n", "max_tokens": 2})
            assert rej.status_code == 503
            assert rej.json()["error"]["code"] == "draining"
            assert c2.get("/readyz").status_code == 503
        rest = [line for line in it if line.startswith("data: ")]
    assert rest[-1] == "data: [DONE]", "in-flight stream must complete during drain"


def test_invalid_deadline_header(make_server: ServerFactory) -> None:
    live = make_server()
    with live.client() as c:
        r = c.post(
            "/v1/completions", headers={"X-Deadline-Ms": "-5"}, json={"model": MODEL, "prompt": "x"}
        )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_header"
