"""The engine in a child process (`--engine process`): the whole HTTP API suite, plus process life.

Every test of ``test_http_api`` and ``test_metrics_admin`` runs again here with the engine in a
spawned child process instead of a thread (the ``server``/``make_server`` fixtures below
override the shared ones). The tests that follow cover what only a process can do: crash, stop,
no weights in the API process. Tests that need a reference model get a separate copy loaded by the
fixture.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from dli.engine import SchedulerConfig
from dli.model import load_model
from dli.server.engine_process import ProcessEngine
from dli.server.factory import BuildOptions, build_server

from .conftest import MODEL, LiveServer, ServerFactory, start_server, stop_server, wait_for
from .test_http_api import *  # noqa: F403  (re-collected with the process-mode fixtures)
from .test_metrics_admin import *  # noqa: F403
from .test_server_process import _launch, _wait_ready


@pytest.fixture
def make_server(tiny_dir: Path) -> Iterator[ServerFactory]:
    started: list[LiveServer] = []

    def make(**overrides: object) -> LiveServer:
        sched = overrides.pop(
            "scheduler", SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=128)
        )
        opts = BuildOptions(
            model_name=MODEL,
            num_blocks=int(overrides.pop("num_blocks", 256)),  # type: ignore[call-overload]
            block_size=4,
            scheduler=sched,
            engine_process=True,
            **overrides,
        )  # type: ignore[arg-type]
        live = start_server(build_server(tiny_dir, opts))
        # Test-side reference copy only (the API process itself holds no weights in this mode).
        live.bundle.model = load_model(tiny_dir)
        started.append(live)
        return live

    yield make
    for live in started:
        stop_server(live)


@pytest.fixture
def server(make_server: ServerFactory) -> LiveServer:
    return make_server()


def _engine(live: LiveServer) -> ProcessEngine:
    engine = live.bundle.engine
    assert isinstance(engine, ProcessEngine)
    return engine


def _complete(live: LiveServer, prompt: str, max_tokens: int = 8) -> httpx.Response:
    with live.client() as c:
        return c.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True},
        )


def test_api_process_holds_no_weights_and_matches_the_threaded_engine(tiny_dir: Path) -> None:
    opts = {"model_name": MODEL, "num_blocks": 256, "block_size": 4}
    threaded = start_server(build_server(tiny_dir, BuildOptions(**opts)))  # type: ignore[arg-type]
    proc = start_server(build_server(tiny_dir, BuildOptions(**opts, engine_process=True)))  # type: ignore[arg-type]
    try:
        assert proc.bundle.model is None
        assert _engine(proc).pid not in (None, os.getpid())
        for p in ("alpha", "beta beta", "gamma!"):
            a, b = _complete(threaded, p, 12), _complete(proc, p, 12)
            assert a.status_code == b.status_code == 200
            assert a.json()["choices"][0]["text"] == b.json()["choices"][0]["text"]
            assert a.json()["usage"] == b.json()["usage"]
    finally:
        stop_server(threaded)
        stop_server(proc)


def test_snapshot_crosses_the_process_boundary(server: LiveServer) -> None:
    assert _complete(server, "count me", 6).status_code == 200
    with server.admin() as a:
        state = a.get("/admin/state").json()["engine"]
    assert state["total_steps"] > 0
    assert state["total_generated_tokens"] >= 6
    assert state["kv_total_blocks"] == 256
    assert state["healthy"] is True


def test_engine_process_crash_fails_inflight_requests_and_readiness(
    make_server: ServerFactory,
) -> None:
    live = make_server(num_blocks=1024)  # room for the long request below
    engine = _engine(live)
    with live.client() as c:
        assert c.get("/readyz").status_code == 200
        with c.stream(
            "POST",
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": "doomed",
                "max_tokens": 3000,
                "stream": True,
                "ignore_eos": True,
            },
        ) as r:
            lines = r.iter_lines()
            next(line for line in lines if line.startswith("data: "))
            assert engine.pid is not None
            os.kill(engine.pid, signal.SIGKILL)
            rest = [line for line in lines if line.startswith("data: ")]
        assert any("engine failure" in line for line in rest), rest[-3:]
        wait_for(lambda: not engine.healthy)
        assert c.get("/readyz").status_code == 503
        late = c.post("/v1/completions", json={"model": MODEL, "prompt": "late"})
        assert late.status_code == 503, late.text


def test_stop_ends_the_engine_process(make_server: ServerFactory) -> None:
    live = make_server()
    engine = _engine(live)
    assert _complete(live, "bye").status_code == 200
    pid = engine.pid
    assert pid is not None
    stop_server(live)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("engine process still alive after stop()")


def test_sigterm_drains_and_leaves_no_engine_process_behind(tiny_dir: Path) -> None:
    proc, port, admin_port = _launch(tiny_dir, "--no-auth", "--engine", "process")
    try:
        _wait_ready(port, proc)
        state = httpx.get(f"http://127.0.0.1:{admin_port}/admin/state", timeout=10).json()
        assert state["engine"]["healthy"] is True
        children = Path(f"/proc/{proc.pid}/task/{proc.pid}/children").read_text().split()
        assert children, "engine child process not found"
        with (
            httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as c,
            c.stream(
                "POST",
                "/v1/completions",
                json={
                    "model": "tiny",
                    "prompt": "drain",
                    "max_tokens": 400,
                    "stream": True,
                    "ignore_eos": True,
                },
            ) as r,
        ):
            lines = r.iter_lines()
            next(line for line in lines if line.startswith("data: "))
            proc.send_signal(signal.SIGTERM)
            rest = [line for line in lines if line.startswith("data: ")]
        assert rest[-1] == "data: [DONE]", "in-flight stream was cut off instead of drained"
        assert proc.wait(timeout=60) == 0
        for child in children:
            assert (
                not Path(f"/proc/{child}").exists()
                or "zombie" in Path(f"/proc/{child}/status").read_text().lower()
            ), f"engine process {child} outlived the server"
    finally:
        if proc.poll() is None:
            proc.kill()
