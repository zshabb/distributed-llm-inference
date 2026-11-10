"""Security properties over real HTTP: authentication order, tenant integrity, body limits."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from dli.engine import SchedulerConfig
from dli.server.auth import ApiKeyAuth, new_key

from .conftest import MODEL, LiveServer, ServerFactory, wait_for


@pytest.fixture
def secured(make_server: ServerFactory) -> tuple[LiveServer, str]:
    token, record = new_key("acme")
    one_slot = SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64)
    live = make_server(
        auth=ApiKeyAuth([record]), max_body_bytes=4096, num_blocks=1024, scheduler=one_slot
    )
    return live, token


def test_missing_and_invalid_keys_get_401(secured: tuple[LiveServer, str]) -> None:
    live, token = secured
    body = {"model": MODEL, "prompt": "x", "max_tokens": 2}
    with live.client() as c:
        r = c.post("/v1/completions", json=body)
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"
        assert r.json()["error"]["type"] == "authentication_error"
        bad = {"Authorization": "Bearer dli_ffffffffffff_nope"}
        assert c.post("/v1/completions", json=body, headers=bad).status_code == 401
        assert c.get("/v1/models").status_code == 401
        ok = c.post("/v1/completions", json=body, headers={"Authorization": f"Bearer {token}"})
        assert ok.status_code == 200, ok.text
        # Probes stay unauthenticated for orchestrators.
        assert c.get("/healthz").status_code == 200


def test_authentication_runs_before_request_validation(secured: tuple[LiveServer, str]) -> None:
    live, _ = secured
    with live.client() as c:
        # Malformed AND unauthenticated: must be 401, not a 400 that reveals the schema.
        r = c.post("/v1/completions", json={"temperature": "nonsense"})
    assert r.status_code == 401


def test_client_supplied_tenant_header_is_ignored(secured: tuple[LiveServer, str]) -> None:
    live, token = secured
    stop = threading.Event()

    def stream() -> None:
        with (
            live.client() as c,
            c.stream(
                "POST",
                "/v1/completions",
                headers={"Authorization": f"Bearer {token}", "X-Tenant-Id": "someone-else"},
                json={
                    "model": MODEL,
                    "prompt": "t",
                    "max_tokens": 3000,
                    "stream": True,
                    "ignore_eos": True,
                },
            ) as r,
        ):
            for _ in r.iter_lines():
                if stop.is_set():
                    return

    t = threading.Thread(target=stream)
    t.start()
    try:
        sched = live.bundle.engine._engine.scheduler  # test-only read of the engine's state
        wait_for(lambda: bool(sched.running))
        assert sched.running[0].tenant == "acme"
    finally:
        stop.set()
        t.join(30)


def test_oversized_bodies_rejected_with_413(secured: tuple[LiveServer, str]) -> None:
    live, token = secured
    auth = {"Authorization": f"Bearer {token}"}
    big = {"model": MODEL, "prompt": "x" * 10_000}
    with live.client() as c:
        r = c.post("/v1/completions", json=big, headers=auth)
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "body_too_large"

        def chunks() -> Iterator[bytes]:  # chunked upload: no Content-Length to check up front
            yield b'{"model": "' + MODEL.encode() + b'", "prompt": "'
            for _ in range(20):
                yield b"y" * 1000
            yield b'"}'

        r = c.post(
            "/v1/completions",
            content=chunks(),
            headers={**auth, "content-type": "application/json"},
        )
        assert r.status_code == 413
