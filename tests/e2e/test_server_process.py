"""The real `python -m dli.server` process: safe-default refusals and graceful SIGTERM drain."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .conftest import _free_port


def _launch(model_dir: Path, *extra: str) -> tuple[subprocess.Popen[str], int, int]:
    port, admin_port = _free_port(), _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "dli.server",
            "--model",
            str(model_dir),
            "--served-model-name",
            "tiny",
            "--port",
            str(port),
            "--admin-port",
            str(admin_port),
            "--num-blocks",
            "1024",
            "--block-size",
            "4",
            *extra,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return proc, port, admin_port


def _wait_ready(port: int, proc: subprocess.Popen[str], timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"server exited early: {proc.stdout.read() if proc.stdout else ''}"
            )
        try:
            if httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise AssertionError("server never became ready")


def test_refuses_unauthenticated_public_bind(tiny_dir: Path) -> None:
    proc, _, _ = _launch(tiny_dir, "--no-auth", "--host", "0.0.0.0")
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode != 0
    assert "refusing --no-auth on a non-loopback --host" in out


def test_requires_keys_by_default(tiny_dir: Path) -> None:
    proc, _, _ = _launch(tiny_dir)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode != 0
    assert "--api-keys-file" in out


def test_sigterm_drains_inflight_stream_then_exits_cleanly(tiny_dir: Path) -> None:
    proc, port, _ = _launch(tiny_dir, "--no-auth")
    try:
        _wait_ready(port, proc)
        with (
            httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as c,
            c.stream(
                "POST",
                "/v1/completions",
                json={
                    "model": "tiny",
                    "prompt": "drain",
                    "max_tokens": 600,
                    "stream": True,
                    "ignore_eos": True,
                },
            ) as r,
        ):
            lines = r.iter_lines()
            next(line for line in lines if line.startswith("data: "))
            proc.send_signal(signal.SIGTERM)  # shutdown starts while the stream is in flight
            time.sleep(0.3)
            busy = httpx.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={"model": "tiny", "prompt": "late"},
                timeout=10,
            )
            rest = [line for line in lines if line.startswith("data: ")]
        assert busy.status_code == 503, busy.text  # drain refuses new work
        assert rest[-1] == "data: [DONE]", "in-flight stream was cut off instead of drained"
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
