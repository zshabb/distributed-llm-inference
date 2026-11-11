"""vLLM baseline launcher: Docker where available, else the pinned venv; one argument list."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from dli.bench import backends


def test_vllm_launcher_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.shutil, "which", lambda _n: "/usr/bin/docker")
    assert backends.vllm_launcher({}) == "docker"
    monkeypatch.setattr(backends.shutil, "which", lambda _n: None)
    assert backends.vllm_launcher({}) == "venv"  # e.g. a RunPod pod
    assert backends.vllm_launcher({"launcher": "docker"}) == "docker"


def test_vllm_args_are_shared_by_both_launchers() -> None:
    args = backends.vllm_args(
        model_name="m",
        dtype="bfloat16",
        max_num_seqs=128,
        max_num_batched_tokens=2048,
        settings={"gpu_memory_utilization": 0.85, "extra_args": ["--enable-prefix-caching"]},
    )
    assert args[args.index("--max-num-seqs") + 1] == "128"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"
    assert args[-1] == "--enable-prefix-caching"
    assert "--host" not in args  # the launcher decides the bind address


FAKE_VLLM = """#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
with open(os.environ["FAKE_VLLM_DUMP"], "w") as f:
    json.dump({"args": args, "env": dict(os.environ)}, f)
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data": []}')
    def log_message(self, *a):
        pass
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def test_venv_launcher_runs_loopback_with_plan_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The venv launcher starts the process on 127.0.0.1, offline, with the plan env."""
    fake = tmp_path / "python"
    fake.write_text(FAKE_VLLM)
    fake.chmod(0o755)
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_VLLM_DUMP", str(dump))
    port = _free_port()
    srv = backends.start(
        "vllm",
        model_dir=tmp_path,
        model_name="m",
        port=port,
        dtype="bfloat16",
        device="cuda",
        max_num_seqs=8,
        max_num_batched_tokens=256,
        log_dir=tmp_path / "logs",
        settings={
            "launcher": "venv",
            "venv_python": str(fake),
            "gpu_memory_utilization": 0.5,
            "env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
            "startup_timeout_s": 30,
        },
    )
    try:
        seen = json.loads(dump.read_text())
        assert seen["args"][seen["args"].index("--host") + 1] == "127.0.0.1"
        assert seen["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
        assert seen["env"]["HF_HUB_OFFLINE"] == "1"
        assert srv.describe()["launcher"] == "venv"
        assert srv.describe()["env"] == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    finally:
        backends.stop(srv)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port
