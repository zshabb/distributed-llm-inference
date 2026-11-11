"""vLLM baseline launcher: Docker where available, else the pinned venv; one argument list."""

from __future__ import annotations

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


def test_backend_env_reaches_both_launchers_and_is_recorded() -> None:
    src = Path(backends.__file__).read_text()
    # Docker gets -e K=V; the venv process gets it merged into its environment; both record it.
    assert '("-e", f"{k}={v}")' in src
    assert "**venv_env}" in src
    assert 'r.info["env"] = venv_env' in src
