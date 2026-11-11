"""Start and stop the servers under test, one at a time, with their exact command lines recorded.

* ``dli``  : this project's server (``python -m dli.server``), auth off, loopback only.
* ``vllm`` : vLLM's OpenAI server in its official Docker image, pinned by digest, published on
  127.0.0.1 only. The Hugging Face cache is mounted read-only so both engines load identical files.
* ``hf``   : the static-batching ``transformers`` baseline (:mod:`dli.bench.hf_server`).

Every backend is configured from the same campaign-level settings (model, dtype, max sequences,
batched-token budget, memory budget); backend-specific flags are listed in the plan, so the exact
configuration of each side is part of the results.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# vLLM v0.32.0 (2026-10-18), x86_64 default build: torch 2.13.0+cu130 (CUDA 13.0), which the
# host driver (580, CUDA 13.0) supports. The CUDA 12.9 variant of the same release was tried first
# and cannot start: its torchcodec needs libnvrtc.so.13, which that image does not ship.
VLLM_IMAGE = (
    "vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9"
)

DLI_METRICS = (
    "dli_kv_blocks_used",
    "dli_kv_blocks_total",
    "dli_engine_running_requests",
    "dli_engine_waiting_requests",
    "dli_prefix_cache_hit_ratio",
    "dli_engine_preemptions_total",
)
VLLM_METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:num_preemptions_total",
)


@dataclass
class Running:
    name: str
    base_url: str
    model_name: str
    cmd: list[str]
    proc: subprocess.Popen[bytes] | None = None
    container: str | None = None
    metrics_url: str | None = None
    metric_names: tuple[str, ...] = ()
    log_path: Path | None = None
    startup_s: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cmd": shlex.join(self.cmd),
            "startup_s": round(self.startup_s, 1),
            **self.info,
        }


def _wait_ready(r: Running, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if r.proc is not None and r.proc.poll() is not None:
            raise RuntimeError(f"{r.name} exited during startup (see {r.log_path})")
        try:
            if httpx.get(f"{r.base_url}/v1/models", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"{r.name} not ready after {timeout_s:.0f}s (see {r.log_path})")


def start(
    name: str,
    *,
    model_dir: Path,
    model_name: str,
    port: int,
    dtype: str,
    device: str,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    log_dir: Path,
    settings: dict[str, Any],
) -> Running:
    """Start backend ``name``; ``settings`` holds backend-specific knobs from the plan."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}-{int(time.time())}.log"
    base = f"http://127.0.0.1:{port}"
    t0 = time.monotonic()
    if name == "dli":
        admin = port + 1
        cmd = [
            sys.executable,
            "-m",
            "dli.server",
            "--model",
            str(model_dir),
            "--served-model-name",
            model_name,
            "--port",
            str(port),
            "--admin-port",
            str(admin),
            "--no-auth",
            "--device",
            device,
            "--dtype",
            dtype,
            "--max-num-seqs",
            str(max_num_seqs),
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
            "--kv-cache-gb",
            str(settings["kv_cache_gb"]),
            "--block-size",
            str(settings.get("block_size", 16)),
            "--max-queued-requests",
            str(settings.get("max_queued_requests", 4096)),
            "--max-inflight-requests",
            str(settings.get("max_inflight_requests", 8192)),
            "--request-timeout",
            str(settings.get("request_timeout_s", 1800)),
            *settings.get("extra_args", []),
        ]
        r = Running(
            name,
            base,
            model_name,
            cmd,
            metrics_url=f"http://127.0.0.1:{admin}/metrics",
            metric_names=DLI_METRICS,
            log_path=log_path,
        )
        r.proc = subprocess.Popen(
            cmd,
            stdout=log_path.open("wb"),
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    elif name == "hf":
        cmd = [
            sys.executable,
            "-m",
            "dli.bench.hf_server",
            "--model",
            str(model_dir),
            "--served-model-name",
            model_name,
            "--port",
            str(port),
            "--device",
            device,
            "--dtype",
            dtype,
            "--max-batch",
            str(settings.get("max_batch", 16)),
            "--max-wait-ms",
            str(settings.get("max_wait_ms", 20)),
        ]
        r = Running(name, base, model_name, cmd, log_path=log_path)
        r.proc = subprocess.Popen(cmd, stdout=log_path.open("wb"), stderr=subprocess.STDOUT)
    elif name == "vllm":
        container = f"dli-bench-vllm-{port}"
        hf_home = Path(settings.get("hf_cache", Path.home() / ".cache" / "huggingface"))
        inner = Path("/root/.cache/huggingface") / model_dir.relative_to(hf_home)
        cmd = [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container,
            "--gpus",
            "all",
            "--ipc=host",
            "-p",
            f"127.0.0.1:{port}:8000",
            "-v",
            f"{hf_home}:/root/.cache/huggingface:ro",
            "-e",
            "HF_HUB_OFFLINE=1",
            settings.get("image", VLLM_IMAGE),
            "--model",
            str(inner),
            "--served-model-name",
            model_name,
            "--dtype",
            dtype,
            "--max-num-seqs",
            str(max_num_seqs),
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(settings["gpu_memory_utilization"]),
            "--max-model-len",
            str(settings.get("max_model_len", 4096)),
            "--seed",
            "0",
            *settings.get("extra_args", []),
        ]
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)
        subprocess.run(cmd, check=True, capture_output=True)
        r = Running(
            name,
            base,
            model_name,
            cmd,
            container=container,
            metrics_url=f"{base}/metrics",
            metric_names=VLLM_METRICS,
            log_path=log_path,
        )
    else:
        raise ValueError(f"unknown backend {name!r}")
    try:
        _wait_ready(r, float(settings.get("startup_timeout_s", 900)))
    except Exception:
        stop(r)
        raise
    r.startup_s = time.monotonic() - t0
    return r


def stop(r: Running) -> None:
    if r.container:
        if r.log_path:
            logs = subprocess.run(["docker", "logs", r.container], capture_output=True, check=False)
            r.log_path.write_bytes(logs.stdout + logs.stderr)
        subprocess.run(
            ["docker", "stop", "-t", "20", r.container], capture_output=True, check=False
        )
    if r.proc and r.proc.poll() is None:
        r.proc.terminate()
        try:
            r.proc.wait(60)
        except subprocess.TimeoutExpired:
            r.proc.kill()
            r.proc.wait(10)
