"""Environment capture for reproducible test and benchmark evidence.

Every result directory stores the output of :func:`collect` as ``env.json`` so a number can always
be traced to the exact hardware, driver, library versions, and code revision that produced it
(ENGINEERING_STANDARDS.md §3). Run as ``python -m dli.envinfo [output.json]``.

The collector is best-effort by design: a missing tool (e.g. ``nvidia-smi`` on a CPU-only laptop)
is recorded as ``None`` instead of raising, because a partial manifest is still evidence and a
crash here would block the measurement it is meant to document.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Fields queried from nvidia-smi. Kept explicit so the manifest schema is stable across drivers.
_NVSMI_FIELDS = (
    "name",
    "uuid",
    "memory.total",
    "compute_cap",
    "driver_version",
    "pcie.link.gen.current",
    "pcie.link.width.current",
    "clocks.max.sm",
    "power.limit",
)
# `nvidia-smi -q` lines that change how GPU numbers must be interpreted (vGPU licensing, MIG).
_NVSMI_Q_KEYS = (
    "Product Architecture",
    "Virtualization Mode",
    "Host VGPU Mode",
    "vGPU Software Licensed Product",
    "License Status",
    "MIG Mode",
    "Persistence Mode",
)


def _run(cmd: list[str], timeout: float = 20.0) -> str | None:
    """Run a command and return stripped stdout, or ``None`` if unavailable or failing."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _mem_total_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 2)
    except (OSError, ValueError):
        pass
    return None


def _gpu_info() -> dict[str, Any] | None:
    csv = _run(
        ["nvidia-smi", f"--query-gpu={','.join(_NVSMI_FIELDS)}", "--format=csv,noheader,nounits"]
    )
    if csv is None:
        return None
    gpus = [
        dict(zip(_NVSMI_FIELDS, (v.strip() for v in row.split(",")), strict=False))
        for row in csv.splitlines()
        if row.strip()
    ]
    details: dict[str, str] = {}
    for line in (_run(["nvidia-smi", "-q"]) or "").splitlines():
        key, sep, value = line.strip().partition(":")
        if sep and key.strip() in _NVSMI_Q_KEYS and key.strip() not in details:
            details[key.strip()] = value.strip()
    return {"devices": gpus, "count": len(gpus), "details": details}


def _torch_info() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    info: dict[str, Any] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "num_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        info["cudnn"] = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        info["devices"] = [
            {
                "name": p.name,
                "capability": f"{p.major}.{p.minor}",
                "total_memory_gib": round(p.total_memory / 2**30, 2),
                "multiprocessors": p.multi_processor_count,
            }
            for p in (torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count()))
        ]
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    return info


def current_revision() -> str | None:
    """Return the code revision of this checkout *now*.

    Long-running measurements must call this at START and pass the value to :func:`collect`:
    reading it when results are written would attribute the run to whatever was committed in the
    meantime (a real provenance bug found in the first M1 parity run; see docs/known-issues.md).
    """
    # Synced GPU-host trees have no .git; scripts/gpu/sync.sh writes REVISION instead.
    rev_file = Path(__file__).resolve().parents[2] / "REVISION"
    if rev_file.is_file():
        return rev_file.read_text().strip()
    rev = _run(["git", "describe", "--always", "--dirty", "--abbrev=12"])
    if rev is None or rev.endswith("-dirty"):
        return rev
    # `git describe --dirty` ignores untracked files, so a run of a brand-new, uncommitted test
    # file would be reported as clean (happened during M2). Untracked, non-ignored files => dirty.
    untracked = _run(["git", "status", "--porcelain", "--untracked-files=normal"])
    return f"{rev}-dirty" if untracked else rev


def collect(revision: str | None = None) -> dict[str, Any]:
    """Collect a JSON-serializable description of the current execution environment.

    Args:
        revision: Code revision captured when the measured run started. If omitted, the
            revision is read now, which is only correct for short, immediate runs.

    Returns:
        Mapping with timestamp, code revision, OS, CPU, memory, Python, PyTorch, and GPU details.
        Unavailable components are ``None``.

    Side effects:
        Spawns short-lived ``nvidia-smi`` and ``git`` subprocesses when they are installed.
    """
    return {
        "schema": "dli-env/v1",
        "collected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "revision": revision if revision is not None else current_revision(),
        "hostname": platform.node(),
        "os": {"platform": platform.platform(), "kernel": platform.release()},
        "cpu": {"model": _cpu_model(), "logical_cores": os.cpu_count()},
        "memory_gib": _mem_total_gib(),
        "python": sys.version.split()[0],
        "torch": _torch_info(),
        "gpu": _gpu_info(),
    }


def main(argv: list[str] | None = None, *, revision: str | None = None) -> int:
    """CLI entry point: print the manifest, or write it to the given path."""
    args = sys.argv[1:] if argv is None else argv
    text = json.dumps(collect(revision), indent=2, sort_keys=True)
    if args:
        Path(args[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(args[0]).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
