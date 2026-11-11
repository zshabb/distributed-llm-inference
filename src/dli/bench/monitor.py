"""Background resource sampling during a benchmark run.

* GPU (NVML, device 0): memory used and SM utilization. NVML reports whole-device memory, so it
  includes every process on the GPU; runs execute one server at a time for that reason.
* Server process: resident set size from ``/proc/<pid>/status`` (CPU runs).
* Prometheus: selected series scraped from the backend's metrics endpoint (KV usage, queue depth,
  prefix-cache hits) so engine-internal state can be read next to client-side latency.

Sampling runs in a daemon thread, so it never blocks the event loop that drives the load.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families


def _rss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def scrape(url: str, names: Sequence[str], client: httpx.Client) -> dict[str, float]:
    """Sum each selected metric family over its label sets (e.g. one value per family)."""
    out: dict[str, float] = {}
    text = client.get(url, timeout=5).text
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            if s.name in names:
                out[s.name] = out.get(s.name, 0.0) + float(s.value)
    return out


@dataclass
class Monitor:
    interval_s: float = 0.5
    pid: int | None = None
    metrics_url: str | None = None
    metric_names: Sequence[str] = ()
    gpu: bool = True
    samples: list[dict[str, Any]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> Monitor:
        self._handle: Any = None
        if self.gpu:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._handle = None
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(5)

    def _loop(self) -> None:
        with httpx.Client() as client:
            while not self._stop.is_set():
                s: dict[str, Any] = {"t": time.perf_counter() - self._t0}
                if self._handle is not None:
                    import pynvml

                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    s["gpu_mem_used"] = int(mem.used)
                    s["gpu_util"] = int(pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
                if self.pid is not None:
                    s["rss"] = _rss_bytes(self.pid)
                if self.metrics_url:
                    with contextlib.suppress(httpx.HTTPError, ValueError):
                        s.update(scrape(self.metrics_url, self.metric_names, client))
                self.samples.append(s)
                self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, float]:
        """Peak and mean of every numeric series (mean excludes missing samples)."""
        out: dict[str, float] = {}
        keys = {k for s in self.samples for k in s if k != "t"}
        for k in sorted(keys):
            vals = [float(s[k]) for s in self.samples if isinstance(s.get(k), int | float)]
            if vals:
                out[f"{k}_max"] = max(vals)
                out[f"{k}_mean"] = sum(vals) / len(vals)
        out["samples"] = len(self.samples)
        return out
