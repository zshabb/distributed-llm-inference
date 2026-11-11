"""Prometheus metrics for the server and engine.

Request-path metrics are updated by the API handlers. Engine state is read from the latest
:class:`EngineSnapshot` at scrape time by a custom collector, so scraping never touches the engine
thread and costs nothing between scrapes.

Each server owns its registry (no global default), so several servers in one process (tests) do not
collide.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.process_collector import ProcessCollector
from prometheus_client.registry import Collector

from dli.server.async_engine import EngineClient

# Latency buckets span sub-millisecond CPU decode steps up to multi-second queued prefills.
_LATENCY = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)


class _EngineCollector(Collector):
    def __init__(self, engine: EngineClient) -> None:
        self._engine = engine

    def collect(self) -> Iterator[Metric]:
        s = self._engine.snapshot
        gauges = {
            "dli_engine_waiting_requests": ("Requests waiting in the engine queue", s.num_waiting),
            "dli_engine_running_requests": (
                "Requests holding KV and being scheduled",
                s.num_running,
            ),
            "dli_engine_queued_prefill_tokens": (
                "Prompt tokens not yet computed",
                s.waiting_prefill_tokens,
            ),
            "dli_kv_blocks_used": ("KV blocks referenced by running requests", s.kv_used_blocks),
            "dli_kv_blocks_free": ("KV blocks on the free list", s.kv_free_blocks),
            "dli_kv_blocks_cached": (
                "Unused KV blocks kept for prefix reuse",
                s.kv_cached_idle_blocks,
            ),
            "dli_kv_blocks_total": ("KV blocks in the pool", s.kv_total_blocks),
            "dli_engine_prefill_tokens_per_second": (
                "EWMA prefill throughput",
                s.prefill_tokens_per_s,
            ),
            "dli_engine_step_seconds": ("EWMA engine step duration", s.step_seconds),
            "dli_engine_healthy": ("1 if the engine loop is running", 1.0 if s.healthy else 0.0),
            "dli_prefix_cache_hit_ratio": (
                "Prompt tokens served from the prefix cache / prompt tokens queried",
                s.prefix_hit_tokens / s.prefix_query_tokens if s.prefix_query_tokens else 0.0,
            ),
        }
        for name, (doc, value) in gauges.items():
            yield GaugeMetricFamily(name, doc, value=float(value))
        yield CounterMetricFamily("dli_engine_steps", "Engine steps executed", value=s.total_steps)
        yield CounterMetricFamily(
            "dli_engine_preemptions", "Requests preempted (recompute)", value=s.total_preemptions
        )
        if s.cuda_graph_status != "off":
            yield GaugeMetricFamily(
                "dli_cuda_graph_captured",
                "1 if at least one decode CUDA graph bucket is captured",
                value=1.0 if s.cuda_graph_status == "captured" else 0.0,
            )
            yield CounterMetricFamily(
                "dli_cuda_graph_replays",
                "Decode steps served by a CUDA graph replay",
                value=s.cuda_graph_replays,
            )
            fallbacks = CounterMetricFamily(
                "dli_cuda_graph_fallbacks",
                "Decode steps run eagerly although graphs are enabled, by reason",
                labels=["reason"],
            )
            for reason, count in s.cuda_graph_fallbacks:
                fallbacks.add_metric([reason], count)
            yield fallbacks
            yield GaugeMetricFamily(
                "dli_cuda_graph_memory_bytes",
                "GPU memory reserved by decode CUDA graph capture (buffers + pool)",
                value=float(s.cuda_graph_memory_bytes),
            )
        # The engine's process publishes its GPU memory with the snapshot (it may not be this one).
        allocated, reserved = s.gpu_memory_allocated_bytes, s.gpu_memory_reserved_bytes
        if not reserved and torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
            allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        if reserved:
            yield GaugeMetricFamily(
                "dli_gpu_memory_allocated_bytes",
                "PyTorch-allocated GPU memory of the engine",
                value=float(allocated),
            )
            yield GaugeMetricFamily(
                "dli_gpu_memory_reserved_bytes",
                "PyTorch-reserved GPU memory of the engine (allocator cache included)",
                value=float(reserved),
            )


class ServerMetrics:
    """All metrics for one server instance."""

    def __init__(self, engine: EngineClient) -> None:
        r = CollectorRegistry()
        self.registry = r
        self.requests = Counter(
            "dli_requests",
            "Generation requests by endpoint and HTTP status",
            ["endpoint", "status"],
            registry=r,
        )
        self.errors = Counter(
            "dli_request_errors", "Failed requests by error type", ["type"], registry=r
        )
        self.rejections = Counter(
            "dli_admission_rejections", "Requests refused by admission", ["reason"], registry=r
        )
        self.prompt_tokens = Counter(
            "dli_prompt_tokens", "Prompt tokens of admitted requests", registry=r
        )
        self.generation_tokens = Counter("dli_generation_tokens", "Tokens generated", registry=r)
        self.ttft = Histogram(
            "dli_time_to_first_token_seconds",
            "Arrival to first token",
            buckets=_LATENCY,
            registry=r,
        )
        self.itl = Histogram(
            "dli_inter_token_latency_seconds",
            "Time between consecutive generated tokens of one request",
            buckets=_LATENCY,
            registry=r,
        )
        self.latency = Histogram(
            "dli_request_latency_seconds", "Arrival to completion", buckets=_LATENCY, registry=r
        )
        r.register(_EngineCollector(engine))
        ProcessCollector(registry=r)  # CPU seconds, resident memory, open fds
