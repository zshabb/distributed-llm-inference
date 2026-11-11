"""Admission control and backpressure: refuse work early instead of queueing it into failure.

Checks, in order, each returning a fast rejection (no engine work done):

1. **Draining** (shutdown in progress) -> 503 ``draining``.
2. **In-flight cap** (HTTP-level concurrency, the backpressure bound) -> 429 ``server_busy``.
3. **Queue cap** (requests waiting in the engine) -> 429 ``queue_full``.
4. **Deadline** (``X-Deadline-Ms``): predicted time-to-first-token, in engine steps times the
   measured step time (see :meth:`AdmissionController.predicted_ttft_s`). If that exceeds the
   deadline, the request is rejected now (429 ``deadline_unattainable``) instead of failing after
   waiting. Without a measured step time yet (cold start) the request is admitted, and the deadline
   is still enforced while it waits (see ``create_app``: abort + 429 ``deadline_exceeded``).

``Retry-After`` is an honest estimate of when capacity frees up (rounded up, at least 1 s).
The prediction is a queueing approximation, not a guarantee. M5 measured the first version
(queued prefill tokens / an EWMA of prefill tokens per forward second) over-predicting by 10-50x
on a decode-heavy GPU engine: a step that admits one short prompt next to 60 decoding sequences
takes as long as the decode work, so it "measured" ~200 prefill tokens/s and rejected about half
of the requests whose real TTFT was ~0.1 s. The step model below follows how the scheduler works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dli.server.async_engine import EngineSnapshot


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    max_inflight_requests: int = 512
    max_queued_requests: int = 256

    def __post_init__(self) -> None:
        if self.max_inflight_requests < 1 or self.max_queued_requests < 0:
            raise ValueError("admission limits must be positive")


@dataclass(frozen=True, slots=True)
class Rejection:
    status: int
    code: str
    message: str
    retry_after_s: int | None = None


class AdmissionController:
    """Tracks in-flight HTTP requests and decides admission. Used from the asyncio loop only."""

    def __init__(self, config: AdmissionConfig) -> None:
        self.config = config
        self.inflight = 0

    def predicted_ttft_s(self, snap: EngineSnapshot, prompt_tokens: int) -> float | None:
        """Steps until this prompt can produce its first token, times the measured step time.

        * Prefill: each step has ``max_num_batched_tokens`` of budget, of which running requests
          take one decode token each; queued prompt tokens (including this one's) drain through
          the rest.
        * Slots: with ``k`` requests waiting and ``free`` sequence slots, this request also needs
          ``k + 1 - free`` running requests to finish; the ``j``-th soonest finishes after at most
          its remaining tokens (one per step).
        * Plus the step that produces the first token.
        """
        if snap.step_seconds <= 0 or snap.max_num_batched_tokens <= 0:
            return None
        capacity = max(1, snap.max_num_batched_tokens - snap.num_running)
        prefill_steps = math.ceil((snap.waiting_prefill_tokens + prompt_tokens) / capacity)
        slot_steps = 0
        need = snap.num_waiting + 1 - max(0, snap.max_num_seqs - snap.num_running)
        remaining = snap.running_remaining_tokens
        if need > 0 and remaining:
            rounds, idx = divmod(need - 1, len(remaining))  # more waiters than running: queue again
            slot_steps = remaining[idx] + rounds * remaining[-1]
        return (max(prefill_steps, slot_steps) + 1) * snap.step_seconds

    def check(
        self,
        snap: EngineSnapshot,
        prompt_tokens: int,
        deadline_ms: int | None,
        *,
        draining: bool,
    ) -> Rejection | None:
        if draining:
            return Rejection(503, "draining", "server is draining for shutdown", None)
        backlog = self.predicted_ttft_s(snap, 0)
        retry = max(1, math.ceil(backlog)) if backlog is not None else 1
        if self.inflight >= self.config.max_inflight_requests:
            return Rejection(429, "server_busy", "too many concurrent requests", retry)
        if snap.num_waiting >= self.config.max_queued_requests:
            return Rejection(429, "queue_full", "request queue is full", retry)
        if deadline_ms is not None:
            predicted = self.predicted_ttft_s(snap, prompt_tokens)
            if predicted is not None and predicted * 1000 > deadline_ms:
                wait = max(1, math.ceil(predicted - deadline_ms / 1000))
                return Rejection(
                    429,
                    "deadline_unattainable",
                    f"predicted time to first token {predicted * 1000:.0f} ms exceeds the "
                    f"{deadline_ms} ms deadline",
                    wait,
                )
        return None
