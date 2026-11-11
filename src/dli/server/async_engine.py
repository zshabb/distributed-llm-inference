"""Bridge between the single-threaded LLMEngine and the asyncio HTTP server.

The engine (scheduler, block manager, KV tensors) is thread-confined by design. Instead of locking
it, :class:`AsyncEngine` runs it on one dedicated thread and talks to it only through a thread-safe
command queue::

    asyncio handlers --(add / abort)--> queue.Queue --> engine thread: drain -> step -> dispatch
    asyncio handlers <--(per-request asyncio.Queue, via loop.call_soon_threadsafe)-- engine thread

* **Cancellation:** a request's async iterator sends ``abort`` from its ``finally`` block when it is
  closed before finishing. That covers a client disconnect (Starlette closes the streaming
  generator), a stop string found by the API layer, a server-side timeout, and handler cancellation.
  The engine applies aborts between steps and frees the KV blocks immediately.
* **Failure isolation:** an exception in the engine loop fails every in-flight request with an error
  event and marks the engine unhealthy. Clients get a clear error instead of a hung connection, and
  readiness flips to false.
* **Observability without locks:** after each step the engine thread publishes an immutable
  :class:`EngineSnapshot`. Reading the reference is atomic, so handlers and metrics see a
  consistent view without blocking the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from dli.engine import FinishReason, LLMEngine, Priority, RequestOutput
from dli.engine.cuda_graph import DecodeGraphs
from dli.sampling import SamplingParams


@dataclass(frozen=True, slots=True)
class EngineEvent:
    """One update for one request: new tokens and/or completion."""

    token_ids: tuple[int, ...] = ()
    finished: bool = False
    finish_reason: FinishReason | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """Read-only engine state published after every step (for admission control and metrics)."""

    num_waiting: int = 0
    num_running: int = 0
    waiting_prefill_tokens: int = (
        0  # prompt tokens not yet computed, across waiting + running requests
    )
    kv_free_blocks: int = 0
    kv_cached_idle_blocks: int = 0
    kv_used_blocks: int = 0
    kv_total_blocks: int = 0
    prefill_tokens_per_s: float = 0.0  # EWMA over steps that did prefill work (metrics only)
    step_seconds: float = 0.0  # EWMA of total step time
    max_num_batched_tokens: int = 0  # scheduler token budget per step
    max_num_seqs: int = 0  # scheduler cap on running sequences
    # Tokens each running request may still generate (max_tokens - generated), ascending: the
    # earliest a sequence slot can free up. An upper bound when a request stops early at EOS.
    running_remaining_tokens: tuple[int, ...] = ()
    total_steps: int = 0
    total_preemptions: int = 0
    total_generated_tokens: int = 0
    prefix_hit_tokens: int = 0
    prefix_query_tokens: int = 0
    # Decode CUDA graphs: "off" when not configured, else the capture status of the bucket set.
    cuda_graph_status: str = "off"
    cuda_graph_replays: int = 0  # decode steps served by a graph replay
    cuda_graph_fallbacks: tuple[tuple[str, int], ...] = ()  # eager decode steps, by reason
    cuda_graph_memory_bytes: int = 0  # reserved by capture (static buffers + shared pool)
    healthy: bool = True
    error: str | None = None


def _put_all(items: list[tuple[asyncio.Queue[EngineEvent], EngineEvent]]) -> None:
    for events, event in items:
        events.put_nowait(event)


@dataclass
class _Pending:
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue[EngineEvent]


@dataclass
class _Counters:
    preemptions: int = 0
    generated: int = 0
    steps: int = 0
    prefill_rate: float = 0.0
    step_s: float = 0.0


@dataclass(frozen=True, slots=True)
class _Add:
    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    priority: Priority
    tenant: str


@dataclass(frozen=True, slots=True)
class _Abort:
    request_id: str


@dataclass(frozen=True, slots=True)
class _Stop:
    pass


_Command = _Add | _Abort | _Stop


class EngineDeadError(RuntimeError):
    """The engine loop has stopped (crashed or shut down) and cannot accept work."""


class AsyncEngine:
    """Runs an :class:`LLMEngine` on a background thread and exposes async streaming generation."""

    _EWMA = 0.2  # weight of the newest step in moving averages

    def __init__(self, engine: LLMEngine, *, idle_poll_s: float = 0.02) -> None:
        self._engine = engine
        # Decode CUDA graphs of the wrapped engine, if configured (captured before start()).
        self.decode_graphs: DecodeGraphs | None = getattr(engine.runner, "decode_graphs", None)
        self._cmds: queue.Queue[_Command] = queue.Queue()
        self._pending: dict[str, _Pending] = {}
        self._idle_poll_s = idle_poll_s
        self._thread: threading.Thread | None = None
        self._snapshot = EngineSnapshot(kv_total_blocks=engine.bm.num_blocks)
        self._c = _Counters()
        self._dead: str | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        # Thread confinement binds to the creating thread; ownership is handed to the loop thread.
        self._thread = threading.Thread(target=self._run, name="dli-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        if self._thread is None:
            return
        self._cmds.put(_Stop())
        self._thread.join(timeout)

    @property
    def snapshot(self) -> EngineSnapshot:
        return self._snapshot

    @property
    def healthy(self) -> bool:
        return self._dead is None and self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ API (called from asyncio)

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        params: SamplingParams,
        *,
        priority: Priority = Priority.INTERACTIVE,
        tenant: str = "default",
    ) -> AsyncGenerator[EngineEvent, None]:
        """Stream events for one request. Closing the iterator early aborts the request.

        Raises:
            EngineDeadError: The engine loop is not running.
        """
        if not self.healthy:
            raise EngineDeadError(self._dead or "engine not running")
        loop = asyncio.get_running_loop()
        pending = _Pending(loop, asyncio.Queue())
        self._pending[request_id] = pending
        self._cmds.put(_Add(request_id, prompt_ids, params, priority, tenant))
        finished = False
        try:
            while True:
                event = await pending.events.get()
                if event.finished or event.error is not None:
                    finished = True
                yield event
                if finished:
                    return
        finally:
            if not finished:
                self.abort(request_id)
            self._pending.pop(request_id, None)

    def abort(self, request_id: str) -> None:
        """Request cancellation (thread-safe, idempotent). Applied by the engine between steps."""
        self._cmds.put(_Abort(request_id))

    # ------------------------------------------------------------------ engine thread

    def _run(self) -> None:
        eng = self._engine
        # The engine was constructed on another thread and is used ONLY here from now on.
        eng.bm.bind_to_current_thread()
        try:
            while True:
                if not self._drain_commands(block=not eng.has_unfinished()):
                    return
                if eng.has_unfinished():
                    outputs = eng.step()
                    self._account(eng)
                    self._deliver(outputs)
                self._publish(eng)
        except Exception as exc:  # crash: fail everyone loudly, never hang clients
            self._dead = f"{type(exc).__name__}: {exc}"
            for rid in list(self._pending):
                self._send(rid, EngineEvent(error=f"engine failure: {self._dead}", finished=True))
            self._snapshot = EngineSnapshot(
                healthy=False, error=self._dead, kv_total_blocks=eng.bm.num_blocks
            )

    def _drain_commands(self, *, block: bool) -> bool:
        """Apply queued commands. Returns False when asked to stop."""
        try:
            cmd = self._cmds.get(timeout=self._idle_poll_s) if block else self._cmds.get_nowait()
        except queue.Empty:
            return True
        while True:
            if isinstance(cmd, _Stop):
                return False
            if isinstance(cmd, _Add):
                req = self._engine.add_request(
                    cmd.request_id,
                    cmd.prompt_ids,
                    cmd.params,
                    priority=cmd.priority,
                    tenant=cmd.tenant,
                )
                if req.is_finished:  # rejected at admission (CAPACITY)
                    rejected = EngineEvent(finished=True, finish_reason=req.finish_reason)
                    self._send(cmd.request_id, rejected)
            else:
                self._engine.abort(cmd.request_id)
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                return True

    def _deliver(self, outputs: list[RequestOutput]) -> None:
        """Hand one step's outputs to the event loop(s) in ONE wake-up per loop.

        A wake-up (``call_soon_threadsafe``) writes to the loop's self-pipe, a system call that
        releases the GIL; re-acquiring it can wait for the loop thread's switch interval. Done per
        token, this cost ~74 us per token of engine-thread time at concurrency 64 on the A5000
        (2.4 s of a 12.6 s run); per step it is one call for up to 64 tokens.
        """
        batches: dict[
            asyncio.AbstractEventLoop, list[tuple[asyncio.Queue[EngineEvent], EngineEvent]]
        ]
        batches = {}
        for out in outputs:
            self._c.generated += len(out.new_token_ids)
            pending = self._pending.get(out.request_id)
            if pending is None:
                continue  # client already gone (aborted); nothing to deliver
            event = EngineEvent(tuple(out.new_token_ids), out.finished, out.finish_reason)
            batches.setdefault(pending.loop, []).append((pending.events, event))
        for loop, items in batches.items():
            with contextlib.suppress(RuntimeError):  # event loop closed during shutdown
                loop.call_soon_threadsafe(_put_all, items)

    def _send(self, request_id: str, event: EngineEvent) -> None:
        pending = self._pending.get(request_id)
        if pending is None:
            return  # client already gone (aborted); nothing to deliver
        with contextlib.suppress(RuntimeError):  # event loop closed during shutdown
            pending.loop.call_soon_threadsafe(pending.events.put_nowait, event)

    def _account(self, eng: LLMEngine) -> None:
        s = eng.stats.steps[-1]
        self._c.steps += 1
        self._c.preemptions += s.num_preempted
        a = self._EWMA
        self._c.step_s = (
            s.total_s if self._c.steps == 1 else (1 - a) * self._c.step_s + a * s.total_s
        )
        if s.num_prefill_tokens and s.forward_s > 0:
            rate = s.num_prefill_tokens / s.forward_s
            self._c.prefill_rate = (
                rate if self._c.prefill_rate == 0 else (1 - a) * self._c.prefill_rate + a * rate
            )
        # Bound memory: engine-level StepStats are already summarized here.
        if len(eng.stats.steps) > 4096:
            del eng.stats.steps[:-1024]

    def _publish(self, eng: LLMEngine) -> None:
        sched = eng.scheduler
        prefill = sum(r.num_tokens - r.num_computed_tokens for r in sched.waiting) + sum(
            max(0, len(r.prompt_ids) - r.num_computed_tokens) for r in sched.running
        )
        st = eng.bm.stats()
        graphs = self.decode_graphs
        graph_fields: dict[str, Any] = {}
        if graphs is not None:
            graph_fields = {
                "cuda_graph_status": graphs.stats.status,
                "cuda_graph_replays": sum(g.stats.replays for g in graphs.graphs),
                "cuda_graph_fallbacks": tuple(sorted(graphs.stats.fallbacks.items())),
                "cuda_graph_memory_bytes": graphs.stats.memory_overhead,
            }
        self._snapshot = EngineSnapshot(
            num_waiting=len(sched.waiting),
            num_running=len(sched.running),
            waiting_prefill_tokens=prefill,
            kv_free_blocks=st.free_blocks,
            kv_cached_idle_blocks=st.cached_idle_blocks,
            kv_used_blocks=st.used_blocks,
            kv_total_blocks=st.num_blocks,
            prefill_tokens_per_s=self._c.prefill_rate,
            step_seconds=self._c.step_s,
            max_num_batched_tokens=sched.config.max_num_batched_tokens,
            max_num_seqs=sched.config.max_num_seqs,
            running_remaining_tokens=tuple(
                sorted(max(0, r.params.max_tokens - len(r.output_ids)) for r in sched.running)
            ),
            total_steps=self._c.steps,
            total_preemptions=self._c.preemptions,
            total_generated_tokens=self._c.generated,
            prefix_hit_tokens=st.prompt_tokens_hit,
            prefix_query_tokens=st.prompt_tokens_queried,
            healthy=True,
            **graph_fields,
        )
