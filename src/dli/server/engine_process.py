"""The engine in its own process: no GIL shared with the HTTP event loop.

With the engine on a thread of the API process (:class:`~dli.server.async_engine.AsyncEngine`), the
engine thread and the asyncio loop take turns on one GIL. At concurrency 64 on the A5000 that made
every step wait for per-token HTTP work: mixed steps took 119 ms instead of 52 ms and decode steps
11.5 ms instead of 9.4 ms (`benchmarks/results/2026-11-11-GPU-A5000x1-runpod-m6-http-profile/`).

:class:`ProcessEngine` keeps the :class:`AsyncEngine` interface (``generate``, ``abort``,
``snapshot``, ``healthy``, ``start``, ``stop``) but runs the engine in a spawned child process::

    API process                                      engine process (spawn)
    asyncio handlers --(add/abort, pipe)-----------> command pipe -> drain -> step
    loop <-(one call_soon_threadsafe per step)- reader thread <-(one message per step)- outputs

* The child loads the weights, builds the engine, captures the decode CUDA graphs and warms up
  before it reports ready; the API process never touches the GPU.
* One message per step carries every request's new tokens and the :class:`EngineSnapshot`, so the
  API process sees the same state (admission, metrics, admin) as with the threaded engine.
* Tokenization, detokenization, stop strings, SSE and metrics stay in the API process.
* Failure isolation: an exception in the child fails every in-flight request with an error event and
  marks the engine unhealthy, as does the child dying (the pipe reports EOF). If the API process
  dies, the child sees EOF on its command pipe and exits. The child ignores SIGINT so that Ctrl-C
  in a terminal drains through the API process instead of killing the engine first.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import signal
import threading
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import torch

from dli.engine import FinishReason, LLMEngine, Priority, SchedulerConfig
from dli.engine.cuda_graph import Shape
from dli.sampling import SamplingParams
from dli.server.async_engine import (
    AsyncEngine,
    EngineDeadError,
    EngineEvent,
    EngineSnapshot,
    _Pending,
    _put_all,
)

log = logging.getLogger(__name__)

# (request_id, new token ids, finished, finish reason)
_Out = tuple[str, tuple[int, ...], bool, FinishReason | None]


@dataclass(frozen=True)
class EngineSpec:
    """Everything the child needs to build the engine (picklable)."""

    model_dir: str
    num_blocks: int
    block_size: int
    device: str
    dtype: torch.dtype
    scheduler: SchedulerConfig
    eos_token_ids: tuple[int, ...]
    enable_prefix_caching: bool = True
    cuda_graphs: tuple[Shape, ...] | None = None
    decode_attention: str = "auto"


def _build(spec: EngineSpec) -> tuple[LLMEngine, dict[str, Any]]:
    from dli.model import load_model

    model = load_model(Path(spec.model_dir), dtype=spec.dtype, device=spec.device)
    eng = LLMEngine.for_model(
        model,
        num_blocks=spec.num_blocks,
        block_size=spec.block_size,
        config=spec.scheduler,
        eos_token_ids=spec.eos_token_ids,
        enable_prefix_caching=spec.enable_prefix_caching,
        cuda_graphs=spec.cuda_graphs,
        decode_attention=spec.decode_attention,
    )
    with torch.inference_mode():  # warm kernels, surface load failures before reporting ready
        ids = torch.tensor([1, 2, 3], device=model.inv_freq.device)
        model(ids, torch.arange(3, device=ids.device), last_token_only=True)
        graphs = getattr(eng.runner, "decode_graphs", None)
        if graphs is not None:
            graphs.capture()
    info: dict[str, Any] = {
        "cuda_graphs": graphs.summary() if graphs is not None else None,
        "decode_attention": getattr(eng.runner, "decode_attention", None),
    }
    return eng, info


def _child_main(spec: EngineSpec, commands: Connection, outputs: Connection) -> None:
    """Engine process: build, report ready, then drain commands -> step -> send, until stopped."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        eng, info = _build(spec)
    except Exception as exc:
        outputs.send(("dead", f"engine start-up failed: {type(exc).__name__}: {exc}"))
        return
    core = AsyncEngine(eng)  # reused for its accounting and snapshot; its thread is never started
    eng.bm.bind_to_current_thread()
    outputs.send(("ready", info, core.snapshot))
    try:
        while True:
            out: list[_Out] = []
            changed = False  # a command changed engine state: publish even without a step
            busy = eng.has_unfinished()
            # Idle: block briefly for work. Busy: take whatever arrived, without waiting.
            if commands.poll(0 if busy else 0.02):
                while True:
                    try:
                        cmd = commands.recv()
                    except EOFError:  # the API process is gone
                        return
                    if cmd[0] == "stop":
                        return
                    changed = True
                    if cmd[0] == "add":
                        _, rid, prompt, params, priority, tenant = cmd
                        req = eng.add_request(rid, prompt, params, priority=priority, tenant=tenant)
                        if req.is_finished:  # rejected at admission (CAPACITY)
                            out.append((rid, (), True, req.finish_reason))
                    else:
                        eng.abort(cmd[1])
                    if not commands.poll(0):
                        break
            if eng.has_unfinished():
                for o in eng.step():
                    out.append((o.request_id, tuple(o.new_token_ids), o.finished, o.finish_reason))
                core._account(eng)
                core._c.generated += sum(len(o[1]) for o in out)
            elif not out and not changed:
                continue
            core._publish(eng)
            outputs.send(("step", out, core.snapshot))
    except Exception as exc:
        with contextlib.suppress(OSError, ValueError):
            outputs.send(("dead", f"{type(exc).__name__}: {exc}"))


class ProcessEngine:
    """:class:`AsyncEngine`'s interface, backed by an engine running in a child process."""

    def __init__(
        self, spec: EngineSpec, *, kv_total_blocks: int, start_timeout_s: float = 900.0
    ) -> None:
        self.spec = spec
        self.start_timeout_s = start_timeout_s
        self.decode_graphs = None  # they live in the child; see ``startup_info``
        self.startup_info: dict[str, Any] = {}
        self._snapshot = EngineSnapshot(kv_total_blocks=kv_total_blocks)
        self._pending: dict[str, _Pending] = {}
        self._dead: str | None = None
        self._proc: mp.process.BaseProcess | None = None
        self._cmd: Connection | None = None
        self._out: Connection | None = None
        self._send_lock = threading.Lock()
        self._reader: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._proc is not None:
            return
        ctx = mp.get_context("spawn")  # CUDA cannot be used in a forked child
        cmd_r, cmd_w = ctx.Pipe(duplex=False)
        out_r, out_w = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_child_main, args=(self.spec, cmd_r, out_w), name="dli-engine", daemon=True
        )
        proc.start()
        cmd_r.close()
        out_w.close()
        self._proc, self._cmd, self._out = proc, cmd_w, out_r
        deadline = time.monotonic() + self.start_timeout_s
        while not out_r.poll(1.0):
            if not proc.is_alive():
                raise RuntimeError(f"engine process exited during start-up ({proc.exitcode})")
            if time.monotonic() > deadline:
                proc.kill()
                raise RuntimeError("engine process did not become ready in time")
        msg = out_r.recv()
        if msg[0] != "ready":
            proc.join(10)
            raise RuntimeError(msg[1])
        self.startup_info, self._snapshot = msg[1], msg[2]
        self._reader = threading.Thread(target=self._read, name="dli-engine-reader", daemon=True)
        self._reader.start()

    def stop(self, timeout: float = 10.0) -> None:
        if self._proc is None:
            return
        with contextlib.suppress(OSError, ValueError):
            self._send(("stop",))
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(5)
        if self._reader is not None:
            self._reader.join(timeout)
        if self._cmd is not None:
            self._cmd.close()

    @property
    def snapshot(self) -> EngineSnapshot:
        return self._snapshot

    @property
    def healthy(self) -> bool:
        return (
            self._dead is None
            and self._proc is not None
            and self._proc.is_alive()
            and self._reader is not None
            and self._reader.is_alive()
        )

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

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
        """Stream events for one request. Closing the iterator early aborts the request."""
        if not self.healthy:
            raise EngineDeadError(self._dead or "engine not running")
        pending = _Pending(asyncio.get_running_loop(), asyncio.Queue())
        self._pending[request_id] = pending
        try:
            self._send(("add", request_id, list(prompt_ids), params, priority, tenant))
        except (OSError, ValueError) as exc:
            self._pending.pop(request_id, None)
            raise EngineDeadError(f"engine process unreachable: {exc}") from exc
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
        with contextlib.suppress(OSError, ValueError):
            self._send(("abort", request_id))

    # ------------------------------------------------------------------ internals

    def _send(self, msg: tuple[Any, ...]) -> None:
        if self._cmd is None:
            raise ValueError("engine process not started")
        with self._send_lock:
            self._cmd.send(msg)

    def _read(self) -> None:
        """Reader thread: one message per engine step -> one loop wake-up per step."""
        assert self._out is not None
        try:
            while True:
                msg = self._out.recv()
                if msg[0] == "step":
                    self._snapshot = msg[2]
                    self._deliver(msg[1])
                elif msg[0] == "dead":
                    self._fail(msg[1])
                    return
        except (EOFError, OSError):
            code = self._proc.exitcode if self._proc is not None else None
            self._fail(f"engine process exited (code {code})")

    def _deliver(self, outputs: Sequence[_Out]) -> None:
        batches: dict[
            asyncio.AbstractEventLoop, list[tuple[asyncio.Queue[EngineEvent], EngineEvent]]
        ]
        batches = {}
        for rid, toks, finished, reason in outputs:
            pending = self._pending.get(rid)
            if pending is None:
                continue  # client already gone (aborted)
            batches.setdefault(pending.loop, []).append(
                (pending.events, EngineEvent(toks, finished, reason))
            )
        for loop, items in batches.items():
            with contextlib.suppress(RuntimeError):  # event loop closed during shutdown
                loop.call_soon_threadsafe(_put_all, items)

    def _fail(self, error: str) -> None:
        if self._dead is None:
            self._dead = error
        self._snapshot = EngineSnapshot(
            healthy=False, error=self._dead, kv_total_blocks=self._snapshot.kv_total_blocks
        )
        failed = EngineEvent(error=f"engine failure: {self._dead}", finished=True)
        self._deliver_errors(list(self._pending), failed)

    def _deliver_errors(self, request_ids: list[str], event: EngineEvent) -> None:
        for rid in request_ids:
            pending = self._pending.get(rid)
            if pending is not None:
                with contextlib.suppress(RuntimeError):
                    pending.loop.call_soon_threadsafe(pending.events.put_nowait, event)
