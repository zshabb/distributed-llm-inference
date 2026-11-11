"""Load generator for any OpenAI-compatible endpoint (native server, vLLM, HF baseline).

Three modes:

* **open loop** (latency): Poisson arrivals at a fixed rate, scheduled in advance from a seed. A
  request is sent at its scheduled time whether or not earlier ones finished, and every latency is
  measured **from the scheduled arrival**, not from when the client got around to sending it. That
  avoids coordinated omission: a slow server cannot hide its queueing delay by slowing the client
  down. The client's own lateness (``sent - scheduled``) is recorded, so client saturation shows.
* **closed loop** (max throughput): ``concurrency`` workers, each sending its next request once
  the previous one completes.
* **sessions** (multi-turn): sessions arrive as a Poisson process; inside a session each turn is
  sent after the previous reply plus an exponential think time, with the generated reply appended to
  the conversation.

Time is ``time.perf_counter()`` relative to the run start. Streaming responses are timed per SSE
chunk; token counts come from the server's final usage record.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
import numpy as np

from dli.bench.workloads import GenRequest, Session


@dataclass
class Target:
    """Where and how to send requests."""

    base_url: str  # e.g. http://127.0.0.1:8000
    model: str
    stream: bool = True
    api_key: str | None = None
    extra_body: dict[str, Any] = field(default_factory=lambda: {"ignore_eos": True})
    headers: dict[str, str] = field(default_factory=dict)
    request_timeout_s: float = 600.0


@dataclass
class RequestRecord:
    rid: str
    scheduled: float
    sent: float = 0.0
    first_token: float | None = None
    end: float | None = None
    chunk_times: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    status: str = "pending"  # ok | rejected | error | timeout
    http_status: int | None = None
    error: str | None = None
    session: str | None = None
    turn: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    mode: str
    records: list[RequestRecord]
    wall_s: float
    params: dict[str, Any]


def _body(
    target: Target, req: GenRequest | None, messages: list[dict[str, str]] | None, max_tokens: int
) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": target.model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        **target.extra_body,
    }
    if target.stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    if messages is not None:
        return "/v1/chat/completions", {**body, "messages": messages}
    assert req is not None
    if req.messages is not None:
        msgs = [{"role": r, "content": c} for r, c in req.messages]
        return "/v1/chat/completions", {**body, "messages": msgs}
    assert req.prompt is not None
    return "/v1/completions", {**body, "prompt": list(req.prompt)}


def _content(choice: dict[str, Any]) -> str:
    if "delta" in choice:
        return str(choice["delta"].get("content") or "")
    if "message" in choice:
        return str(choice["message"].get("content") or "")
    return str(choice.get("text") or "")


async def send_one(
    client: httpx.AsyncClient,
    target: Target,
    rec: RequestRecord,
    t0: float,
    *,
    req: GenRequest | None = None,
    messages: list[dict[str, str]] | None = None,
    max_tokens: int = 16,
    headers: dict[str, str] | None = None,
) -> str:
    """Send one request, filling ``rec`` in place. Returns the generated text ('' on failure)."""
    path, body = _body(target, req, messages, max_tokens if req is None else req.max_tokens)
    hdrs = {**target.headers, **(headers or {})}
    if target.api_key:
        hdrs["Authorization"] = f"Bearer {target.api_key}"
    if req is not None and req.priority:
        hdrs["X-Request-Priority"] = req.priority
    parts: list[str] = []
    rec.sent = time.perf_counter() - t0
    try:
        if target.stream:
            async with client.stream("POST", path, json=body, headers=hdrs) as r:
                rec.http_status = r.status_code
                if r.status_code != 200:
                    await r.aread()
                    rec.status = "rejected" if r.status_code in (429, 503) else "error"
                    rec.error = r.text[:300]
                    rec.end = time.perf_counter() - t0
                    return ""
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    now = time.perf_counter() - t0
                    obj = json.loads(data)
                    if "error" in obj:
                        rec.status, rec.error = "error", str(obj["error"])[:300]
                        continue
                    for ch in obj.get("choices") or []:
                        text = _content(ch)
                        if text:
                            parts.append(text)
                            rec.chunk_times.append(now)
                            if rec.first_token is None:
                                rec.first_token = now
                    usage = obj.get("usage")
                    if usage:
                        rec.prompt_tokens = usage.get("prompt_tokens")
                        rec.output_tokens = usage.get("completion_tokens")
        else:
            r = await client.post(path, json=body, headers=hdrs)
            rec.http_status = r.status_code
            now = time.perf_counter() - t0
            if r.status_code != 200:
                rec.status = "rejected" if r.status_code in (429, 503) else "error"
                rec.error, rec.end = r.text[:300], now
                return ""
            obj = r.json()
            parts.append(_content(obj["choices"][0]))
            rec.first_token = rec.end = now  # no streaming: the first token arrives with the last
            rec.chunk_times.append(now)
            usage = obj.get("usage") or {}
            rec.prompt_tokens = usage.get("prompt_tokens")
            rec.output_tokens = usage.get("completion_tokens")
        if rec.end is None:
            rec.end = time.perf_counter() - t0
        if rec.status == "pending":
            rec.status = "ok"
    except httpx.TimeoutException as exc:
        rec.status, rec.error, rec.end = "timeout", type(exc).__name__, time.perf_counter() - t0
    except httpx.HTTPError as exc:
        rec.status, rec.error, rec.end = (
            "error",
            f"{type(exc).__name__}: {exc}"[:300],
            time.perf_counter() - t0,
        )
    return "".join(parts)


def _client(target: Target) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=target.base_url,
        timeout=httpx.Timeout(target.request_timeout_s, connect=30.0),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    )


def poisson_arrivals(n: int, rate: float, seed: int) -> list[float]:
    """Arrival offsets (seconds) of a Poisson process: cumulative exponential gaps, first at 0."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    gaps = np.random.default_rng(seed).exponential(1.0 / rate, size=n)
    gaps[0] = 0.0
    return [float(x) for x in np.cumsum(gaps)]


async def _finish(
    tasks: list[asyncio.Task[Any]], records: list[RequestRecord], drain_s: float, t0: float
) -> None:
    if tasks:
        _, pending = await asyncio.wait(tasks, timeout=drain_s)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    now = time.perf_counter() - t0
    for r in records:
        if r.status == "pending":
            r.status, r.error, r.end = "timeout", "unfinished at drain deadline", now


async def open_loop(
    target: Target,
    requests: Sequence[GenRequest],
    rate: float,
    *,
    seed: int = 0,
    drain_s: float = 120.0,
) -> RunResult:
    arrivals = poisson_arrivals(len(requests), rate, seed)
    records = [RequestRecord(r.rid, a) for r, a in zip(requests, arrivals, strict=True)]
    async with _client(target) as client:
        t0 = time.perf_counter()
        tasks: list[asyncio.Task[Any]] = []
        for req, rec in zip(requests, records, strict=True):
            delay = rec.scheduled - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send_one(client, target, rec, t0, req=req)))
        await _finish(tasks, records, drain_s, t0)
        wall = time.perf_counter() - t0
    return RunResult("open", records, wall, {"rate": rate, "seed": seed, "n": len(requests)})


async def closed_loop(
    target: Target, requests: Sequence[GenRequest], concurrency: int, *, drain_s: float = 600.0
) -> RunResult:
    records = [RequestRecord(r.rid, 0.0) for r in requests]
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(len(requests)):
        queue.put_nowait(i)
    async with _client(target) as client:
        t0 = time.perf_counter()

        async def worker() -> None:
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                records[i].scheduled = (
                    time.perf_counter() - t0
                )  # closed loop: sent when a slot frees
                await send_one(client, target, records[i], t0, req=requests[i])

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await _finish(tasks, records, drain_s, t0)
        wall = time.perf_counter() - t0
    return RunResult("closed", records, wall, {"concurrency": concurrency, "n": len(requests)})


async def sessions_open_loop(
    target: Target,
    sessions: Sequence[Session],
    session_rate: float,
    *,
    think_s: float = 1.0,
    seed: int = 0,
    drain_s: float = 180.0,
) -> RunResult:
    arrivals = poisson_arrivals(len(sessions), session_rate, seed)
    records: list[RequestRecord] = []
    rng = np.random.default_rng(seed + 1)
    thinks = [list(rng.exponential(think_s, size=len(s.user_turns))) for s in sessions]
    async with _client(target) as client:
        t0 = time.perf_counter()

        async def run_session(k: int, s: Session) -> None:
            msgs: list[dict[str, str]] = [{"role": "system", "content": s.system}]
            due = arrivals[k]
            for turn, user in enumerate(s.user_turns):
                delay = due - (time.perf_counter() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
                msgs.append({"role": "user", "content": user})
                rec = RequestRecord(f"{s.sid}-t{turn}", due, session=s.sid, turn=turn)
                records.append(rec)
                reply = await send_one(
                    client, target, rec, t0, messages=list(msgs), max_tokens=s.max_tokens
                )
                if rec.status != "ok":
                    return  # a failed turn ends the session, as it would for a real user
                msgs.append({"role": "assistant", "content": reply})
                due = (rec.end or 0.0) + thinks[k][turn]

        tasks = [asyncio.create_task(run_session(k, s)) for k, s in enumerate(sessions)]
        await _finish(tasks, records, drain_s + arrivals[-1], t0)
        wall = time.perf_counter() - t0
    return RunResult(
        "sessions",
        records,
        wall,
        {"session_rate": session_rate, "think_s": think_s, "seed": seed, "sessions": len(sessions)},
    )


# ----------------------------------------------------------------------------------- metrics


def ttft_by_arrival(ok: Sequence[RequestRecord]) -> list[tuple[float, float]]:
    """(scheduled arrival, TTFT) of completed requests."""
    return [(r.scheduled, r.first_token - r.scheduled) for r in ok if r.first_token is not None]


def run_metrics(result: RunResult) -> dict[str, Any]:
    """Per-run metrics. Latencies run from the scheduled arrival (open) or the send (closed)."""
    from dli.bench.stats import summarize

    recs = result.records
    ok = [r for r in recs if r.status == "ok" and r.end is not None and r.first_token is not None]
    counts = {s: sum(r.status == s for r in recs) for s in ("ok", "rejected", "error", "timeout")}
    out_tokens = sum(r.output_tokens or 0 for r in ok)
    start = min((r.scheduled for r in recs), default=0.0)
    last = max((r.end for r in ok if r.end is not None), default=start)
    span = max(last - start, 1e-9)
    ttft = [r.first_token - r.scheduled for r in ok if r.first_token is not None]
    e2e = [r.end - r.scheduled for r in ok if r.end is not None]
    tpot = [
        (r.end - r.first_token) / (r.output_tokens - 1)
        for r in ok
        if r.end is not None
        and r.first_token is not None
        and r.output_tokens is not None
        and r.output_tokens > 1
    ]
    itl = [b - a for r in ok for a, b in zip(r.chunk_times, r.chunk_times[1:], strict=False)]
    rejected_latency = [
        r.end - r.sent for r in recs if r.status == "rejected" and r.end is not None
    ]
    lag = [r.sent - r.scheduled for r in recs if r.sent]
    # Queue growth: median TTFT of the last third of arrivals vs the first third. Stable load keeps
    # it flat; past saturation the queue, and so TTFT, grows through the run.
    by_arrival = sorted(ttft_by_arrival(ok), key=lambda x: x[0])
    third = len(by_arrival) // 3
    ttft_first = float(np.median([t for _, t in by_arrival[:third]])) if third >= 3 else None
    ttft_last = float(np.median([t for _, t in by_arrival[-third:]])) if third >= 3 else None

    def s(v: list[float]) -> dict[str, float] | None:
        x = summarize(v)
        return x.as_dict() if x else None

    return {
        "mode": result.mode,
        **result.params,
        "requests": len(recs),
        **counts,
        "completion_rate": counts["ok"] / max(len(recs), 1),
        "span_s": span,
        "output_tokens": out_tokens,
        "output_tok_per_s": out_tokens / span,
        "completed_req_per_s": len(ok) / span,
        "ttft_s": s(ttft),
        "e2e_s": s(e2e),
        "tpot_s": s(tpot),
        "itl_s": s(itl),
        "rejection_latency_s": s(rejected_latency),
        "client_lag_s": s(lag),
        "ttft_first_third_p50_s": ttft_first,
        "ttft_last_third_p50_s": ttft_last,
    }
