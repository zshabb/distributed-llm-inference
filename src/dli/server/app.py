"""OpenAI-compatible HTTP API (contract C1): /v1/models, /v1/completions, /v1/chat/completions.

Request flow::

    validate (pydantic + unsupported-feature check) -> tokenize / render chat template (sandboxed)
    -> length checks -> AsyncEngine.generate() -> IncrementalDetokenizer -> StopStringMatcher
    -> JSON response, or SSE chunks ("data: {...}" ... "data: [DONE]")

Termination is decided in one place (:func:`_text_stream`): EOS / stop token / max_tokens from the
engine, or a stop string found in the decoded text. In the stop-string case the stream is closed,
which aborts the engine request immediately instead of letting it generate tokens nobody receives.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from dli.engine import FinishReason, Priority
from dli.sampling import SamplingParams
from dli.server.admission import AdmissionConfig, AdmissionController
from dli.server.async_engine import AsyncEngine, EngineDeadError
from dli.server.auth import ApiKeyAuth, AuthMiddleware
from dli.server.limits import BodySizeLimit
from dli.server.metrics import ServerMetrics
from dli.server.schemas import ChatCompletionRequest, CompletionRequest
from dli.server.text import (
    ChatMessage,
    ChatTemplateError,
    IncrementalDetokenizer,
    StopStringMatcher,
    TextCodec,
)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Serving limits and identity.

    Attributes:
        model_name: Id clients must send in ``model`` (and that /v1/models lists).
        max_model_len: Max prompt + generated tokens (the model's position limit).
        kv_capacity_tokens: Token slots in the KV pool. A request needing more can never run.
        default_max_tokens: Used when a request omits ``max_tokens`` (safe, small default).
        max_tokens_limit: Upper bound a client may request.
        request_timeout_s: Hard wall-clock limit per request; the engine request is then aborted.
    """

    model_name: str
    max_model_len: int
    kv_capacity_tokens: int
    vocab_size: int
    default_max_tokens: int = 16
    max_tokens_limit: int = 4096
    request_timeout_s: float = 600.0
    max_body_bytes: int = 2 * 1024 * 1024


class APIError(Exception):
    """An error rendered in the OpenAI error envelope."""

    def __init__(
        self,
        status: int,
        message: str,
        err_type: str,
        code: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status, self.message, self.err_type, self.code = status, message, err_type, code
        self.retry_after = retry_after


def error_response(
    status: int,
    message: str,
    err_type: str,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = {"error": {"message": message, "type": err_type, "code": code}}
    return JSONResponse(body, status_code=status, headers=headers)


@dataclass
class _Delta:
    text: str


@dataclass
class _Finish:
    reason: str
    completion_tokens: int


@dataclass
class _Timing:
    start: float
    first_token: float | None = None
    end: float | None = None

    def as_dict(self) -> dict[str, float]:
        end = self.end or time.perf_counter()
        ttft = (self.first_token or end) - self.start
        return {
            "ttft_ms": round(ttft * 1000, 2),
            "decode_ms": round((end - (self.first_token or end)) * 1000, 2),
            "total_ms": round((end - self.start) * 1000, 2),
        }


_FINISH = {
    FinishReason.STOP: "stop",
    FinishReason.LENGTH: "length",
    FinishReason.ABORTED: "cancelled",
}


def create_app(
    engine: AsyncEngine,
    codec: TextCodec,
    config: ServerConfig,
    admission: AdmissionConfig | None = None,
    auth: ApiKeyAuth | None = None,
    metrics: ServerMetrics | None = None,
) -> FastAPI:
    """Build the API app. ``auth=None`` disables authentication (local development only:
    the server entry point refuses to start without keys on a non-loopback address)."""
    app = FastAPI(title="distributed-llm-inference", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.engine = engine
    app.state.codec = codec
    app.state.config = config
    app.state.draining = False
    gate = AdmissionController(admission or AdmissionConfig())
    app.state.admission = gate
    m = metrics or ServerMetrics(engine)
    app.state.metrics = m
    app.add_middleware(BodySizeLimit, max_bytes=config.max_body_bytes)

    # Pure-ASGI layers wrap the app; the LAST added runs FIRST: auth precedes the body-size check.
    app.add_middleware(AuthMiddleware, auth=auth)

    def _tenant(request: Request) -> str:
        # Set by AuthMiddleware from the verified key; client X-Tenant-Id headers are ignored.
        return str(getattr(request.state, "tenant", "default"))

    # ------------------------------------------------------------------ errors

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError) -> JSONResponse:
        _count(request, exc.status, exc.err_type)
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return error_response(exc.status, exc.message, exc.err_type, exc.code, headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        _count(request, 400, "invalid_request_error")
        first = exc.errors()[0] if exc.errors() else {"msg": "invalid request", "loc": ()}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        msg = f"{where}: {first['msg']}" if where else str(first["msg"])
        return error_response(400, msg, "invalid_request_error", "invalid_parameter")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        kind = "not_found_error" if exc.status_code == 404 else "invalid_request_error"
        return error_response(exc.status_code, str(exc.detail), kind)

    def _count(request: Request, status: int, err_type: str | None = None) -> None:
        path = request.url.path
        endpoint = path if path in ("/v1/completions", "/v1/chat/completions") else "other"
        m.requests.labels(endpoint, str(status)).inc()
        if err_type is not None:
            m.errors.labels(err_type).inc()

    # ------------------------------------------------------------------ ops

    @app.get("/healthz")
    async def healthz() -> Response:
        # Liveness: the process serves HTTP. An unhealthy engine is reported by readiness, so an
        # orchestrator stops routing traffic first and only restarts if it never recovers.
        return Response("ok", media_type="text/plain")

    @app.get("/readyz")
    async def readyz() -> Response:
        if app.state.draining or not engine.healthy:
            return Response("not ready", status_code=503, media_type="text/plain")
        return Response("ready", media_type="text/plain")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": config.model_name, "object": "model", "owned_by": "dli"}],
        }

    # ------------------------------------------------------------------ generation

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request) -> Response:
        prompt_ids = body.prompt if isinstance(body.prompt, list) else codec.encode(body.prompt)
        return await _serve(request, body, prompt_ids, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
        try:
            prompt_ids = codec.render_chat([ChatMessage(m.role, m.content) for m in body.messages])
        except ChatTemplateError as exc:
            raise APIError(400, str(exc), "invalid_request_error", "invalid_messages") from exc
        return await _serve(request, body, prompt_ids, chat=True)

    async def _serve(
        request: Request,
        body: CompletionRequest | ChatCompletionRequest,
        prompt_ids: list[int],
        *,
        chat: bool,
    ) -> Response:
        tenant = _tenant(request)
        _validate(body, prompt_ids)
        params = SamplingParams(
            max_tokens=body.max_tokens or config.default_max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            seed=body.seed,
            ignore_eos=body.ignore_eos,
        )
        priority = _priority(request.headers.get("x-request-priority"))
        deadline_ms = _deadline(request.headers.get("x-deadline-ms"))
        rejection = gate.check(
            engine.snapshot, len(prompt_ids), deadline_ms, draining=app.state.draining
        )
        if rejection is not None:
            m.rejections.labels(rejection.code).inc()
            kind = "service_unavailable" if rejection.status == 503 else "rate_limit_error"
            raise APIError(
                rejection.status, rejection.message, kind, rejection.code, rejection.retry_after_s
            )
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
        timing = _Timing(time.perf_counter())
        gate.inflight += 1
        m.prompt_tokens.inc(len(prompt_ids))
        handed_off = False
        try:
            raw = _text_stream(rid, prompt_ids, params, body.stop_list, priority, tenant, timing)
            # Wait for the first event BEFORE sending headers: errors and deadline misses up to the
            # first token are real HTTP statuses (429/4xx/5xx), not in-band events in a 200 stream.
            # The client may vanish while the request is queued or (non-streaming) generating;
            # nothing would notice until the response is written, so watch the connection.
            try:
                first = await _unless_disconnected(request, _first_event(raw, deadline_ms))
            except _ClientGoneError:
                return _gone(request)
            stream = _chain(first, raw)
            if body.stream:
                handed_off = True  # the SSE generator now owns the in-flight slot
                return StreamingResponse(
                    _release_after(_sse(rid, body, chat, len(prompt_ids), stream, timing), request),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            try:
                parts, finish = await _unless_disconnected(request, _collect(stream))
            except _ClientGoneError:
                return _gone(request)
        finally:
            if not handed_off:
                gate.inflight -= 1
        text = "".join(parts)
        _count(request, 200)
        usage = _usage(len(prompt_ids), finish.completion_tokens)
        choice: dict[str, Any] = {"index": 0, "finish_reason": finish.reason, "logprobs": None}
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        return JSONResponse(
            {
                "id": rid,
                "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()),
                "model": config.model_name,
                "choices": [choice],
                "usage": usage,
                "x_timing": timing.as_dict(),
            }
        )

    async def _collect(
        stream: AsyncGenerator[_Delta | _Finish, None],
    ) -> tuple[list[str], _Finish]:
        parts: list[str] = []
        finish = _Finish("length", 0)
        try:
            async with aclosing(stream) as events:
                async for ev in events:
                    if isinstance(ev, _Delta):
                        parts.append(ev.text)
                    else:
                        finish = ev
        except TimeoutError as exc:
            raise APIError(
                504, f"request exceeded {config.request_timeout_s:.0f}s", "timeout", "timeout"
            ) from exc
        return parts, finish

    def _gone(request: Request) -> Response:
        # Nobody receives this; 499 (client closed request) is for logs and metrics only.
        _count(request, 499, "client_disconnected")
        return Response(status_code=499)

    def _validate(body: CompletionRequest | ChatCompletionRequest, prompt_ids: list[int]) -> None:
        if body.model != config.model_name:
            raise APIError(
                404,
                f"model {body.model!r} is not served here",
                "not_found_error",
                "model_not_found",
            )
        feature = body.unsupported()
        if feature:
            raise APIError(
                400,
                f"{feature} is not supported by this server",
                "invalid_request_error",
                "unsupported_parameter",
            )
        if any(t >= config.vocab_size for t in prompt_ids):
            raise APIError(
                400,
                "prompt contains token ids outside the vocabulary",
                "invalid_request_error",
                "invalid_prompt",
            )
        if not prompt_ids:
            raise APIError(
                400, "prompt is empty after tokenization", "invalid_request_error", "invalid_prompt"
            )
        max_tokens = body.max_tokens or config.default_max_tokens
        if max_tokens > config.max_tokens_limit:
            raise APIError(
                400,
                f"max_tokens must be <= {config.max_tokens_limit}",
                "invalid_request_error",
                "invalid_parameter",
            )
        total = len(prompt_ids) + max_tokens
        limit = min(config.max_model_len, config.kv_capacity_tokens)
        if total > limit:
            raise APIError(
                400,
                f"prompt ({len(prompt_ids)} tokens) + max_tokens ({max_tokens}) "
                f"exceeds the {limit}-token limit",
                "invalid_request_error",
                "context_length_exceeded",
            )

    async def _first_event(
        raw: AsyncGenerator[_Delta | _Finish, None], deadline_ms: int | None
    ) -> _Delta | _Finish:
        try:
            if deadline_ms is None:
                return await anext(raw)
            # Cancelling anext() unwinds the generator, whose aclosing() aborts the engine request,
            # so a missed deadline never leaves work queued in the engine.
            return await asyncio.wait_for(anext(raw), deadline_ms / 1000)
        except TimeoutError as exc:
            m.rejections.labels("deadline_exceeded").inc()
            await raw.aclose()
            raise APIError(
                429,
                f"no first token within the {deadline_ms} ms deadline",
                "rate_limit_error",
                "deadline_exceeded",
                1,
            ) from exc
        except BaseException:
            await raw.aclose()
            raise

    async def _release_after(
        gen: AsyncGenerator[str, None], request: Request
    ) -> AsyncGenerator[str, None]:
        failed = False
        try:
            async with aclosing(gen) as g:
                async for item in g:
                    failed = failed or item.startswith('data: {"error"')
                    yield item
        finally:
            gate.inflight -= 1
            # A stream answers 200 even if it later fails in-band; count the failure separately.
            _count(request, 200, "stream_error" if failed else None)

    async def _text_stream(
        rid: str,
        prompt_ids: list[int],
        params: SamplingParams,
        stops: list[str],
        priority: Priority,
        tenant: str,
        timing: _Timing,
    ) -> AsyncGenerator[_Delta | _Finish, None]:
        """Engine tokens -> text deltas, deciding termination (EOS/length or stop string)."""
        detok = IncrementalDetokenizer(codec, prompt_ids)
        matcher = StopStringMatcher(stops)
        produced = 0
        last_token_at = timing.start
        events = engine.generate(rid, prompt_ids, params, priority=priority, tenant=tenant)
        deadline = asyncio.get_running_loop().time() + config.request_timeout_s
        async with aclosing(events) as evs:
            async for ev in _guard_dead(_before(evs, deadline)):
                if ev.error is not None:
                    raise APIError(500, ev.error, "server_error", "engine_failure")
                if ev.finish_reason is FinishReason.CAPACITY:
                    raise APIError(
                        400,
                        "request can never fit in the KV cache",
                        "invalid_request_error",
                        "context_length_exceeded",
                    )
                for tok in ev.token_ids:
                    produced += 1
                    now = time.perf_counter()
                    m.generation_tokens.inc()
                    if timing.first_token is None:
                        timing.first_token = now
                        m.ttft.observe(now - timing.start)
                    else:
                        m.itl.observe(now - last_token_at)
                    last_token_at = now
                    safe, hit = matcher.feed(detok.push(tok))
                    if safe:
                        yield _Delta(safe)
                    if hit:
                        # Leaving `async with` closes the engine stream, aborting the request.
                        timing.end = time.perf_counter()
                        m.latency.observe(timing.end - timing.start)
                        yield _Finish("stop", produced)
                        return
                if ev.finished:
                    tail = matcher.flush()
                    if tail:
                        yield _Delta(tail)
                    timing.end = time.perf_counter()
                    m.latency.observe(timing.end - timing.start)
                    reason = ev.finish_reason or FinishReason.LENGTH
                    yield _Finish(_FINISH.get(reason, "stop"), produced)
                    return

    async def _sse(
        rid: str,
        body: CompletionRequest | ChatCompletionRequest,
        chat: bool,
        prompt_tokens: int,
        stream: AsyncGenerator[_Delta | _Finish, None],
        timing: _Timing,
    ) -> AsyncGenerator[str, None]:
        created = int(time.time())
        obj = "chat.completion.chunk" if chat else "text_completion"

        def chunk(delta: str | None, finish: str | None, first: bool = False) -> str:
            choice: dict[str, Any] = {"index": 0, "finish_reason": finish, "logprobs": None}
            if chat:
                d: dict[str, str] = {"role": "assistant"} if first else {}
                if delta is not None:
                    d["content"] = delta
                choice["delta"] = d
            else:
                choice["text"] = delta or ""
            payload = {
                "id": rid,
                "object": obj,
                "created": created,
                "model": config.model_name,
                "choices": [choice],
            }
            return f"data: {json.dumps(payload)}\n\n"

        if chat:
            yield chunk("", None, first=True)
        completion_tokens = 0
        try:
            async with aclosing(stream) as events:
                async for ev in events:
                    if isinstance(ev, _Delta):
                        yield chunk(ev.text, None)
                    else:
                        completion_tokens = ev.completion_tokens
                        yield chunk(None if chat else "", ev.reason)
        except APIError as exc:
            # Headers are already sent: report the error in-band (as OpenAI does), then end.
            err = {"error": {"message": exc.message, "type": exc.err_type, "code": exc.code}}
            yield f"data: {json.dumps(err)}\n\n"
        except TimeoutError:
            err = {"error": {"message": "request timed out", "type": "timeout", "code": "timeout"}}
            yield f"data: {json.dumps(err)}\n\n"
        if body.include_usage:
            usage = {
                "id": rid,
                "object": obj,
                "created": created,
                "model": config.model_name,
                "choices": [],
                "usage": _usage(prompt_tokens, completion_tokens),
                "x_timing": timing.as_dict(),
            }
            yield f"data: {json.dumps(usage)}\n\n"
        yield "data: [DONE]\n\n"

    return app


class _ClientGoneError(Exception):
    """The client closed the connection before the response was sent."""


async def _wait_for_disconnect(request: Request) -> None:
    # The body is already consumed, so the next ASGI message is the disconnect notification.
    while (await request.receive())["type"] != "http.disconnect":
        await asyncio.sleep(0.05)  # defensive: never spin if a server sends other messages


async def _unless_disconnected[T](request: Request, work: Coroutine[Any, Any, T]) -> T:
    """Await work, cancelling it (which aborts the engine request) if the client leaves."""
    task = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(_wait_for_disconnect(request))
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task  # let the generator unwind and send the abort
        raise _ClientGoneError
    finally:
        watcher.cancel()
        task.cancel()  # no-op when done; covers cancellation of this coroutine itself


async def _chain(
    first: _Delta | _Finish, rest: AsyncGenerator[_Delta | _Finish, None]
) -> AsyncGenerator[_Delta | _Finish, None]:
    """Re-attach a prefetched first event; closing the chain closes (and aborts) the source."""
    async with aclosing(rest) as r:
        yield first
        if isinstance(first, _Finish):
            return
        async for ev in r:
            yield ev


def _deadline(header: str | None) -> int | None:
    if header is None:
        return None
    try:
        value = int(header)
    except ValueError:
        value = 0
    if value < 1:
        raise APIError(
            400,
            "X-Deadline-Ms must be a positive integer",
            "invalid_request_error",
            "invalid_header",
        )
    return value


async def _before(events: AsyncIterator[Any], deadline: float) -> AsyncGenerator[Any, None]:
    """Yield events until the loop-time deadline, then raise TimeoutError.

    Each wait is bounded separately, in whichever task is iterating. asyncio.timeout() around
    the whole iteration would be wrong here: it binds to the task that *entered* it, and these
    generators are started in one task (first-token wait) and iterated in another (the streaming
    response), so its cancellation would hit a task that has already moved on.
    """
    loop = asyncio.get_running_loop()
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError
        try:
            ev = await asyncio.wait_for(anext(events), remaining)
        except StopAsyncIteration:
            return
        yield ev


async def _guard_dead(events: AsyncIterator[Any]) -> AsyncGenerator[Any, None]:
    """Translate an engine that is down into a 503 (raised on first iteration, not at call time)."""
    try:
        async for ev in events:
            yield ev
    except EngineDeadError as exc:
        raise APIError(503, f"engine unavailable: {exc}", "service_unavailable") from exc


def _priority(header: str | None) -> Priority:
    if header is None or header == "interactive":
        return Priority.INTERACTIVE
    if header == "batch":
        return Priority.BATCH
    raise APIError(
        400,
        "X-Request-Priority must be 'interactive' or 'batch'",
        "invalid_request_error",
        "invalid_header",
    )


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
