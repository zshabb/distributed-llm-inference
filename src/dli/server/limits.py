"""ASGI middleware: reject oversized request bodies before the application sees them.

The body is read here, bounded by the limit, and only a body that fits is replayed to the app:

* a declared ``Content-Length`` above the limit is refused without reading anything;
* chunked uploads (no ``Content-Length``) are refused as soon as the received bytes exceed the
  limit, so memory use is bounded by ``max_bytes`` regardless of what the client sends.

Signalling "too large" by raising from ``receive()`` does not work: FastAPI's body parser catches
any exception there and answers 400. That was observed in testing, hence the buffer-and-replay
design.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class BodySizeLimit:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length" and value.isdigit() and int(value) > self.max_bytes:
                await _reject(send, self.max_bytes)
                return
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # client went away mid-upload; nothing to answer
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_bytes:
                await _reject(send, self.max_bytes)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()  # afterwards: pass through disconnect notifications

        await self.app(scope, replay, send)


async def _reject(send: Send, limit: int) -> None:
    err = {"message": f"request body exceeds {limit} bytes", "type": "invalid_request_error",
           "code": "body_too_large"}  # fmt: skip
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json")]})  # fmt: skip
    await send({"type": "http.response.body", "body": json.dumps({"error": err}).encode()})
