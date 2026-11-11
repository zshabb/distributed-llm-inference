"""Admin listener: /metrics, /admin/state, /admin/drain. Served on a SEPARATE port.

Kept off the public API port because metrics labels and engine state reveal traffic and tenant
activity, and drain is a control action. The entry point binds it to 127.0.0.1 by default. If an
admin token is configured, every admin route requires ``Authorization: Bearer <token>`` (compared in
constant time).
"""

from __future__ import annotations

import dataclasses
import secrets
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from dli.server.async_engine import AsyncEngine
from dli.server.metrics import ServerMetrics


def create_admin_app(
    api_app: FastAPI, engine: AsyncEngine, metrics: ServerMetrics, token: str | None = None
) -> FastAPI:
    admin = FastAPI(title="dli-admin", docs_url=None, redoc_url=None)

    @admin.middleware("http")
    async def _require_token(request: Request, call_next: Any) -> Response:
        if token is not None:
            got = request.headers.get("authorization", "")
            if not secrets.compare_digest(got.encode(), f"Bearer {token}".encode()):
                return JSONResponse(
                    {"error": {"message": "admin token required", "type": "authentication_error"}},
                    status_code=401,
                )
        response: Response = await call_next(request)
        return response

    @admin.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @admin.get("/admin/state")
    async def state() -> dict[str, Any]:
        return {
            "engine": dataclasses.asdict(engine.snapshot),
            "inflight_requests": api_app.state.admission.inflight,
            "draining": api_app.state.draining,
        }

    @admin.post("/admin/drain")
    async def drain() -> dict[str, Any]:
        api_app.state.draining = True
        return {"draining": True, "inflight_requests": api_app.state.admission.inflight}

    @admin.post("/admin/undrain")
    async def undrain() -> dict[str, bool]:
        api_app.state.draining = False
        return {"draining": False}

    return admin
