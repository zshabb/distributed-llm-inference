"""API-key authentication. Tenant identity comes from the key, never from request headers.

Key format: ``dli_<key_id>_<secret>``. The server stores only ``{key_id, tenant, argon2id(secret)}``
(see :func:`new_key`), so a leaked key file does not reveal usable keys. Design choices:

* **Lookup by key id, verify one hash.** Verifying a presented key against every stored hash would
  cost one Argon2id evaluation per key. The id selects the single candidate.
* **No key-id enumeration by timing.** An unknown id still runs one dummy verification, so "no such
  key" and "wrong secret" take the same time.
* **Verified-token cache.** Argon2id is deliberately slow (tens of ms). Successful verifications are
  cached under the SHA-256 of the full token (never the token itself) so steady-state requests pay
  microseconds. Failures are never cached.
* **Tenant comes from authentication only.** The fairness scheduler trusts the tenant, so a client
  can never set it (``X-Tenant-Id`` from clients is ignored).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import sys
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_PREFIX = "dli_"
_HASHER = PasswordHasher()  # Argon2id with library defaults (RFC 9106 low-memory profile)
_DUMMY_HASH = _HASHER.hash("timing-equalizer")


class AuthError(Exception):
    """Missing, malformed, or invalid credentials (rendered as 401)."""


@dataclass(frozen=True, slots=True)
class StoredKey:
    key_id: str
    tenant: str
    secret_hash: str


def new_key(tenant: str) -> tuple[str, StoredKey]:
    """Create a key. Returns (token to give the client once, record to store server-side)."""
    key_id = secrets.token_hex(6)
    secret = secrets.token_urlsafe(32)
    return f"{_PREFIX}{key_id}_{secret}", StoredKey(key_id, tenant, _HASHER.hash(secret))


class ApiKeyAuth:
    """Validates ``Authorization: Bearer dli_<id>_<secret>`` and returns the caller's tenant."""

    def __init__(self, keys: Sequence[StoredKey], cache_size: int = 4096) -> None:
        self._by_id = {k.key_id: k for k in keys}
        if len(self._by_id) != len(keys):
            raise ValueError("duplicate key ids")
        self._cache: OrderedDict[bytes, str] = OrderedDict()
        self._cache_size = cache_size

    @classmethod
    def from_file(cls, path: str | Path) -> ApiKeyAuth:
        """Load ``[{"key_id": ..., "tenant": ..., "secret_hash": ...}, ...]`` from JSON."""
        rows = json.loads(Path(path).read_text())
        return cls([StoredKey(r["key_id"], r["tenant"], r["secret_hash"]) for r in rows])

    def cached(self, authorization: str | None) -> str | None:
        """Tenant for an already-verified token (fast path, no Argon2), else None."""
        if not authorization or not authorization.startswith("Bearer "):
            return None
        digest = hashlib.sha256(authorization[len("Bearer ") :].strip().encode()).digest()
        return self._cache.get(digest)

    def authenticate(self, authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthError("missing bearer token")
        token = authorization[len("Bearer ") :].strip()
        digest = hashlib.sha256(token.encode()).digest()
        tenant = self._cache.get(digest)
        if tenant is not None:
            self._cache.move_to_end(digest)
            return tenant
        key_id, secret = _split(token)
        stored = self._by_id.get(key_id) if key_id else None
        try:
            _HASHER.verify(stored.secret_hash if stored else _DUMMY_HASH, secret or "-")
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            raise AuthError("invalid API key") from None
        if stored is None:  # dummy verification "succeeded" only if secret matched the dummy
            raise AuthError("invalid API key")
        self._cache[digest] = stored.tenant
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return stored.tenant


class AuthMiddleware:
    """Authenticates ``/v1/*`` requests BEFORE any body parsing or validation.

    Running first means an unauthenticated client learns nothing about request schemas from
    validation errors, and no tokenization or template rendering happens for it. The verified tenant
    is stored in the ASGI scope (``request.state.tenant``). Uncached Argon2id checks run in a worker
    thread so they never stall other streams on the event loop.
    """

    def __init__(self, app: Any, auth: ApiKeyAuth | None, protected_prefix: str = "/v1/") -> None:
        self.app = app
        self.auth = auth
        self.prefix = protected_prefix

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(self.prefix):
            await self.app(scope, receive, send)
            return
        tenant = "default"
        if self.auth is not None:
            header = None
            for name, value in scope.get("headers", []):
                if name == b"authorization":
                    header = value.decode("latin-1")
            try:
                cached = self.auth.cached(header)
                tenant = (
                    cached
                    if cached is not None
                    else await asyncio.to_thread(self.auth.authenticate, header)
                )
            except AuthError as exc:
                err = {
                    "message": str(exc),
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
                headers = [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]
                await send({"type": "http.response.start", "status": 401, "headers": headers})
                await send(
                    {"type": "http.response.body", "body": json.dumps({"error": err}).encode()}
                )
                return
        scope.setdefault("state", {})["tenant"] = tenant
        await self.app(scope, receive, send)


def _split(token: str) -> tuple[str | None, str | None]:
    if not token.startswith(_PREFIX):
        return None, None
    key_id, sep, secret = token[len(_PREFIX) :].partition("_")
    return (key_id, secret) if sep and key_id and secret else (None, None)


def _cli(argv: list[str]) -> int:
    """``python -m dli.server.auth new <tenant>``: print a new key and its storable record."""
    if len(argv) != 2 or argv[0] != "new":
        print("usage: python -m dli.server.auth new <tenant>", file=sys.stderr)
        return 2
    token, record = new_key(argv[1])
    print(f"API key (shown once, give it to the client): {token}", file=sys.stderr)
    print(
        json.dumps(
            {"key_id": record.key_id, "tenant": record.tenant, "secret_hash": record.secret_hash}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
