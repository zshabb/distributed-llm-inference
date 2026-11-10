from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from dli.server.auth import ApiKeyAuth, AuthError, new_key


def test_valid_key_resolves_tenant_and_store_holds_only_a_hash() -> None:
    token, record = new_key("acme")
    assert token.startswith("dli_")
    secret = token.split("_", 2)[2]
    assert secret not in record.secret_hash
    assert record.secret_hash.startswith("$argon2id$")
    assert ApiKeyAuth([record]).authenticate(f"Bearer {token}") == "acme"


@pytest.mark.parametrize(
    "header",
    [None, "", "Basic abc", "Bearer ", "Bearer sk-not-ours", "Bearer dli_", "Bearer dli_onlyid"],
)
def test_malformed_credentials_rejected(header: str | None) -> None:
    _, record = new_key("acme")
    with pytest.raises(AuthError):
        ApiKeyAuth([record]).authenticate(header)


def test_wrong_secret_and_unknown_id_rejected_with_comparable_cost() -> None:
    _, record = new_key("acme")
    auth = ApiKeyAuth([record])
    key_id = record.key_id
    wrong_secret = f"Bearer dli_{key_id}_wrong"
    unknown_id = "Bearer dli_ffffffffffff_wrong"
    timings = {}
    for name, header in (("wrong_secret", wrong_secret), ("unknown_id", unknown_id)):
        t0 = time.perf_counter()
        with pytest.raises(AuthError):
            auth.authenticate(header)
        timings[name] = time.perf_counter() - t0
    # Both paths must perform one Argon2id verification (no cheap early exit that reveals which
    # key ids exist). Argon2id with library defaults costs milliseconds; a dict miss costs ~1 us.
    assert min(timings.values()) > 0.003, timings


def test_verified_tokens_are_cached_failures_are_not() -> None:
    token, record = new_key("acme")
    auth = ApiKeyAuth([record])
    auth.authenticate(f"Bearer {token}")
    t0 = time.perf_counter()
    assert auth.cached(f"Bearer {token}") == "acme"
    assert time.perf_counter() - t0 < 0.001
    with pytest.raises(AuthError):
        auth.authenticate("Bearer dli_ffffffffffff_x")
    assert auth.cached("Bearer dli_ffffffffffff_x") is None


def test_key_file_round_trip(tmp_path: Path) -> None:
    t1, r1 = new_key("acme")
    t2, r2 = new_key("globex")
    path = tmp_path / "keys.json"
    path.write_text(
        json.dumps(
            [
                r.__dict__
                if hasattr(r, "__dict__")
                else {"key_id": r.key_id, "tenant": r.tenant, "secret_hash": r.secret_hash}
                for r in (r1, r2)
            ]
        )
    )
    auth = ApiKeyAuth.from_file(path)
    assert auth.authenticate(f"Bearer {t1}") == "acme"
    assert auth.authenticate(f"Bearer {t2}") == "globex"
