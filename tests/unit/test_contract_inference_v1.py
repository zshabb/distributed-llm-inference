"""Contract C1 must be a valid OpenAPI 3.1 document exposing the agreed surface.

Consumers in other portfolio repositories vendor this file. Any accidental removal of a path,
extension header, or error response is a breaking change and must fail here first.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from openapi_spec_validator import validate

SPEC = Path(__file__).resolve().parents[2] / "contracts" / "inference" / "v1" / "openapi.yaml"


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(SPEC.read_text())
    return loaded


def test_document_is_valid_openapi_31(spec: dict[str, Any]) -> None:
    validate(spec)  # raises on any schema violation
    assert spec["openapi"].startswith("3.1")


def test_required_paths_exist(spec: dict[str, Any]) -> None:
    assert {"/v1/models", "/v1/completions", "/v1/chat/completions"} <= spec["paths"].keys()


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
def test_generation_endpoints_declare_contract(spec: dict[str, Any], path: str) -> None:
    op = spec["paths"][path]["post"]
    headers = {
        spec["components"]["parameters"][p["$ref"].rsplit("/", 1)[1]]["name"]
        for p in op["parameters"]
    }
    assert headers == {"X-Request-Priority", "X-Deadline-Ms", "X-Session-Id", "X-Tenant-Id"}
    assert {"200", "400", "429", "503"} <= op["responses"].keys()


def test_extension_headers_are_optional(spec: dict[str, Any]) -> None:
    # Optional by contract: a plain OpenAI client must work without them.
    assert all(not p.get("required", False) for p in spec["components"]["parameters"].values())
