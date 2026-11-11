"""Live responses must conform to contract C1 (contracts/inference/v1/openapi.yaml).

Every response below comes from the real server over TCP. Each is checked twice: its status code
must be documented for that operation, and its body must validate against the schema the contract
names for that status (JSON Schema 2020-12, as OpenAPI 3.1 specifies). Streams are checked event
by event. A field rename, a new undocumented status, or a changed error shape fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from dli.server.auth import ApiKeyAuth, new_key

from .conftest import MODEL, LiveServer, ServerFactory

SPEC_PATH = Path(__file__).resolve().parents[2] / "contracts" / "inference" / "v1" / "openapi.yaml"
SPEC: dict[str, Any] = yaml.safe_load(SPEC_PATH.read_text())
URI = "urn:dli:contract:inference:v1"
REGISTRY: Registry[Any] = Registry().with_resource(
    URI, Resource.from_contents(SPEC, default_specification=DRAFT202012)
)


def _validator(pointer: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": f"{URI}#{pointer}"}, registry=REGISTRY)


def _deref(node: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in node:
        target: Any = SPEC
        for part in node["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        node = target
    return node


def _errors(validator: Draft202012Validator, instance: object) -> list[str]:
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(instance)]


CHUNK = _validator("/components/schemas/CompletionChunk")
ERROR = _validator("/components/schemas/Error")


def conforms(path: str, method: str, r: httpx.Response) -> None:
    """The status is documented for the operation and the JSON body matches its schema."""
    responses = SPEC["paths"][path][method]["responses"]
    status = str(r.status_code)
    assert status in responses, f"{method.upper()} {path} returned undocumented {status}: {r.text}"
    spec = _deref(responses[status])
    for header in spec.get("headers", {}):
        assert header in r.headers, f"{status} must carry {header}"
    schema = spec["content"]["application/json"]["schema"]
    errs = _errors(_validator(schema["$ref"].removeprefix("#")), r.json())
    assert not errs, f"{status} body violates the contract: {errs}"


def stream_conforms(path: str, r: httpx.Response) -> list[dict[str, Any]]:
    """A 200 event stream: every event is a CompletionChunk (or in-band Error), then [DONE]."""
    assert r.status_code == 200
    assert "text/event-stream" in _deref(SPEC["paths"][path]["post"]["responses"]["200"])["content"]
    assert r.headers["content-type"].startswith("text/event-stream")
    data = [line.removeprefix("data: ") for line in r.iter_lines() if line.startswith("data: ")]
    assert data[-1] == "[DONE]"
    events = [json.loads(d) for d in data[:-1]]
    for ev in events:
        errs = _errors(ERROR if "error" in ev else CHUNK, ev)
        assert not errs, f"stream event violates the contract: {errs} in {ev}"
    return events


def _request_conforms(schema: str, body: dict[str, Any]) -> dict[str, Any]:
    """Guard against testing the server with requests the contract itself would not allow."""
    errs = _errors(_validator(f"/components/schemas/{schema}"), body)
    assert not errs, f"test request is not contract-valid: {errs}"
    return body


def completion(**kw: Any) -> dict[str, Any]:
    return _request_conforms("CompletionRequest", {"model": MODEL, "prompt": "hello", **kw})


def chat(**kw: Any) -> dict[str, Any]:
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    return _request_conforms("ChatCompletionRequest", {"model": MODEL, "messages": msgs, **kw})


def test_models(server: LiveServer) -> None:
    with server.client() as c:
        conforms("/v1/models", "get", c.get("/v1/models"))


def test_success_bodies(server: LiveServer) -> None:
    with server.client() as c:
        for body in (
            completion(max_tokens=6, temperature=0),
            completion(prompt=[5, 6, 7], max_tokens=6, seed=3, top_p=0.9, top_k=5),
            completion(max_tokens=20, temperature=0, stop=["e", "l"], ignore_eos=True),
        ):
            conforms("/v1/completions", "post", c.post("/v1/completions", json=body))
        r = c.post("/v1/chat/completions", json=chat(max_tokens=6, temperature=0))
        conforms("/v1/chat/completions", "post", r)
        assert r.json()["choices"][0]["message"]["role"] == "assistant"


def test_every_contract_sampling_field_is_accepted(server: LiveServer) -> None:
    """A request using every SamplingParams property at a contract-valid value must succeed."""
    every = {
        "max_tokens": 4,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 1,
        "stop": "zz",
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "ignore_eos": False,
        "stream": False,
        "stream_options": {"include_usage": False},
    }
    assert set(every) == set(SPEC["components"]["schemas"]["SamplingParams"]["properties"])
    with server.client() as c:
        conforms("/v1/completions", "post", c.post("/v1/completions", json=completion(**every)))
        conforms("/v1/chat/completions", "post", c.post("/v1/chat/completions", json=chat(**every)))


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
def test_streams(server: LiveServer, path: str) -> None:
    make = completion if path == "/v1/completions" else chat
    with server.client() as c:
        for opts in ({}, {"stream_options": {"include_usage": True}}):
            body = make(max_tokens=7, stream=True, ignore_eos=True, **opts)
            with c.stream("POST", path, json=body) as r:
                events = stream_conforms(path, r)
            finals = [e for e in events if e["choices"] and e["choices"][0]["finish_reason"]]
            assert [e["choices"][0]["finish_reason"] for e in finals] == ["length"]
            if opts:
                assert events[-1]["choices"] == []
                assert events[-1]["usage"]["completion_tokens"] == 7


def test_client_errors(server: LiveServer) -> None:
    with server.client() as c:
        for body, code in (
            ({"model": MODEL, "prompt": "x", "n": 3}, 400),  # outside the C1 subset
            ({"model": MODEL, "prompt": "x", "temperature": 3}, 400),  # contract maximum is 2
            ({"model": MODEL, "messages": []}, 400),
            ({"model": "other-model", "prompt": "x"}, 404),
        ):
            path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
            r = c.post(path, json=body)
            assert r.status_code == code, r.text
            conforms(path, "post", r)
        r = c.post("/v1/completions", content=b"{not json", headers={"content-type": "x"})
        conforms("/v1/completions", "post", r)


def test_auth_and_size_errors(make_server: ServerFactory) -> None:
    token, record = new_key("acme")
    live = make_server(auth=ApiKeyAuth([record]), max_body_bytes=2048)
    with live.client() as c:
        r = c.post("/v1/completions", json=completion())
        assert r.status_code == 401
        conforms("/v1/completions", "post", r)
        r = c.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json=chat(messages=[{"role": "user", "content": "x" * 4096}]),
        )
        assert r.status_code == 413
        conforms("/v1/chat/completions", "post", r)


def test_overload_unavailable_and_timeout(make_server: ServerFactory) -> None:
    live = make_server()
    with live.client() as c:
        c.post("/v1/completions", json=completion(max_tokens=4))  # measures the prefill rate
        r = c.post(
            "/v1/completions",
            headers={"X-Deadline-Ms": "1"},
            json=completion(prompt="y" * 200, max_tokens=4),
        )
        assert r.status_code == 429, r.text
        conforms("/v1/completions", "post", r)

    live = make_server(request_timeout_s=0.05, num_blocks=1024)
    with live.client() as c:
        r = c.post("/v1/completions", json=completion(max_tokens=3000, ignore_eos=True))
        assert r.status_code == 504, r.text
        conforms("/v1/completions", "post", r)
        live.bundle.app.state.draining = True
        r = c.post("/v1/chat/completions", json=chat())
        assert r.status_code == 503
        conforms("/v1/chat/completions", "post", r)
