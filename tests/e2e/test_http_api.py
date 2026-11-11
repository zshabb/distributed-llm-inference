"""End-to-end over real HTTP: correctness, streaming, stop strings, disconnects, validation."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from dli.engine import SchedulerConfig
from dli.generate import greedy_generate

from .conftest import MODEL, LiveServer, ServerFactory, wait_for


def reference_text(server: LiveServer, prompt_ids: list[int], max_tokens: int) -> str:
    b = server.bundle
    ids = greedy_generate(b.model, prompt_ids, max_tokens, eos_token_ids=b.codec.eos_token_ids)
    return b.codec.decode(ids)


def sse_events(resp: httpx.Response) -> list[Any]:
    events = []
    for line in resp.iter_lines():
        if not line:
            continue
        assert line.startswith("data: "), line
        data = line[len("data: ") :]
        events.append("[DONE]" if data == "[DONE]" else json.loads(data))
    return events


# ------------------------------------------------------------------ basics


def test_ops_endpoints(server: LiveServer) -> None:
    with server.client() as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 200
        models = c.get("/v1/models").json()
        assert models["data"][0]["id"] == MODEL


def test_completion_matches_engine_independent_reference(server: LiveServer) -> None:
    prompt = "The quick brown fox"
    with server.client() as c:
        r = c.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": prompt, "max_tokens": 24, "temperature": 0},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    ids = server.bundle.codec.encode(prompt)
    assert body["choices"][0]["text"] == reference_text(server, ids, 24)
    assert body["usage"]["prompt_tokens"] == len(ids)
    assert body["object"] == "text_completion"
    assert body["x_timing"]["ttft_ms"] >= 0


def test_streaming_concatenation_equals_non_streaming(server: LiveServer) -> None:
    req = {"model": MODEL, "prompt": "hello world", "max_tokens": 30, "temperature": 0}
    with server.client() as c:
        full = c.post("/v1/completions", json=req).json()["choices"][0]["text"]
        stream_req = {**req, "stream": True, "stream_options": {"include_usage": True}}
        with c.stream("POST", "/v1/completions", json=stream_req) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            events = sse_events(r)
    assert events[-1] == "[DONE]"
    usage = events[-2]
    assert usage["choices"] == []
    assert usage["usage"]["completion_tokens"] > 0
    chunks = events[:-2]
    assert "".join(e["choices"][0]["text"] for e in chunks) == full
    assert chunks[-1]["choices"][0]["finish_reason"] in ("length", "stop")


def test_chat_completion_and_stream(server: LiveServer) -> None:
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    # temperature defaults to 1.0 (OpenAI semantics): request greedy decoding explicitly.
    req = {"model": MODEL, "messages": msgs, "max_tokens": 12, "temperature": 0}
    with server.client() as c:
        r = c.post("/v1/chat/completions", json=req)
        assert r.status_code == 200, r.text
        body = r.json()
        with c.stream("POST", "/v1/chat/completions", json={**req, "stream": True}) as s:
            events = sse_events(s)
    codec = server.bundle.codec
    ids = codec.render_chat([codec_msg(m) for m in msgs])
    assert codec.decode(ids) == "<system>be brief<user>hi<assistant>"
    expected = {"role": "assistant", "content": reference_text(server, ids, 12)}
    assert body["choices"][0]["message"] == expected
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    streamed = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1])
    assert streamed == body["choices"][0]["message"]["content"]
    assert events[-1] == "[DONE]"


def codec_msg(m: dict[str, str]) -> Any:
    from dli.server.text import ChatMessage

    return ChatMessage(m["role"], m["content"])


def test_concurrent_requests_are_all_correct(server: LiveServer) -> None:
    prompts = [f"request number {i}: " for i in range(16)]

    def one(p: str) -> tuple[str, str]:
        with server.client() as c:
            r = c.post(
                "/v1/completions",
                json={"model": MODEL, "prompt": p, "max_tokens": 20, "temperature": 0},
            )
            return p, r.json()["choices"][0]["text"]

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(one, prompts))
    for p, text in results:
        assert text == reference_text(server, server.bundle.codec.encode(p), 20), p


# ------------------------------------------------------------------ termination


def test_stop_string_spanning_tokens_truncates_and_never_leaks(server: LiveServer) -> None:
    prompt = "abcdefg"
    full = reference_text(server, server.bundle.codec.encode(prompt), 40)
    if len(full) < 10:
        pytest.skip("reference generation too short for a mid-text stop string")
    stop = full[5:8]  # 3 characters = 3 tokens with the char tokenizer: spans token boundaries
    expected = full[: full.index(stop)]
    req = {"model": MODEL, "prompt": prompt, "max_tokens": 40, "stop": [stop], "temperature": 0}
    with server.client() as c:
        body = c.post("/v1/completions", json=req).json()
        with c.stream("POST", "/v1/completions", json={**req, "stream": True}) as s:
            events = sse_events(s)
    assert body["choices"][0]["text"] == expected
    assert body["choices"][0]["finish_reason"] == "stop"
    streamed = "".join(e["choices"][0]["text"] for e in events if e != "[DONE]")
    assert streamed == expected
    assert stop not in streamed


def test_client_disconnect_mid_stream_aborts_and_frees_kv(server: LiveServer) -> None:
    engine = server.bundle.engine
    with (
        server.client() as c,
        c.stream(
            "POST",
            "/v1/completions",
            json={"model": MODEL, "prompt": "go", "max_tokens": 900, "stream": True},
        ) as r,
    ):
        seen = 0
        for line in r.iter_lines():
            if line.startswith("data: "):
                seen += 1
            if seen >= 3:
                break  # leaving the context closes the TCP connection mid-stream
    wait_for(lambda: engine.snapshot.num_running == 0 and engine.snapshot.num_waiting == 0)
    wait_for(lambda: engine.snapshot.kv_used_blocks == 0)
    assert engine.snapshot.total_generated_tokens < 900, "generation continued after disconnect"


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"prompt": "x"}, 400, "invalid_parameter"),  # missing model
        ({"model": MODEL}, 400, "invalid_parameter"),  # missing prompt
        ({"model": MODEL, "prompt": "x", "temperature": 5}, 400, "invalid_parameter"),
        ({"model": MODEL, "prompt": "x", "top_p": 0}, 400, "invalid_parameter"),
        ({"model": MODEL, "prompt": "x", "max_tokens": 0}, 400, "invalid_parameter"),
        ({"model": MODEL, "prompt": "x", "top_k": 0}, 400, "invalid_parameter"),
        (
            {"model": MODEL, "prompt": "x", "stop": ["a", "b", "c", "d", "e"]},
            400,
            "invalid_parameter",
        ),
        ({"model": MODEL, "prompt": ""}, 400, "invalid_parameter"),
        ({"model": MODEL, "prompt": [1, 2, 500]}, 400, "invalid_prompt"),  # outside the vocabulary
        ({"model": MODEL, "prompt": "x", "n": 2}, 400, "unsupported_parameter"),
        ({"model": MODEL, "prompt": "x", "presence_penalty": 0.5}, 400, "unsupported_parameter"),
        ({"model": MODEL, "prompt": "x", "logprobs": 3}, 400, "unsupported_parameter"),
        ({"model": MODEL, "prompt": "x", "best_of": 2}, 400, "unsupported_parameter"),
        ({"model": MODEL, "prompt": "x", "max_tokens": 5000}, 400, "invalid_parameter"),
        ({"model": MODEL, "prompt": "x" * 1020, "max_tokens": 10}, 400, "context_length_exceeded"),
        ({"model": "gpt-4", "prompt": "x"}, 404, "model_not_found"),
    ],
)
def test_invalid_completion_requests(
    server: LiveServer, payload: dict[str, Any], status: int, code: str
) -> None:
    with server.client() as c:
        r = c.post("/v1/completions", json=payload)
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert err["code"] == code
    assert err["message"]


@pytest.mark.parametrize(
    "payload",
    [
        {"model": MODEL, "messages": []},
        {"model": MODEL, "messages": [{"role": "robot", "content": "hi"}]},
        {"model": MODEL, "messages": [{"role": "user"}]},
        {"model": MODEL, "messages": [{"role": "user", "content": "x", "extra": 1}]},
    ],
)
def test_invalid_chat_requests(server: LiveServer, payload: dict[str, Any]) -> None:
    with server.client() as c:
        r = c.post("/v1/chat/completions", json=payload)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_malformed_json_and_bad_headers(server: LiveServer) -> None:
    with server.client() as c:
        r = c.post(
            "/v1/completions", content=b"{not json", headers={"content-type": "application/json"}
        )
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "invalid_request_error"
        r = c.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": "x"},
            headers={"X-Request-Priority": "urgent"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_header"
        assert c.get("/v1/nope").status_code == 404


def _abandon(live: LiveServer, payload: dict[str, Any]) -> None:
    """Send a request and drop the TCP connection before any response byte arrives."""
    with (
        httpx.Client(base_url=live.url, timeout=httpx.Timeout(30.0, read=0.3)) as c,
        pytest.raises(httpx.ReadTimeout),
    ):
        c.post("/v1/completions", json=payload)


def test_client_disconnect_during_non_streaming_request_aborts(make_server: ServerFactory) -> None:
    server = make_server(num_blocks=1024)  # room for 3000 tokens
    engine = server.bundle.engine
    _abandon(server, {"model": MODEL, "prompt": "go", "max_tokens": 3000, "ignore_eos": True})
    wait_for(lambda: engine.snapshot.num_running == 0 and engine.snapshot.kv_used_blocks == 0)
    assert engine.snapshot.total_generated_tokens < 3000, "generation continued after disconnect"
    assert server.bundle.app.state.admission.inflight == 0


def test_client_disconnect_while_queued_aborts_before_running(make_server: ServerFactory) -> None:
    live = make_server(
        scheduler=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64), num_blocks=1024
    )
    engine = live.bundle.engine
    holder = threading.Thread(
        target=_abandon,
        args=(live, {"model": MODEL, "prompt": "hold", "max_tokens": 1500, "ignore_eos": True}),
    )
    holder.start()
    wait_for(lambda: engine.snapshot.num_running == 1)
    _abandon(live, {"model": MODEL, "prompt": "queued", "max_tokens": 1500, "ignore_eos": True})
    holder.join()
    wait_for(lambda: engine.snapshot.num_running == 0 and engine.snapshot.num_waiting == 0)
    wait_for(lambda: engine.snapshot.kv_used_blocks == 0)
    assert engine.snapshot.total_generated_tokens < 1500, "abandoned requests kept generating"
