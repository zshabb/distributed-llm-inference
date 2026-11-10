"""The official OpenAI Python SDK (pinned dev dependency) against the real server.

Compatibility means more than matching JSON: the SDK parses responses into typed models, iterates
SSE streams, maps HTTP statuses to exception classes, and honours Retry-After. A plain client
written for the OpenAI API must work unchanged, with server extensions passed via ``extra_body``.
"""

from __future__ import annotations

import asyncio

import openai
import pytest

from dli.server.auth import ApiKeyAuth, new_key

from .conftest import MODEL, LiveServer, ServerFactory, wait_for


def sdk(live: LiveServer, key: str = "unused") -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{live.url}/v1", api_key=key, max_retries=0, timeout=30)


def test_models_and_completions(server: LiveServer) -> None:
    client = sdk(server)
    assert [m.id for m in client.models.list()] == [MODEL]

    out = client.completions.create(
        model=MODEL, prompt="abc", max_tokens=6, temperature=0, extra_body={"ignore_eos": True}
    )
    assert out.object == "text_completion"
    assert out.choices[0].finish_reason == "length"
    assert out.usage is not None
    assert out.usage.completion_tokens == 6
    assert out.usage.total_tokens == out.usage.prompt_tokens + 6

    again = client.completions.create(
        model=MODEL, prompt="abc", max_tokens=6, temperature=0, extra_body={"ignore_eos": True}
    )
    assert again.choices[0].text == out.choices[0].text  # greedy is deterministic

    streamed = client.completions.create(
        model=MODEL,
        prompt="abc",
        max_tokens=6,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": True},
    )
    chunks = list(streamed)
    assert "".join(c.choices[0].text for c in chunks if c.choices) == out.choices[0].text
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 6


def test_chat_completions(server: LiveServer) -> None:
    client = sdk(server)
    messages: list[openai.types.chat.ChatCompletionMessageParam] = [
        {"role": "system", "content": "terse"},
        {"role": "user", "content": "hello"},
    ]
    out = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=8, temperature=0, seed=7
    )
    assert out.object == "chat.completion"
    assert out.choices[0].message.role == "assistant"
    assert out.choices[0].finish_reason in {"stop", "length"}

    stream = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=8, temperature=0, seed=7, stream=True
    )
    chunks = list(stream)
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == out.choices[0].message.content
    assert chunks[-1].choices[0].finish_reason == out.choices[0].finish_reason


def test_stop_sequences(server: LiveServer) -> None:
    client = sdk(server)
    full = (
        client.completions.create(
            model=MODEL, prompt="q", max_tokens=30, temperature=0, extra_body={"ignore_eos": True}
        )
        .choices[0]
        .text
    )
    stop = full[5:7]
    cut = client.completions.create(
        model=MODEL,
        prompt="q",
        max_tokens=30,
        temperature=0,
        stop=[stop],
        extra_body={"ignore_eos": True},
    ).choices[0]
    assert cut.finish_reason == "stop"
    assert cut.text == full[: full.index(stop)]


def test_errors_map_to_sdk_exceptions(make_server: ServerFactory) -> None:
    token, record = new_key("acme")
    live = make_server(auth=ApiKeyAuth([record]))
    with pytest.raises(openai.AuthenticationError):
        sdk(live, "dli_bogus_key").completions.create(model=MODEL, prompt="x")

    client = sdk(live, token)
    with pytest.raises(openai.BadRequestError) as bad:
        client.completions.create(model=MODEL, prompt="x", n=2)
    assert bad.value.code == "unsupported_parameter"
    with pytest.raises(openai.NotFoundError):
        client.completions.create(model="gpt-unknown", prompt="x")

    live.bundle.app.state.draining = True
    with pytest.raises(openai.InternalServerError) as busy:  # the SDK's class for 5xx
        client.completions.create(model=MODEL, prompt="x")
    assert busy.value.status_code == 503


def test_rate_limit_error_carries_retry_after(make_server: ServerFactory) -> None:
    live = make_server()
    client = sdk(live)
    client.completions.create(model=MODEL, prompt="warm up", max_tokens=4)
    with pytest.raises(openai.RateLimitError) as limited:
        client.completions.create(
            model=MODEL, prompt="y" * 200, max_tokens=4, extra_headers={"X-Deadline-Ms": "1"}
        )
    assert limited.value.code == "deadline_unattainable"
    assert int(limited.value.response.headers["retry-after"]) >= 1


def test_async_client_concurrency_and_cancellation(server: LiveServer) -> None:
    engine = server.bundle.engine

    async def run() -> None:
        async with openai.AsyncOpenAI(
            base_url=f"{server.url}/v1", api_key="unused", max_retries=0
        ) as client:
            outs = await asyncio.gather(
                *(
                    client.completions.create(
                        model=MODEL,
                        prompt=f"p{i}",
                        max_tokens=5,
                        temperature=0,
                        extra_body={"ignore_eos": True},
                    )
                    for i in range(8)
                )
            )
            assert all(o.usage is not None and o.usage.completion_tokens == 5 for o in outs)

            stream = await client.completions.create(
                model=MODEL,
                prompt="long",
                max_tokens=900,
                stream=True,
                extra_body={"ignore_eos": True},
            )
            seen = 0
            async for _ in stream:
                seen += 1
                if seen == 3:
                    break
            await stream.close()  # what an application does when its user goes away

    asyncio.run(run())
    wait_for(lambda: engine.snapshot.num_running == 0 and engine.snapshot.kv_used_blocks == 0)
    assert engine.snapshot.total_generated_tokens < 8 * 5 + 900
