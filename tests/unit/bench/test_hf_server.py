"""The HF static-batching baseline must be a *correct* baseline: batching may not change outputs.

A weak or broken baseline would inflate the comparison, so the batched, left-padded path is checked
against unbatched greedy decoding of the same ``transformers`` model, token for token (fp32).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from dli.bench.hf_server import StaticBatcher, _Pending, create_app
from tests.helpers.tiny import build_tiny_model, write_char_tokenizer


@pytest.fixture(scope="module")
def hf(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d: Path = tmp_path_factory.mktemp("hf-tiny")
    build_tiny_model("llama", d)
    write_char_tokenizer(d)
    model = AutoModelForCausalLM.from_pretrained(
        d, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    return model, AutoTokenizer.from_pretrained(d)


def _unbatched(model: object, ids: list[int], n: int) -> list[int]:
    out = model.generate(  # type: ignore[attr-defined]
        input_ids=torch.tensor([ids]),
        max_new_tokens=n,
        min_new_tokens=n,
        do_sample=False,
        pad_token_id=0,
    )
    return [int(t) for t in out[0, len(ids) :]]


def test_left_padded_static_batch_equals_unbatched_greedy(hf: tuple[object, object]) -> None:
    model, tok = hf
    b = StaticBatcher(model, tok, max_batch=8, max_wait_ms=1)
    prompts = [[5, 6, 7], [8, 9, 10, 11, 12, 13, 14], [15], [16, 17, 18, 19, 20]]
    lengths = [4, 9, 2, 6]
    loop = asyncio.new_event_loop()
    pend = [
        _Pending(p, n, True, loop.create_future()) for p, n in zip(prompts, lengths, strict=True)
    ]
    outs = b._generate(pend)
    loop.close()
    for (gen, _), p, n in zip(outs, prompts, lengths, strict=True):
        assert len(gen) == n  # truncated to its own max_tokens after the batch ran to 9
        assert gen == _unbatched(model, p, n)


def test_api_batches_concurrent_requests_and_reports_usage(hf: tuple[object, object]) -> None:
    model, tok = hf
    b = StaticBatcher(model, tok, max_batch=4, max_wait_ms=200)
    with TestClient(create_app(b, "tiny")) as c:
        import concurrent.futures

        body = {
            "model": "tiny",
            "prompt": [5, 6, 7],
            "max_tokens": 5,
            "temperature": 0,
            "ignore_eos": True,
        }
        with concurrent.futures.ThreadPoolExecutor(4) as ex:
            rs = list(ex.map(lambda _: c.post("/v1/completions", json=body), range(4)))
        assert all(r.status_code == 200 for r in rs)
        assert {r.json()["usage"]["completion_tokens"] for r in rs} == {5}
        assert b.batches == 1  # all four arrived within the window: one static batch
        assert c.post("/v1/completions", json={**body, "stream": True}).status_code == 400
        assert c.post("/v1/completions", json={**body, "temperature": 0.7}).status_code == 400
        chat = c.post(
            "/v1/chat/completions",
            json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "temperature": 0,
                "ignore_eos": True,
            },
        )
        assert chat.status_code == 200
        assert chat.json()["usage"]["completion_tokens"] == 3
