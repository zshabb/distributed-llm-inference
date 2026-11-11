"""M4 milestone check: the HTTP server on real checkpoints, through the official OpenAI SDK.

The tiny e2e model proves the plumbing; this proves the text layer on real tokenizers (byte-level
BPE, multi-byte characters, special tokens, real chat templates and EOS ids):

1. Tokenization parity, all 100 M1 prompts: the server's prompt ids (``TextCodec.encode`` and the
   sandboxed ``render_chat``) equal ``transformers``' (tokenizer call / ``apply_chat_template``).
2. Generation over HTTP: the first ``DLI_HTTP_PROMPTS`` prompts are sent CONCURRENTLY (so the
   engine batches them) as completion or chat requests, greedy, EOS honoured. Each response text
   must equal ``transformers``' decode of the unbatched M1-path greedy tokens, cut at the first EOS.
   finish_reason and usage must match too. A divergence is accepted only where the reference's
   top-2 logit gap is a near-tie (same rule as M3: fp32 < 1e-3; bf16 <= 1.0).
3. Each request is also streamed; in fp32 the concatenated deltas must equal the non-streamed text.
   In bf16 this is recorded, not asserted: the two copies run in different batches, and bf16 output
   depends on batch composition (M3; see the 2026-11-11 bf16 result's ANALYSIS.md, where the
   original all-dtype assertion failed).

Run: ``uv run --frozen pytest -m parity -s tests/parity/test_http_real_checkpoint.py``
GPU: ``DLI_PARITY_DEVICE=cuda DLI_PARITY_DTYPE=bfloat16 DLI_EVIDENCE_LABEL=GPU-A16x1-vultr ...``
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openai
import pytest
import torch

from dli import envinfo
from dli.engine import SchedulerConfig
from dli.model import hub
from dli.server.factory import BuildOptions, build_server
from dli.server.text import ChatMessage
from tests.e2e.conftest import LiveServer, start_server, stop_server

from .test_engine_parity import _unbatched_with_gaps
from .test_hf_greedy_parity import _prompt_ids

MODELS = ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
NUM_PROMPTS = int(os.environ.get("DLI_HTTP_PROMPTS", "24"))
NEW_TOKENS = int(os.environ.get("DLI_HTTP_TOKENS", "48"))
DEVICE = os.environ.get("DLI_PARITY_DEVICE", "cpu")
DTYPE = getattr(torch, os.environ.get("DLI_PARITY_DTYPE", "float32"))
LABEL = os.environ.get("DLI_EVIDENCE_LABEL", "CPU-LAPTOP")
NEAR_TIE = 1e-3 if DTYPE is torch.float32 else 1.0
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_REVISION = envinfo.current_revision()
PROMPTS = [
    json.loads(line) for line in Path(__file__).with_name("prompts.jsonl").read_text().splitlines()
]

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module", params=MODELS)
def live(request: pytest.FixtureRequest) -> Iterator[tuple[str, LiveServer, Any]]:
    from transformers import AutoTokenizer

    repo_id: str = request.param
    path = hub.download(repo_id)
    opts = BuildOptions(
        model_name=repo_id,
        num_blocks=2048,
        block_size=16,
        device=DEVICE,
        dtype=DTYPE,
        scheduler=SchedulerConfig(max_num_seqs=32, max_num_batched_tokens=512),
    )
    server = start_server(build_server(path, opts))
    try:
        yield repo_id, server, AutoTokenizer.from_pretrained(path)
    finally:
        stop_server(server)


def _server_ids(server: LiveServer, prompt: dict[str, Any]) -> list[int]:
    codec = server.bundle.codec
    if prompt["kind"] == "chat":
        return codec.render_chat([ChatMessage(m["role"], m["content"]) for m in prompt["messages"]])
    return codec.encode(prompt["text"])


def test_prompt_tokenization_matches_transformers(live: tuple[str, LiveServer, Any]) -> None:
    _, server, hf = live
    mismatched = [p["id"] for p in PROMPTS if _server_ids(server, p) != _prompt_ids(hf, p)]
    assert not mismatched, f"prompt ids differ from transformers for prompts {mismatched}"


async def _call(client: openai.AsyncOpenAI, model: str, prompt: dict[str, Any]) -> dict[str, Any]:
    common: dict[str, Any] = {"model": model, "max_tokens": NEW_TOKENS, "temperature": 0}
    t0 = time.perf_counter()
    if prompt["kind"] == "chat":
        out = await client.chat.completions.create(messages=prompt["messages"], **common)
        text, finish = out.choices[0].message.content or "", out.choices[0].finish_reason
        stream = await client.chat.completions.create(
            messages=prompt["messages"], stream=True, **common
        )
        streamed = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
    else:
        res = await client.completions.create(prompt=prompt["text"], **common)
        text, finish, out = res.choices[0].text, res.choices[0].finish_reason, res  # type: ignore[assignment]
        cstream = await client.completions.create(prompt=prompt["text"], stream=True, **common)
        streamed = "".join([c.choices[0].text async for c in cstream if c.choices])
    assert out.usage is not None
    return {
        "text": text,
        "finish": finish,
        "streamed": streamed,
        "prompt_tokens": out.usage.prompt_tokens,
        "completion_tokens": out.usage.completion_tokens,
        "wall_s": time.perf_counter() - t0,
    }


def test_http_generation_matches_reference(live: tuple[str, LiveServer, Any]) -> None:
    repo_id, server, hf = live
    prompts = PROMPTS[:NUM_PROMPTS]
    eos = set(server.bundle.codec.eos_token_ids)

    async def run_all() -> list[dict[str, Any]]:
        async with openai.AsyncOpenAI(
            base_url=f"{server.url}/v1", api_key="unused", max_retries=0, timeout=1800
        ) as client:
            return await asyncio.gather(*(_call(client, repo_id, p) for p in prompts))

    t0 = time.perf_counter()
    results = asyncio.run(run_all())
    t_http = time.perf_counter() - t0

    records: list[dict[str, Any]] = []
    for prompt, res in zip(prompts, results, strict=True):
        ids = _prompt_ids(hf, prompt)
        toks, gaps = _unbatched_with_gaps(server.bundle.model, ids, NEW_TOKENS)
        cut = next((i for i, t in enumerate(toks) if t in eos), None)
        gen = toks if cut is None else toks[:cut]
        ref_text = hf.decode(gen, skip_special_tokens=True)
        record: dict[str, Any] = {
            "id": prompt["id"],
            "kind": prompt["kind"],
            "identical": res["text"] == ref_text,
            "stream_equals_non_stream": res["streamed"] == res["text"],
            "prompt_tokens_ok": res["prompt_tokens"] == len(ids),
            "reference_hit_eos": cut is not None,
        }
        if record["identical"]:
            record["finish_ok"] = res["finish"] == ("length" if cut is None else "stop")
            record["completion_tokens_ok"] = res["completion_tokens"] == (
                NEW_TOKENS if cut is None else cut + 1
            )
        else:
            # First reference token whose text is no longer a prefix of the server's output.
            k = next(
                i
                for i in range(len(gen))
                if not res["text"].startswith(hf.decode(gen[: i + 1], skip_special_tokens=True))
            )
            record["first_divergence"] = k
            record["ref_top2_gap_at_divergence"] = gaps[k]
        records.append(record)

    summary = {
        "model": repo_id,
        "revision": hub.PINNED_REVISIONS[repo_id],
        "code_revision": RUN_REVISION,
        "evidence_label": LABEL,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "prompts": len(records),
        "concurrent_requests": 2 * len(records),
        "max_new_tokens": NEW_TOKENS,
        "identical_texts": sum(r["identical"] for r in records),
        "stream_equals_non_stream": sum(r["stream_equals_non_stream"] for r in records),
        "reference_hit_eos": sum(r["reference_hit_eos"] for r in records),
        "finish_and_usage_correct": sum(
            r["identical"] and r["finish_ok"] and r["completion_tokens_ok"] for r in records
        ),
        "wall_seconds_http": round(t_http, 1),
        "note": "Correctness run; requests are concurrent and wall time is not a benchmark.",
    }
    out = (
        REPO_ROOT
        / "benchmarks"
        / "results"
        / (f"{datetime.now(UTC):%Y-%m-%d}-{LABEL}-m4-http-parity-{str(DTYPE).split('.')[-1]}")
    )
    out.mkdir(parents=True, exist_ok=True)
    envinfo.main([str(out / "env.json")], revision=RUN_REVISION)
    (out / f"{repo_id.split('/')[-1]}.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=2))

    for r in records:
        if DTYPE is torch.float32:
            assert r["stream_equals_non_stream"], r
        assert r["prompt_tokens_ok"], r
        if r["identical"]:
            assert r["finish_ok"], r
            assert r["completion_tokens_ok"], r
        else:
            assert r["ref_top2_gap_at_divergence"] < NEAR_TIE, r
