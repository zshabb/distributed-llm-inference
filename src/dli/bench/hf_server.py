"""Baseline: Hugging Face ``transformers`` with static batching, behind the same OpenAI-style API.

Static batching, as compared against in the continuous-batching literature:

1. While the model is idle, the collector waits for a first request, then up to ``max_wait_ms`` for
   more, up to ``max_batch`` requests.
2. The batch is left-padded and run through ``model.generate`` until its **longest** request is
   done;
   shorter requests are truncated afterwards (their slots keep computing: the waste being measured).
3. Requests arriving meanwhile wait for the next batch.

Greedy decoding only (benchmarks use temperature 0); ``ignore_eos`` is honoured with
``min_new_tokens``. No streaming: the whole completion arrives at once, so this baseline's time to
first token equals its end-to-end latency, and the load generator runs it with ``stream=False``.

Usage: ``python -m dli.bench.hf_server --model <dir> [--device cuda --dtype bfloat16]``
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@dataclass
class _Pending:
    ids: list[int]
    max_tokens: int
    ignore_eos: bool
    future: asyncio.Future[tuple[list[int], str]] = field(repr=False)


class StaticBatcher:
    def __init__(self, model: Any, tokenizer: Any, *, max_batch: int, max_wait_ms: float) -> None:
        self.model, self.tok = model, tokenizer
        self.max_batch, self.max_wait = max_batch, max_wait_ms / 1000
        self.queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self.pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        self.batches = 0
        self.batch_sizes: list[int] = []

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except TimeoutError:
                    break
            try:
                outs = await asyncio.to_thread(self._generate, batch)
            except Exception as exc:  # fail the batch, keep serving
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(exc)
                continue
            for p, out in zip(batch, outs, strict=True):
                if not p.future.done():
                    p.future.set_result(out)

    @torch.inference_mode()
    def _generate(self, batch: list[_Pending]) -> list[tuple[list[int], str]]:
        self.batches += 1
        self.batch_sizes.append(len(batch))
        width = max(len(p.ids) for p in batch)
        dev = self.model.device
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, p in enumerate(batch):  # left padding: generation continues from the right edge
            ids[i, width - len(p.ids) :] = torch.tensor(p.ids)
            mask[i, width - len(p.ids) :] = 1
        steps = max(p.max_tokens for p in batch)
        forced = max((p.max_tokens for p in batch if p.ignore_eos), default=0)
        out = self.model.generate(
            input_ids=ids.to(dev),
            attention_mask=mask.to(dev),
            max_new_tokens=steps,
            min_new_tokens=forced,
            do_sample=False,
            pad_token_id=self.pad_id,
        )
        results = []
        eos = set(self._eos_ids())
        for i, p in enumerate(batch):
            gen = out[i, width:].tolist()[: p.max_tokens]
            if not p.ignore_eos:
                cut = next((j for j, t in enumerate(gen) if t in eos), None)
                gen = gen if cut is None else gen[: cut + 1]
            results.append((gen, self.tok.decode(gen, skip_special_tokens=True)))
        return results

    def _eos_ids(self) -> list[int]:
        e = self.model.generation_config.eos_token_id
        return [] if e is None else (list(e) if isinstance(e, list) else [int(e)])


def create_app(batcher: StaticBatcher, model_name: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(batcher.run())
        yield
        task.cancel()

    app = FastAPI(title="HF static-batching baseline", lifespan=lifespan)

    def err(status: int, msg: str) -> JSONResponse:
        return JSONResponse(
            {"error": {"message": msg, "type": "invalid_request_error", "code": None}}, status
        )

    async def serve(body: dict[str, Any], ids: list[int], chat: bool) -> JSONResponse:
        if body.get("stream"):
            return err(400, "streaming is not supported by the static-batching baseline")
        if body.get("temperature", 1.0) not in (0, 0.0):
            return err(400, "the baseline is greedy only: set temperature 0")
        fut: asyncio.Future[tuple[list[int], str]] = asyncio.get_running_loop().create_future()
        p = _Pending(
            ids, int(body.get("max_tokens") or 16), bool(body.get("ignore_eos", False)), fut
        )
        await batcher.queue.put(p)
        gen, text = await fut
        choice: dict[str, Any] = {
            "index": 0,
            "finish_reason": "length" if len(gen) >= p.max_tokens else "stop",
        }
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        usage = {
            "prompt_tokens": len(ids),
            "completion_tokens": len(gen),
            "total_tokens": len(ids) + len(gen),
        }
        return JSONResponse(
            {
                "id": f"hf-{uuid.uuid4().hex}",
                "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [choice],
                "usage": usage,
            }
        )

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "batches": batcher.batches,
            "mean_batch": sum(batcher.batch_sizes) / max(len(batcher.batch_sizes), 1),
        }

    @app.post("/v1/completions")
    async def completions(request: Request) -> JSONResponse:
        body = await request.json()
        prompt = body.get("prompt")
        ids = (
            list(prompt)
            if isinstance(prompt, list)
            else batcher.tok(str(prompt), add_special_tokens=False)["input_ids"]
        )
        return await serve(body, [int(t) for t in ids], chat=False)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> JSONResponse:
        body = await request.json()
        ids = batcher.tok.apply_chat_template(
            body["messages"], add_generation_prompt=True, tokenize=True
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return await serve(body, [int(t) for t in ids], chat=True)

    return app


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(prog="hf-static-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--max-wait-ms", type=float, default=20.0)
    a = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.model, dtype=getattr(torch, a.dtype), attn_implementation="sdpa"
        )
        .to(a.device)
        .eval()
    )
    batcher = StaticBatcher(model, tok, max_batch=a.max_batch, max_wait_ms=a.max_wait_ms)
    uvicorn.run(
        create_app(batcher, a.served_model_name or a.model),
        host=a.host,
        port=a.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
