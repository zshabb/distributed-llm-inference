# Architecture

> Status markers: **[implemented]** exists and is tested · **[planned Mx]** designed, not built yet.
> Never read a planned component as available. Decision records: `docs/design-decisions/`.

## 1. System context

```
 OpenAI-compatible clients ──HTTPS/SSE──► Router (Go) [planned M8] ──► Engine workers (Python/PyTorch) [M1–M7]
                                              │                              │
                                   K8s EndpointSlices, load reports    GPU (CUDA/Triton) or CPU
```

## 2. Engine worker layers

| Layer | Package | Responsibility | Status |
|---|---|---|---|
| API | `dli.server` | OpenAI-compatible routes, SSE streaming, disconnect cancellation, admission/backpressure, auth, limits, metrics, admin, drain | implemented M4 |
| Admission / scheduling | `dli.engine.scheduler` | Token-budget step planning, chunked prefill, LIFO recompute preemption, priority + aging + weighted tenant fairness, capacity rejection, abort | implemented M3; deadline/capacity admission in front of it in `dli.server.admission` (M4) |
| KV memory | `dli.kv` | Paged block pool, block tables, ref counts, copy-on-write, prefix cache (SHA-256 chains), LRU/cost-aware eviction, paged KV store | implemented M2 |
| Runner / sampler / engine loop | `dli.engine.runner`, `dli.sampling`, `dli.engine.engine` | One batched forward per step (`BatchedKV`), per-request seeded sampling, step loop with timing | implemented M3 (CUDA graphs: M10) |
| Generation (bridge) | `dli.generate` | Single-sequence decoding over contiguous (M1) or paged (M2) KV: kept as the reference the engine is tested against | implemented M1–M2 |
| Model | `dli.model` | Llama-family decoder (Llama, Qwen2), weight loading, RoPE | implemented M1 |
| Kernels / parallel | `dli.kernels`, `dli.parallel` | Triton paged attention; tensor-parallel linear layers | planned M6 / M7 |
| Environment evidence | `dli.envinfo` | Hardware/software manifest for every result | implemented M0 |

Dependency rule: each layer depends only on layers below it. The model layer has no notion of requests, and the
scheduler never touches tensors.

## 3. Model layer (M1)

- `ModelConfig.from_hf_dict` maps a Hugging Face `config.json` to an immutable config. Unsupported features
  (sliding window attention, unknown RoPE scaling, non-SiLU activations) **raise** instead of silently mis-computing.
- `LlamaForCausalLM`: embedding → N × (RMSNorm → GQA attention with RoPE → residual → RMSNorm → SwiGLU MLP → residual)
  → RMSNorm → LM head (tied or untied). Qwen2 is the same graph with biases on the q/k/v projections.
- The forward takes explicit `positions` and an optional KV cache object, so the same model will later run under the
  paged-KV runner without changes to the math.
- `load_model` loads `safetensors` only (no pickle code execution) with exact tensor-name and shape checking: a
  missing, unexpected, or mis-shaped tensor is a hard error.

## 4. Server layer (M4)

```
 HTTP client ──► uvicorn / asyncio event loop                                   engine thread
                 BodySizeLimit → AuthMiddleware → route handler                  ┌──────────────────────┐
                 validate → admission check → AsyncEngine.generate() ──_Add───► │ LLMEngine.step() loop │
                 ◄── EngineEvent(token ids) via loop.call_soon_threadsafe ◄──── │ (scheduler, KV, model)│
                 detokenize → stop strings → JSON or SSE                        └──────────────────────┘
                 client gone / stop hit / deadline → generator closed ──_Abort─►  frees KV blocks
 admin port ──► /metrics (reads published EngineSnapshot), /admin/state|drain|undrain
```

- **One engine thread, one event loop.** The engine and its `BlockManager` are single-threaded (thread-confined since
  M2). `AsyncEngine` owns a dedicated thread that runs `step()` and receives typed commands (add, abort, stop) through
  a queue; results return to the loop with `call_soon_threadsafe`. Nothing on the event loop touches engine state.
- **Snapshot, not locks.** After each step the engine thread publishes an immutable `EngineSnapshot` (queue depths,
  KV usage, prefill-rate and step-time EWMAs, counters, health). Admission control, `/readyz`, `/admin/state` and
  Prometheus scrapes read it without synchronizing with the engine.
- **Cancellation is structural.** Every request is an async generator; closing it (client disconnect, stop string,
  timeout, deadline miss, server shutdown) aborts the engine request in a `finally`, so no path leaks KV blocks.
- **Admission before work.** Requests are refused fast (429 + `Retry-After`) when in-flight or queued limits are hit,
  or when the predicted time to first token (queued prefill tokens ÷ measured prefill rate) exceeds `X-Deadline-Ms`.
  A request still queued at its deadline is aborted and answered with 429 before any byte is sent.
- **Text layer.** `TextCodec` (tokenizer + chat template rendered in a sandboxed, immutable Jinja environment),
  `IncrementalDetokenizer` (never splits a UTF-8 character), `StopStringMatcher` (holds back possible partial
  matches so a stop string never leaks across chunk boundaries).
- **Security boundaries.** Auth runs as ASGI middleware before body parsing; Argon2 verification runs off the loop;
  the body limit applies before parsing; metrics and control endpoints live on a separate (loopback) listener.

## 5. Data ownership

The engine is stateless across restarts: model weights are read-only inputs, and KV cache contents are per-process
and reconstructible by recompute. No persistent store exists. The only server-side file is the API key list (Argon2id hashes), read at startup.

## 6. Failure boundaries (target)

Router ↔ worker over HTTP with health checks. A worker crash loses only its in-flight requests, which the router fails
fast or retries if safe (spec AC8). Within a worker, out-of-memory risk is handled by admission watermarks and preemption,
not crashes (M3).
