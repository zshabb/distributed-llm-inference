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
| API | `dli.api` | OpenAI routes, SSE streaming, cancellation, health | planned M4 |
| Admission / scheduling | `dli.scheduler` | Queues, token-budget step planning, chunked prefill, preemption, deadlines | planned M3 |
| KV memory | `dli.kv` | Paged block pool, block tables, ref counts, copy-on-write, prefix cache (SHA-256 chains), LRU/cost-aware eviction, paged KV store | implemented M2 |
| Runner | `dli.runner` | Batch metadata, forward, sampling, CUDA graphs | planned M3 |
| Generation (bridge) | `dli.generate` | Greedy decoding of ONE sequence over contiguous (M1) or paged (M2) KV; replaced by the batched runner in M3 | implemented M1–M2 |
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

## 4. Data ownership

The engine is stateless across restarts: model weights are read-only inputs, and KV cache contents are per-process
and reconstructible by recompute. No persistent store exists at M1.

## 5. Failure boundaries (target)

Router ↔ worker over HTTP with health checks. A worker crash loses only its in-flight requests, which the router fails
fast or retries if safe (spec AC8). Within a worker, out-of-memory risk is handled by admission watermarks and preemption,
not crashes (M3).
