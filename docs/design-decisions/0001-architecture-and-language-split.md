# ADR-0001: System architecture and language split

- Status: Accepted
- Date: 2026-11-10
- Context source: `specs/01-distributed-llm-inference.md` (portfolio planning repository)

## Context

We are building an inference-serving system whose value lies in **original engine and routing mechanisms** (paged
KV cache, continuous batching, admission control, a paged-attention kernel, tensor parallelism, workload-aware routing),
measured against established baselines (Hugging Face `transformers`, vLLM). The system must run correctness tests on a
CPU-only laptop and performance tests on NVIDIA GPUs, and expose an OpenAI-compatible API.

## Decision

Two deployable components with a strict boundary:

1. **Engine worker (Python 3.12 + PyTorch, Triton for GPU kernels).** One process owns one model replica (one GPU, or a
   tensor-parallel group). Internal layering, with dependencies pointing downward only:
   `api → scheduler/admission → kv (block manager, prefix cache) → runner → model → kernels/parallel`.
   The model layer never knows about requests, and the scheduler never touches tensors.
2. **Router (Go).** Stateless HTTP/SSE proxy: discovery, health, load collection, routing policy, retries for
   requests that are safe to retry. It talks to workers only through the public OpenAI API plus a small internal load
   endpoint, so it can also front vLLM workers.

## Rationale

- **Python/PyTorch for the engine:** weights, tokenizers, and `torch.distributed` are native there, and the same code runs
  on CPU (correctness) and CUDA (performance). The hot path lives in kernels, not interpreter overhead. The scheduler
  overhead per step will be measured (M5) instead of assumed small.
- **Triton over raw CUDA for custom kernels:** readable, and testable on CPU through the Triton interpreter, which matters
  because the development machine has no GPU.
- **Go for the router:** predictable latency for long-lived SSE streams, `client-go` for Kubernetes discovery, and a single
  static binary. A Python router would share the GIL with no benefit, and Envoy would push policy logic into Lua/Wasm.
- **Strict layering** keeps the original components (block manager, scheduler) unit-testable without a model, and lets
  the model be tested without a scheduler.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Wrap vLLM/SGLang and add features | Defeats the purpose: the engine internals are the original contribution. vLLM stays as a **baseline** |
| C++/CUDA engine | Far slower iteration, and kernels could not be tested on the CPU-only machine |
| Single Python process for router + engine | Couples failure domains; the router must survive worker crashes and front heterogeneous backends |
| Pipeline parallelism in v1 | Mainly helps multi-node serving, which is out of validation scope; TP covers target model sizes (spec §8) |

## Consequences

- Two toolchains (uv + Go) and two CI jobs.
- Python step overhead is a known risk. Mitigations: CUDA graphs for decode (M10) and keeping per-step Python work O(batch).
- The internal load-report endpoint is a versioned contract between router and worker.
