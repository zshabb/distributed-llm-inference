# distributed-llm-inference

**An LLM inference engine built from first principles (paged KV cache, continuous batching, custom kernels, tensor
parallelism) plus a workload-aware router, measured against established baselines.**

> **Status: early development (milestones M0–M1).** What exists today is a verified model core: a native
> Llama-family implementation that reproduces Hugging Face `transformers` token-for-token. The serving features listed
> under *Planned capabilities* do **not** exist yet. This README only claims what the tests and labeled result files show.

## The engineering problem

Self-hosted LLM serving wastes GPU memory on KV-cache fragmentation, stalls short requests behind long ones, and throws
away reusable prompt prefixes when load balancers spread requests across replicas. This project builds the mechanisms
that address those problems itself, so each one can be studied, modified, and measured, instead of wrapping an existing engine.

## What works today

| Capability | Where | Evidence |
|---|---|---|
| Llama-family decoder (Llama 2/3, SmolLM2, Qwen2): GQA, RoPE (default/linear/llama3), SwiGLU, tied/untied heads | `src/dli/model/` | Logit parity vs `transformers` within 1e-4 on 7 architecture variants (`tests/unit/model/`) |
| Strict, pickle-free checkpoint loading at pinned revisions | `src/dli/model/loader.py`, `hub.py` | Rejection-path tests |
| Greedy generation with KV cache | `src/dli/generate.py` | Cached == recompute; 100/100 token-identical on two real models, fp32 CPU and bf16 GPU (see *Verified results*) |
| Inference API contract (OpenAPI 3.1) | `contracts/inference/v1/` | Schema-validated in tests; **not served yet** (M4) |
| Reproducible GPU host tooling | `scripts/gpu/`, `docs/operations.md` | Environment validation results in `benchmarks/results/` |

## Planned capabilities (not implemented)

Paged KV cache with prefix sharing and cost-aware eviction (M2) · continuous batching, chunked prefill, preemption (M3)
· OpenAI-compatible server with SSE, admission control, backpressure (M4) · benchmark harness vs HF and vLLM (M5) ·
Triton paged-attention kernel (M6) · tensor parallelism (M7) · Go prefix/KV-aware router (M8) · Kubernetes deployment
(M9). See `docs/implementation-plan.md`.

## Architecture

```
 clients ──► Router (Go, M8) ──► Engine workers (Python/PyTorch)
                                   api (M4) → scheduler (M3) → kv (M2) → runner (M3) → model (M1 ✔) → kernels (M6)
```
Details: `docs/architecture.md`, decisions: `docs/design-decisions/`, internals: `ENGINEERING_GUIDE.md`.

## Quick start (CPU)

```bash
uv sync --frozen --extra cpu
uv run --frozen pytest -q                  # fast suite (~10 s, no downloads)
uv run --frozen pytest -m parity -s        # real-checkpoint parity (downloads ~1.3 GB; 70 min on the i7-1355U, 15 min on the A16)
```

```python
from dli.model import hub, load_model
from dli.generate import greedy_generate
from transformers import AutoTokenizer  # tokenizer only; the model is ours

path = hub.download("HuggingFaceTB/SmolLM2-135M-Instruct")
tok, model = AutoTokenizer.from_pretrained(path), load_model(path)
ids = tok("The capital of France is", add_special_tokens=False)["input_ids"]
print(tok.decode(greedy_generate(model, ids, 16)))
```

## Verified results

Every number links to a result directory containing raw records and `env.json`. Labels:
`[CPU-LAPTOP]` = i7-1355U / WSL2, `[GPU:A16x1@vultr]` = one NVIDIA A16 (single-GPU evidence only).

### M1: greedy-decoding parity with Hugging Face `transformers` (correctness, not performance)

100 fixed prompts (70 completions, 30 chat-templated) × 128 greedy tokens, identical input ids for both
implementations, no EOS stopping, pinned checkpoints.

| Model (pinned revision) | fp32 · `[CPU-LAPTOP]` | bf16 · `[GPU:A16x1@vultr]` |
|---|---|---|
| SmolLM2-135M-Instruct (`12fd25f`) | **100 / 100** token-identical | **100 / 100** token-identical |
| Qwen2.5-0.5B-Instruct (`7ae5576`) | **100 / 100** token-identical | **100 / 100** token-identical |

Evidence: [`2026-11-10-CPU-LAPTOP-m1-greedy-parity-float32/`](benchmarks/results/2026-11-10-CPU-LAPTOP-m1-greedy-parity-float32/)
(see its `PROVENANCE.md`) and [`2026-11-10-GPU-A16x1-vultr-m1-greedy-parity-bfloat16/`](benchmarks/results/2026-11-10-GPU-A16x1-vultr-m1-greedy-parity-bfloat16/).

What this shows: for **these two checkpoints, these 100 prompts, greedy decoding of 128 tokens, fp32 on CPU and bf16
on one A16 GPU**, the native model, RoPE, GQA, tied heads, loader, and KV-cached decoding produced exactly the reference's
tokens. It is evidence for the tested configurations, **not a proof of universal numerical equivalence**: other
models, longer contexts (> ~181 positions here), sampling, batching, or other hardware are not covered by it. It says
nothing about speed. The wall-clock fields in the records are incidental, from sequential and unbatched runs.

### GPU host environment `[GPU:A16x1@vultr]`

NVIDIA A16 (sm_86, 14.61 GiB usable, PCI pass-through), driver 580.95.05, torch 2.14.1+cu130, Triton 3.8.0. fp32 matmul and
causal SDPA agree with CPU references to ≤ 6.2e-7 relative error, a Triton kernel compiles and runs, and the GPU is
visible inside containers. [`2026-11-10-GPU-A16x1-vultr-env-validation/`](benchmarks/results/2026-11-10-GPU-A16x1-vultr-env-validation/).
**Single-GPU evidence only.**

## Testing

| Suite | Command | Scope |
|---|---|---|
| Fast | `uv run --frozen pytest -q` | Unit, tiny-model parity, contract validation |
| Parity | `uv run --frozen pytest -m parity -s` | 100 prompts × 128 tokens, 2 real models |
| GPU | `scripts/gpu/remote.sh …` | See `docs/operations.md` |

## Security

Pickle-free loading, pinned revisions, locked and audited dependencies (both PyTorch builds), gitleaks, credential-free
GPU host. See `SECURITY.md`.

## Contributing / License

`CONTRIBUTING.md` · Apache-2.0 (`LICENSE`).
