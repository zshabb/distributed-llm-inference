# distributed-llm-inference

**An LLM inference engine built from first principles (paged KV cache, continuous batching, custom kernels, tensor
parallelism) plus a workload-aware router, measured against established baselines.**

> **Status: active development (milestones M0–M4).** What exists today: a verified native Llama-family model, a paged
> KV-cache with prefix caching, a continuous-batching engine (chunked prefill, preemption, fairness, sampling) that
> reproduces the unbatched path token-for-token in fp32 on the tested configurations, and an OpenAI-compatible async
> HTTP server in front of it. **Nothing has been benchmarked yet** (M5): there are no latency or throughput claims for the
> server. The features listed under *Planned capabilities* do **not** exist. This README only claims what the tests and
> labeled result files show.

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
| Inference API contract C1 (OpenAPI 3.1) | `contracts/inference/v1/` | Every live response (statuses, bodies, stream events) validated against it over HTTP |
| Paged KV cache: block tables, ref counting, copy-on-write, watermark, atomic failure | `src/dli/kv/block_manager.py` | Model-based property tests + 4/4 injected faults caught; 10⁶-op and 10k-request soaks with 0 violations / 0 leaks `[CPU-LAPTOP]` |
| Prefix caching: SHA-256 hash-chained blocks, leaf-first eviction, LRU / cost-aware (GDSF) policies | `src/dli/kv/` | Paged decoding == contiguous decoding (tiny models; SmolLM2 40/40 runs); policy study `[SIM]` (see *Verified results*) |
| Continuous-batching engine: chunked prefill, LIFO recompute preemption, capacity rejection, cancellation, priority + aging + weighted tenant fairness | `src/dli/engine/` | Engine == unbatched path on 100 prompts × 128 tokens with preemption, fp32 CPU and GPU; randomized adversarial scheduler tests; fault-injected fairness/preemption checks |
| Sampling: greedy, temperature, top-k, top-p, per-request seeded generators (batch-invariant) | `src/dli/sampling.py` | Distribution, reproducibility, and batch-invariance tests |
| OpenAI-compatible server: `/v1/completions`, `/v1/chat/completions`, SSE streaming, stop strings, sandboxed chat templates | `src/dli/server/` | Official `openai` SDK tests; real-checkpoint HTTP parity (see *Verified results*) |
| Cancellation and overload: client-disconnect abort (streaming, non-streaming, queued), deadline/capacity admission with 429 + `Retry-After`, request timeouts, graceful SIGTERM drain | `src/dli/server/` | Real-HTTP tests check KV blocks return to 0 and generation stops; real-process drain test |
| Security controls: API keys (Argon2id hashes, checked before parsing), body limit, loopback defaults, sandboxed templates, separate admin listener | `src/dli/server/auth.py`, `limits.py`, `__main__.py` | Auth/limit/refusal tests; `SECURITY.md` |
| Prometheus metrics (requests, errors, rejections, tokens, TTFT/ITL/latency histograms, queue depth, KV usage, process/GPU memory) and admin state/drain | `src/dli/server/metrics.py`, `admin.py` | Counters asserted against exact traffic |
| Reproducible GPU host tooling | `scripts/gpu/`, `docs/operations.md` | Environment validation results in `benchmarks/results/` |

## Planned capabilities (not implemented)

Benchmark harness vs HF and vLLM, latency percentiles under load (M5) ·
Triton paged-attention kernel (M6) · tensor parallelism (M7) · Go prefix/KV-aware router (M8) · Kubernetes deployment
(M9). See `docs/implementation-plan.md`.

## Architecture

```
 clients ──► Router (Go, M8) ──► Engine workers (Python/PyTorch)
                                   api (M4 ✔) → scheduler (M3 ✔) → kv (M2 ✔) → runner (M3 ✔) → model (M1 ✔) → kernels (M6)
```
Details: `docs/architecture.md`, decisions: `docs/design-decisions/`, internals: `ENGINEERING_GUIDE.md`.

## Quick start (CPU)

```bash
uv sync --frozen --extra cpu
uv run --frozen pytest -q                  # fast suite (~10 s, no downloads)
uv run --frozen pytest -m parity -s        # real-checkpoint parity (downloads ~1.3 GB; 70 min on the i7-1355U, 15 min on the A16)
```

### Serve it (OpenAI-compatible)

```bash
uv run --frozen dli-serve --model HuggingFaceTB/SmolLM2-135M-Instruct --no-auth   # loopback only; see docs/operations.md for keys
```
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    stream=True,
    max_tokens=64,
    messages=[{"role": "user", "content": "Name three prime numbers."}],
):
    print(chunk.choices[0].delta.content or "", end="")
```

### Use the model directly

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

### M2: paged KV cache (correctness and policy study)

| Claim | Label | Evidence |
|---|---|---|
| Paged decoding produced the same tokens as the contiguous path in 40/40 runs (SmolLM2-135M, 20 prompts × {cold, shared-prefix} × 32 tokens, block 16). Every shared-prefix run served its prefix from cache | `[CPU-LAPTOP]` | [`2026-11-10-CPU-LAPTOP-m2-paged-parity/`](benchmarks/results/2026-11-10-CPU-LAPTOP-m2-paged-parity/) |
| 10⁶ random block-manager operations per policy: 0 invariant violations, 0 leaked blocks; 10,000-request lifecycle soak: 0 leaks | `[CPU-LAPTOP]` | `tests/stress/` (`-m stress`) |
| Cost-aware (GDSF) eviction beats LRU on shared-system-prompt traces (+2.5 to +4.1 hit-rate points under pressure, 5/5 seeds) but **loses on multi-turn traces** (down to −10.9 points). LRU remains the default | `[SIM]` (hit rate only, no latency) | [`2026-11-10-SIM-m2-prefix-eviction/`](benchmarks/results/2026-11-10-SIM-m2-prefix-eviction/), [`docs/algorithms/prefix-cache-eviction.md`](docs/algorithms/prefix-cache-eviction.md) |

Not yet measured: throughput, TTFT, or memory capacity gains from paging. They need the batched scheduler (M3) and the
benchmark harness (M5). The M2 paged read path gathers pages and is slower than the contiguous cache until the M6 kernel.

### M3: continuous-batching engine

| Claim | Label | Evidence |
|---|---|---|
| All 100 M1 prompts submitted at once (KV pool 25% of demand → 47 preemptions; 256-token budget → chunked prefill): engine output identical to the unbatched path for 100/100 prompts × 128 tokens, SmolLM2-135M and Qwen2.5-0.5B | `[CPU-LAPTOP]` fp32 · `[GPU:A16x1@vultr]` fp32 | [`CPU`](benchmarks/results/2026-11-10-CPU-LAPTOP-m3-engine-parity-float32/), [`GPU`](benchmarks/results/2026-11-10-GPU-A16x1-vultr-m3-engine-parity-float32/) |
| In **bf16**, batched vs unbatched outputs diverge for most prompts (22/100 and 18/100 identical): my pre-set criterion **failed**. Every divergence is at a bf16 near-tie (reference top-2 gap ≤ 0.375), and the fp32 runs above rule out an engine defect | `[GPU:A16x1@vultr]` | [`ANALYSIS.md`](benchmarks/results/2026-11-10-GPU-A16x1-vultr-m3-engine-parity-bfloat16/ANALYSIS.md) |
| **Preliminary CPU observation, not a GPU or production benchmark:** batched engine 107 vs 21 output tok/s for sequential decoding on SmolLM2-135M on a laptop CPU (5.2×, median of 3 interleaved runs, closed-loop, no latency percentiles); engine overhead 0.6% of step time | `[CPU-LAPTOP]` | [`m3-scheduler-overhead/`](benchmarks/results/2026-11-10-CPU-LAPTOP-m3-scheduler-overhead/) |

Not yet measured: latency percentiles (TTFT/ITL) under open-loop load, or GPU throughput vs vLLM (M5).

### M4: OpenAI-compatible server (correctness, not performance)

24 prompts (completions and chat) × ≤ 48 greedy tokens, each sent non-streaming and streaming: 48 concurrent requests
through the official `openai` SDK to the real server. Reference: `transformers`' decode of the unbatched M1-path tokens,
cut at the first EOS.

| Check | SmolLM2-135M | Qwen2.5-0.5B | Label |
|---|---|---|---|
| Prompt ids equal `transformers`' tokenizer / `apply_chat_template`, 100 prompts | 100/100 | 100/100 | `[CPU-LAPTOP]` |
| fp32: text identical to reference · stream == non-stream · finish_reason + usage | 24/24 · 24/24 · 24/24 | 24/24 · 24/24 · 24/24 | `[CPU-LAPTOP]` and `[GPU:A16x1@vultr]` |
| bf16: text identical to reference (all divergences at near-ties, gap ≤ 0.375) | 13/24 | 9/24 | `[GPU:A16x1@vultr]` |
| bf16: stream == non-stream (pre-set criterion, **failed**) | 11/24 | 15/24 | `[GPU:A16x1@vultr]` |

Evidence: [`CPU fp32`](benchmarks/results/2026-11-11-CPU-LAPTOP-m4-http-parity-float32/), [`GPU fp32`](benchmarks/results/2026-11-11-GPU-A16x1-vultr-m4-http-parity-float32/), [`GPU bf16 + ANALYSIS.md`](benchmarks/results/2026-11-11-GPU-A16x1-vultr-m4-http-parity-bfloat16/).
The bf16 failure is the batch-composition effect from M3: the two copies of a prompt run in different batches. The
fp32 control rules out a streaming defect. Only 4 of 48 reference generations reached EOS within 48 tokens, so EOS
termination on real models is covered more thinly than length termination. Wall times in the files are incidental,
not benchmarks.

Behaviour verified over real HTTP on a tiny model (63 tests): disconnect cancellation (KV returns to 0), overload
(429 + `Retry-After`), deadline misses, timeouts (504 / in-band), malformed input, auth, body limits, metrics accuracy,
SIGTERM drain on the real process, contract C1 conformance, and the official `openai` SDK.

### GPU host environment `[GPU:A16x1@vultr]`

NVIDIA A16 (sm_86, 14.61 GiB usable, PCI pass-through), driver 580.95.05, torch 2.14.1+cu130, Triton 3.8.0. fp32 matmul and
causal SDPA agree with CPU references to ≤ 6.2e-7 relative error, a Triton kernel compiles and runs, and the GPU is
visible inside containers. [`2026-11-10-GPU-A16x1-vultr-env-validation/`](benchmarks/results/2026-11-10-GPU-A16x1-vultr-env-validation/).
**Single-GPU evidence only.**

## Testing

| Suite | Command | Scope |
|---|---|---|
| Fast | `uv run --frozen pytest -q` | Unit, tiny-model parity, contract validation, real-HTTP server e2e (231 tests, ~45 s) |
| Parity | `uv run --frozen pytest -m parity -s` | 100 prompts × 128 tokens, 2 real models; HTTP parity via the SDK |
| GPU | `scripts/gpu/remote.sh …` | See `docs/operations.md` |

## Security

Pickle-free loading, pinned revisions, locked and audited dependencies (both PyTorch builds), gitleaks, credential-free
GPU host. Server: authentication on by default (Argon2id-hashed API keys), loopback binding by default, body and
generation limits, sandboxed chat templates, metrics and controls on a separate loopback listener. See `SECURITY.md`
and `docs/operations.md`.

## Contributing / License

`CONTRIBUTING.md` · Apache-2.0 (`LICENSE`).
