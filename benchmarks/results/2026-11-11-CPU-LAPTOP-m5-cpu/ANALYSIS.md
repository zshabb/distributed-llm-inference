# M5 CPU: continuous batching vs HF static batching (AC3, CPU part)

`[CPU-LAPTOP]` i7-1355U (WSL2), SmolLM2-135M-Instruct fp32, 5 repetitions with backend order rotated
(`benchmarks/plans/m5-cpu.json`, revision `d6ffccc`). Raw data: `runs/`, generated tables: `SUMMARY.md`.

## AC3 (CPU): **failed**
Pre-registered criterion: native engine output tokens/s ≥ 2× HF static batching at closed-loop concurrency 16.

| Concurrency 16, input 128 / output 64 | Output tok/s, median [min–max], 5 reps |
|---|---|
| native engine (`dli`) | 54.5 [51.1–57.8] |
| HF static batching | 74.4 [65.6–78.4] |
| **ratio** | **0.73×** (criterion 2×) |

At concurrency 1 the two are equal within noise (17.7 vs 18.7 tok/s): both run one sequence per forward.

## What the native engine does better here
Static batching makes every request in a batch wait for the batch's longest member and makes later arrivals wait
for the whole batch. Under open-loop Poisson load (log-normal lengths, 0.5 req/s), median TTFT was **539 ms** for the
native engine vs **31.6 s** for HF static batching (where TTFT equals end-to-end latency: it does not stream). At
1 req/s the native engine's median TTFT was 1.2 s; the HF sweep had already stopped at 0.5 req/s. Continuous
batching wins on latency under load even though it loses on peak throughput here.

## Why it loses on throughput
The same cause the GPU profile measures (`2026-11-11-GPU-A16x1-vultr-m5-engine-profile/`): attention runs as a
per-sequence Python loop in every layer, with the cached prefix gathered from pages and GQA heads expanded by copy
for each sequence, each decode step. HF runs one padded, batched attention call per layer. On a CPU, the per-sequence
loop also forfeits the matrix-multiply efficiency that batching gives the HF path. The fix is the M6 batched
paged-attention kernel; this result is the baseline it will be measured against.

## Context
M3 measured 107 vs 21 tok/s (5.2×) for the engine vs *sequential unbatched* decoding, in-process. That comparison
had no batched baseline; this one does, through HTTP, and the batched baseline is faster. Both are kept.
Per the owner's GPU-first policy (2026-11-11) no further CPU performance campaigns are planned; AC3's CPU part is
recorded as failed and, if re-measured after M6, will be re-measured as this single focused point.
