# M6: batched decode rows inside mixed prefill/decode steps

`[GPU:A5000x1@runpod]`, Qwen2.5-0.5B-Instruct bf16, revision `b3f47b5` (clean, `SYNC_FROM=head`), 2026-11-11.
One run per configuration: a before/after measurement, not a campaign. "Before" is the pipeline audit
(`…-m6-pipeline-audit/`, revision `7fd2671`) and the earlier HTTP A/B (`…-m6-cuda-graph-http-ab/`, revision
`5eda2db`), run with identical scripts, plan, settings and seeds.

Change: a step with prefill chunks and decode rows now attends the decode rows in one batched call (as a pure decode
step does) and only the prefill chunks per sequence; before, every row of such a step took the per-sequence path.

## In-process closed loop (`pipeline_profile.py`, 22:04:50–22:07:08 UTC with the HTTP runs below)

| closed-fixed, no HTTP | before | after | change |
|---|---|---|---|
| **c=64 graphs: output tok/s** | 2,781 | **5,086** | **1.83×** |
| c=64 graphs: mixed step mean (35 steps) | 206 ms | **52 ms** | −75% |
| c=64 graphs: mixed share of wall | 61% | 28% | |
| c=16 graphs: output tok/s | 2,255 | 2,485 | 1.10× |
| c=16 graphs: mixed step mean (11 steps) | 76 ms | 42 ms | −45% |
| c=64 eager: output tok/s | 1,983 | 2,776 | 1.40× |
| c=16 eager: output tok/s | 849 | 828 | ≈ (noise) |

Decode-only steps are unchanged (9.4 vs 9.3 ms at c=64, 5.5 ms at c=16). A mixed step with ~1,814 prefill tokens and
~51 decode rows now costs about what a pure 2,048-token prefill does (47–57 ms). The audit predicted 1.6–1.9× at c=64
and 1.05–1.15× at c=16.

## Over HTTP (plan `m6-a5000-cuda-graph-http-ab.json`, results `…-m6-mixed-step-http-ab/`)

| closed-fixed, HTTP | `--cuda-graphs` before | `--cuda-graphs` after | change |
|---|---|---|---|
| **c=64 output tok/s** | 2,166 | **3,171** | **1.46×** |
| c=64 TTFT p50 / p99 | 843 / 1,673 ms | **387 / 967 ms** | −54% / −42% |
| c=64 TPOT p50 / p99 | 23 / 28 ms | 17 / 20 ms | |
| c=64 ITL p99 | 296 ms | 101 ms | −66% |
| c=64 E2E p99 | 4,436 ms | 3,101 ms | −30% |
| c=16 output tok/s | 1,765 | 1,726 | ≈ (−2%, one repetition) |
| c=16 TTFT p50 / p99 | 180 / 299 ms | 208 / 320 ms | +16% / +7% |
| eager c=64 output tok/s / TTFT p50 | 1,128 / 651 ms | 1,297 / 385 ms | 1.15× / −41% |

- The c=64 TTFT p50 regression recorded with `--cuda-graphs` (651 → 843 ms) is gone: mixed steps were the cause.
  ITL p99 at c=64 fell from 296 to 101 ms, since a mixed step is the longest gap a stream sees.
- At c=16 mixed steps were only 23% of in-process wall; the in-process gain (1.10×) does not show over HTTP in one
  repetition (−2%, TTFT +16%). Not distinguishable from run-to-run noise without repetitions.
- Graph usage: 498 of 510 steps (c=16) and 465 of 502 (c=64) replayed a graph; no eager decode fallbacks; the
  remainder are mixed/prefill steps. No preemptions. Whole-device VRAM peak unchanged (21.78 / 21.92 GB).

## What this exposes
HTTP now reaches **38% less than in-process at c=64** (3,171 vs 5,086 tok/s) and 31% less at c=16 (was 22% at both):
the serving layer, not the engine, is now the largest gap at c=64. Mean running requests over HTTP were 52.9 of 64 at
c=64 (3.4 waiting), and GPU utilization averaged 54%.

## Correctness (same revision)
- CPU (fp32): mixed engines vs the per-sequence reference over GQA, biases, MHA, chunked prefill, preemption, prefix
  sharing and a reordered schedule: identical tokens, logits within 1e-4 every step, equal KV pools.
- GPU: `-m gpu` 14 passed. Real checkpoints over HTTP in fp32 on CUDA with `--cuda-graphs`
  (`…-m6-mixed-http-parity-float32/`): **24/24 texts identical** to the unbatched reference for Qwen2.5-0.5B and
  SmolLM2-135M, streamed = non-streamed, finish reasons and usage correct, 0 graph fallbacks.
