# M6 HTTP A/B: eager batched decode vs `--cuda-graphs`, closed-fixed at concurrency 16 and 64

`[GPU:A5000x1@runpod]`, Qwen2.5-0.5B-Instruct bf16, revision `5eda2db` (clean, `SYNC_FROM=head`), 2026-11-11
21:05:19–21:06:58 UTC (99 s of A5000 time). Plan `benchmarks/plans/m6-a5000-cuda-graph-http-ab.json`: the native
server started fresh per variant and warmed up, then closed-loop clients over HTTP (streaming), input exactly 256
tokens, output exactly 128, `4 x concurrency` requests per point, the same seeds for both variants. Both variants use
the focused M6 baseline settings (block 16, 100,416 KV blocks, `max_num_seqs` 128, `max_num_batched_tokens` 2048) and
differ only in `--cuda-graphs` (buckets 16/64 rows x 24 blocks). **One repetition**: a short comparison, not a campaign;
no confidence intervals.

## Result

| closed-fixed | eager batched | `--cuda-graphs` | change |
|---|---|---|---|
| **c=16 output tok/s** | 602 | **1,765** | **2.9×** |
| c=16 TTFT p50 / p99 | 183 / 326 ms | 180 / 299 ms | same / −8% |
| c=16 TPOT p50 / p99 | 25 / 26 ms | **8 / 9 ms** | −68% |
| c=16 E2E p99 | 3,570 ms | 1,247 ms | −65% |
| **c=64 output tok/s** | 1,128 | **2,166** | **1.9×** |
| c=64 TTFT p50 / p99 | 651 / 2,022 ms | **843** / 1,673 ms | **+30%** / −17% |
| c=64 TPOT p50 / p99 | 52 / 58 ms | 23 / 28 ms | −56% |
| c=64 E2E p99 | 8,632 ms | 4,436 ms | −49% |
| Requests completed, preemptions | 100%, 0 | 100%, 0 | |

## Graph usage and fallbacks
From the server's Prometheus counters, sampled every 0.5 s during each point (growth from the first to the last
sample, so a few steps at the edges are missed):

| point | engine steps | decode steps replayed | eager decode fallbacks | not eligible (prefill/mixed steps) |
|---|---|---|---|---|
| c=16 | 451 | 439 (97%) | 0 | 12 |
| c=64 | 488 | 451 (92%) | 0 | 37 |

Every pure decode step replayed a graph; no step fell back for batch or width (the `dli_cuda_graph_fallbacks` series
never appeared). The steps that ran eagerly are prefill and mixed prefill+decode steps, which graphs do not cover by
design.

## VRAM
- Capture at start-up: both buckets captured in 0.30 s with 3.9 GiB free next to the 19.7 GB KV pool; **84 MiB
  reserved** by capture (static buffers + shared pool), 122 MiB less free device memory at driver level.
- Whole-device peak (NVML) during the runs: 21.78 vs 21.65 GB at c=16 and 21.91 vs 21.78 GB at c=64, i.e.
  **+0.13 GB (~126 MiB) with graphs**, consistent with the capture figures.

## Reading
- Graphs remove most of the per-token cost: TPOT drops by 56–68% and throughput rises 2.9× at c=16 and 1.9× at c=64.
- **TTFT p50 at c=64 got worse (651 → 843 ms)** while p99 improved. Not investigated in this step. A likely cause:
  in a closed loop, faster decoding returns requests sooner, so new requests (and their 256-token prefills) arrive at
  almost twice the rate (8.8 → 16.9 req/s), while prefill and mixed steps run eagerly and are unchanged. The
  single repetition cannot separate this from noise.
- The HTTP gain at c=64 (2,166 tok/s) is far below the direct-engine decode rate at batch 64 (6,381 tok/s,
  `2026-11-11-GPU-A5000x1-runpod-m6-cuda-graph-buckets/`). The difference is the non-graphed prefill/mixed steps
  (37 steps, but each much longer than a decode step) plus HTTP and streaming overhead; not broken down here.

## Limits
- One repetition per point; no open-loop, shared-prefix or multi-turn runs; no vLLM comparison in this A/B.
- Contexts over 24 blocks (open-lognormal, shared-prefix workloads) would run eagerly; not exercised here.
- The graphs are opt-in (`--cuda-graphs`); the default server is unchanged.
