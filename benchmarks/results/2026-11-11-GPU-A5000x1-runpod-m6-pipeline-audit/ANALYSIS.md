# M6 pipeline audit: where closed-loop serving time goes after decode CUDA graphs

`[GPU:A5000x1@runpod]`, Qwen2.5-0.5B-Instruct bf16, revision `7fd2671` + the then-untracked
`benchmarks/scripts/pipeline_profile.py` (env.json records `-dirty`), 2026-11-11 21:46:19–21:48:18 UTC (2.0 min).
One run per configuration; an audit to rank optimizations, not a benchmark.

## 1. Closed loop in-process, by step kind (`pipeline_profile.py`)
closed-fixed without HTTP: input 256, output 128, `concurrency` requests in flight, `4 x concurrency` requests,
block 16, 100,416 KV blocks, `max_num_seqs` 128, `max_num_batched_tokens` 2048.

| | c=16 graphs | c=64 graphs | c=16 eager | c=64 eager |
|---|---|---|---|---|
| Output tok/s | 2,255 | 2,781 | 849 | 1,983 |
| Decode-only steps: share of wall, mean | 75%, 5.5 ms | 38%, 9.3 ms | 91%, 17.5 ms | 53%, 18.2 ms |
| **Mixed steps** (prefill + decode rows): share, mean | 23%, 76 ms | **61%, 206 ms** | 8%, 72 ms | 46%, 219 ms |
| Prefill-only step (2,048 tokens, 8 prompts) | 47 ms | 53 ms | 46 ms | 45 ms |

A mixed step at c=64 (~1,800 prefill tokens + ~55 decode rows) takes 215–231 ms, against 53 ms for a pure
2,048-token prefill: every decode row of a mixed step ran the per-sequence reference path. Over HTTP the same
configurations reached 1,765 (c=16) and 2,166 tok/s (c=64) with graphs (`…-m6-cuda-graph-http-ab/`), i.e. **22% below**
in-process at both concurrencies (HTTP/async overhead; cause not measured).

## 2. Long-context decode, eager (`engine_profile.py`, input 1,024 / 2,048)

| | B=16, ctx ~1,100 | B=64, ctx ~1,090 | B=16, ctx ~2,120 |
|---|---|---|---|
| Step, unprofiled | 15.9 ms | 22.0 ms | 19.6 ms |
| Output tok/s | 1,007 | 2,903 | 815 |
| Attention kernel (mem-efficient SDPA) per step | 2.44 ms | 8.64 ms | 4.59 ms |
| K/V page gather per step | 0.62 ms | 2.59 ms | 1.23 ms |

Contexts over 24 blocks fit no graph bucket and run eagerly. At B=64 and ~1,090 tokens the attention kernel moves
the step's 855 MB of KV at an effective ~99 GB/s (A5000 peak 768 GB/s): consistent with each KV head being read once
per query head (7x for Qwen2.5-0.5B's 14/2 heads) on top of the gather copy (hypothesis, not measured directly).

## Ranking that followed (reported to the owner)
1. Batched decode rows inside mixed steps (measured: 61% of c=64 wall) — approved and implemented next.
2. GQA-aware paged decode attention (measured: attention + gather ≈ 50% of decode GPU time at ~330 tokens, ~75% at 1k).
3. HTTP/async overhead (measured: −22%; cause hypothesized).
4. Graph buckets for longer contexts (measured: long-context decode still launch-bound).
5. CPU/GPU overlap across decode steps (estimated 1–2 ms exposed host time per step).
