# M6: Triton paged GQA decode attention vs `main`, over HTTP

`[GPU:A5000x1@runpod]`, Qwen2.5-0.5B-Instruct bf16, plan `benchmarks/plans/m6-a5000-paged-attention.json`: the server
as shipped (engine process, `--cuda-graphs`, default decode attention), block 16, 100,416 KV blocks, `max_num_seqs`
128, `max_num_batched_tokens` 2048, closed loop. Two revisions in separate checkouts on the same pod:
- `main` = `dbb8be0` (SDPA decode attention, graph buckets 16/64 x 24 blocks);
- `branch` = `782d0b3` (Triton paged kernel via `auto`; graph buckets 16/64 x 24 and 16/64 x 160 blocks).

Three repetitions per revision, **alternating main and branch within one session**
(2026-11-11 23:50:01–23:56:12 UTC, ~10 min): `main/rep{0,1,2}/`, `branch/rep{0,1,2}/`, each a full campaign with
the server restarted. `ab_summary.json` from `benchmarks/scripts/ab_summary.py <this dir> main branch`.

## Throughput (output tok/s, median [min–max] of 3)

| workload | c | main | branch | speedup |
|---|---|---|---|---|
| closed-fixed, 256 in / 128 out | 16 | 2,235 [2,198–2,261] | **2,672 [2,551–2,699]** | **1.20×** |
| closed-fixed | 64 | 4,955 [4,855–4,981] | **6,245 [5,944–6,627]** | **1.26×** |
| closed-1k, 1,024 in / 256 out | 16 | 918 [915–934] | **2,323 [2,100–2,450]** | **2.53×** |
| closed-1k | 64 | 2,547 [2,536–2,553] | **5,053 [4,330–5,098]** | **1.98×** |
| closed-2k, 2,048 in / 128 out | 16 | 700 [585–753] | **1,402 [1,288–1,437]** | **2.00×** |
| closed-2k | 64 | 1,277 [1,257–1,294] | **2,294 [2,027–2,321]** | **1.80×** |

Every branch range lies entirely above the main range. The closed-fixed gain at c=64 (1.26×) is at the low end
of the audit's 1.25–1.35× estimate; c=16 (1.20×) is above its 1.1–1.15×.

## Latency (medians of the per-run percentiles) and graph use

| workload | c | TTFT p50 main → branch | TTFT p99 | TPOT p50 | E2E p99 | graph replays / step | eager decode fallbacks |
|---|---|---|---|---|---|---|---|
| closed-fixed | 16 | 126 → 116 ms | 210 → 292 ms | 6.2 → 4.9 ms | 975 → 819 ms | 0.97 → 0.97 | 0 → 0 |
| closed-fixed | 64 | 249 → 263 ms | 613 → 663 ms | 10.8 → 7.4 ms | 1,868 → 1,514 ms | 0.93 → 0.92 | 0 → 0 |
| closed-1k | 16 | 178 → 242 ms | 331 → 377 ms | 16.7 → 5.8 ms | 4,567 → 1,803 ms | 0 → 0.97 | 1,462 → 0 |
| closed-1k | 64 | 481 → 490 ms | 1,425 → 1,328 ms | 22.8 → 10.5 ms | 7,208 → 3,968 ms | 0 → 0.89 | 1,396 → 0 |
| closed-2k | 16 | 350 → 474 ms | 560 → 983 ms | 20.7 → 7.4 ms | 3,259 → 1,706 ms | 0 → 0.89 | 691 → 0 |
| closed-2k | 64 | 1,200 → 1,043 ms | 2,326 → 1,977 ms | 39.4 → 18.9 ms | 7,149 → 4,222 ms | 0 → 0.64 | 636 → 0 |

- Long contexts gain most: on `main` every decode step over 24 blocks ran eagerly (all the fallbacks above) and
  attended over padded, gathered pages; on the branch they replay the 160-block graphs and read only live keys.
- **TTFT rose at c=16 on the long-context workloads** (1k: p50 178 → 242 ms; 2k: p50 350 → 474 ms, p99 560 →
  983 ms) and closed-fixed p99 at c=16/64 rose 210 → 292 / 613 → 663 ms. In a closed loop, faster decoding returns
  requests sooner, so new 1–2k-token prompts arrive more often (req/s 3.6 → 9.1 and 5.5 → 11.0 at c=16) and wait
  for prefill, which this change does not speed up. E2E p99 fell everywhere (by 17–61%).
- Whole-device GPU memory peak: 20.43 → 20.47 GiB (the two 160-block graphs).

## Correctness (revision `782d0b3`)
- GPU (`-m gpu`, A5000): 41 passed — kernel vs float64 reference over GQA 7:1/3:1, MHA, head_dim 128 and 80, block
  sizes 4/16, fp32/bf16, every split factor, with out-of-range table entries poisoned; CUDA-graph replay bitwise
  equal to eager; Triton vs SDPA engines (decode, mixed, graphs incl. 160-block buckets, preemption): identical
  tokens, logits within 1e-4.
- Real checkpoints over HTTP, engine process + graphs + Triton:
  - fp32 (`…-m6-triton-http-parity-float32/`): **24/24 identical texts** for Qwen2.5-0.5B and SmolLM2-135M,
    streamed = non-streamed, finish reasons and usage correct.
  - bf16 (`…-m6-triton-http-parity-bfloat16/`), against the same test on `main` with SDPA
    (`2026-11-12-…-m6-sdpa-http-parity-bfloat16/`, run minutes later; dated by UTC midnight): identical texts
    **12/24 vs 10/24** (Qwen) and **16/24 vs 13/24** (SmolLM2); every divergence in both is at a near-tie (top-2
    logit gap ≤ 0.125). The Triton kernel accumulates in fp32, which is at least as close to the reference.

## Limits
One model (Qwen2.5-0.5B, 14/2 heads); closed-loop workloads only (no open-loop or multi-turn runs); 3 repetitions.
Prefill and the dense layers are unchanged.
