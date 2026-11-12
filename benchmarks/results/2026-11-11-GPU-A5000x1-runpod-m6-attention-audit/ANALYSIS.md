# M6 audit: GQA-aware paged decode attention (current path vs query reshaping vs Triton kernel)

`[GPU:A5000x1@runpod]`, revision `dbb8be0` + two untracked audit scripts (`benchmarks/scripts/decode_kernel_breakdown.py`,
`benchmarks/scripts/decode_attention_bench.py`), torch 2.14.1+cu130, Triton 3.8.0, 2026-11-11 23:35–23:37 UTC (~1 min).
Qwen2.5-0.5B shapes: 24 layers, 14 query heads, 2 KV heads (group 7), head_dim 64, block 16, bf16.

## 1. Where a decode step's GPU time goes (engine, 20 profiled steps)

| | B=64, ctx ~305, graphs | B=16, ctx ~302, graphs | B=64, ctx ~1,085, eager |
|---|---|---|---|
| step (unprofiled) | 9.02 ms | 5.55 ms | 22.47 ms |
| GPU time per step | 8.38 ms | 5.23 ms | 15.98 ms |
| **attention (mem-efficient SDPA)** | **3.06 ms (36%)** | 0.88 ms (17%) | **8.62 ms (54%)** |
| **KV page gather** | **0.85 ms (10%)** | 0.17 ms | **2.59 ms (16%)** |
| GEMMs | 2.50 ms | 2.44 ms | 2.58 ms |
| norm / RoPE / elementwise (~145 kernels) | 1.73 ms | 1.55 ms | 1.99 ms |
| KV write, sampling, other | 0.24 ms | 0.18 ms | 0.21 ms |

The long-context step is also launch-bound (22.5 ms wall for 16 ms of GPU work): contexts over 24 blocks fit no graph.

## 2. Attention alone, three implementations (24 layers in one CUDA graph, GPU ms per decode step)

Same paged pools and scattered block tables for every variant; lengths 85–100% of the max context; widths as the
engine pads them (24 blocks = the graph bucket; 72 / 136 = eager widths). All variants match a float64 reference
within bf16 rounding (max abs error 4.5e-4 – 1.4e-3, the same as the current path; one Triton case 5.8e-4 vs 4.5e-4).

| batch, max ctx (mean) | current | reshape | Triton paged (best split) | Triton vs current | effective KV bandwidth, Triton |
|---|---|---|---|---|---|
| 64, 383 (354) | 4.38 | 1.99 | **0.52** (1 split) | **8.4×** | 535 GB/s |
| 16, 383 (351) | 1.23 | 0.94 | **0.22** (4 splits) | 5.5× | 310 GB/s |
| 64, 1,150 (1,057) | 12.41 | 5.13 | **1.36** (1) | 9.1× | 611 GB/s |
| 16, 1,150 (1,068) | 3.56 | 2.50 | **0.46** (4) | 7.8× | 457 GB/s |
| 64, 2,175 (2,012) | 23.30 | 9.23 | **2.47** (1) | 9.4× | 640 GB/s |
| 16, 2,175 (2,004) | 6.60 | 4.46 | **0.74** (4) | 8.9× | 534 GB/s |

`current` moves the live KV at 56–68 GB/s (A5000 peak 768 GB/s): the gather copies every padded page, then the
mem-efficient kernel re-reads each KV head for each of its 7 query heads. `reshape` (queries viewed as
`[B, kv_heads, 7, d]`) removes the 7× re-read (2.2–2.5× at B=64) but keeps the gather (0.97 ms of its 1.99 ms at
B=64/383) and the padding. The Triton kernel reads each live K/V row once, straight from the pool through the block
table, for the whole query group (535–640 GB/s at B=64). Split-K helps only when B × kv_heads underfills the 64
SMs (B=16: 4 splits best; B=64: 1 split best).

## 3. Projected end-to-end effect (estimates, not measurements)
HTTP closed-fixed at c=64, engine process (`…-m6-http-profile-proc/`): of ~7.15 s, decode steps take ~4.7 s
(500 × 9.35 ms), mixed steps ~1.9 s (36 × 54 ms), the engine is idle ~5%.
- **Triton**: decode step −3.4 ms (attention+gather 3.9 → ~0.5 ms) → ~5.6 ms; mixed steps −~3 ms each.
  Wall ~7.15 → ~5.3 s: **~1.25–1.35× at c=64 (≈5,900–6,400 tok/s from 4,756)**. c=16: −0.83 ms per 5.55 ms
  step, **~1.1–1.15×**. The API loop would carry ~35% more tokens/s; it was ~60% idle at 4,600 tok/s.
- **Reshape**: decode step −2.4 ms → **~1.15–1.2× at c=64**, ~1.05× at c=16.
- **Long contexts** (open-lognormal, shared prefix): the measured attention saving is 11 ms of 16 ms GPU per step at
  B=64/1k, but the step stays launch-bound (~22 ms) until graphs cover wider tables. Without a gather, a wide graph
  bucket no longer pays for padding, so Triton + wider buckets would take that step to roughly 5–6 ms (~4×, estimate).

After attention, the GEMMs (2.5 ms) and ~145 small norm/RoPE/elementwise kernels (1.7 ms) dominate the B=64 step.
