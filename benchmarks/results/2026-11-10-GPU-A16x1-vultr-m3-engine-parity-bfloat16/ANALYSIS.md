# Analysis: bf16 engine parity FAILED its pre-set criterion

Raw results in this directory are unedited. Code `6132814c0417` (synced from HEAD), `[GPU:A16x1@vultr]`, bf16, both
the engine and the unbatched reference running on the same GPU. KV pool at 25% of demand (47 preemptions per model),
256-token step budget.

## What failed
The pre-set bf16 criterion (copied from M1) required ≥ 98% of prompts identical over the first 64 tokens.
Observed: SmolLM2-135M 32/100 (22/100 over all 128 tokens), Qwen2.5-0.5B 18/100 over 128 tokens. **Criterion failed.**

## What the data shows
| | SmolLM2-135M | Qwen2.5-0.5B |
|---|---|---|
| Reference top-2 logit gap at first divergence: min / median / max | 0.0 / **0.0** / 0.3125 | 0.0 / 0.0625 / 0.375 |
| Divergences at gap > 1.0 (the M1 "beyond bf16 rounding" bound) | **0** | **0** |
| Divergent prompts that had been preempted | 34 / 78 | 37 / 82 |

- Every divergence happens at a near-tie within bf16 resolution (0.25 at logit magnitude 32). In SmolLM2 most
  divergences are at **exact** bf16 ties (gap 0.0).
- Divergence is not concentrated in preempted requests, so preemption-recompute is not implicated.

## Interpretation (hypothesis, tested separately)
The engine batches many sequences into one matmul. bf16 GEMMs with different shapes use different kernels and
accumulation orders, so logits move at bf16 resolution relative to the unbatched path. Near-ties flip, and greedy decoding
then diverges for the rest of the sequence. M1's criterion held because M1 compared two *unbatched* implementations with
the same shapes. Applying it to batched-vs-unbatched bf16 was an error in my acceptance rule, not evidence of correctness.

**Decisive check:** the same engine in **fp32** on the same GPU (`*-m3-engine-parity-float32` on the A16) and on CPU.
If scheduling, batching, or preemption were wrong, fp32 would also diverge, at large gaps.

## Result of the decisive check (added after the fp32 run)
Same engine, same configuration, same GPU, **fp32** (code `5b4254c`, results in
`2026-11-10-GPU-A16x1-vultr-m3-engine-parity-float32/`): **100/100 prompts token-identical over all 128 tokens for both
models**, with the same 47 preemptions per model. Scheduling, batching, chunked prefill, and preemption-recompute are
therefore exact. The bf16 divergence is a property of bf16 arithmetic under different batch shapes, not an engine defect.

**Consequence for acceptance:** batched-vs-unbatched token identity is a valid criterion only in fp32. For bf16 the meaningful
checks are (a) no divergence at a gap beyond bf16 rounding (met: max 0.375 ≤ 1.0) and (b) model-level quality, which
belongs to evaluation (M5), not to exact-match parity. The failed 98% rule stays recorded as failed.
