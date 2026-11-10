# M5 evaluation plan: numerical fidelity and output quality of reduced-precision serving

Status: **plan** (written during M4, executed in M5). It responds to the M3 bf16 result, which stays recorded as a
**failed** criterion (`benchmarks/results/2026-11-10-GPU-A16x1-vultr-m3-engine-parity-bfloat16/`). Nothing here
redefines that criterion as passed. It replaces an exact-match check that cannot be meaningful for bf16 with measurements
that can.

## 1. What M3 established
- **fp32:** the batched engine is token-identical to the unbatched path (100/100 × 2 models, CPU and A16, with 47
  preemptions each). Scheduling, batching, chunked prefill, and recompute are exact.
- **bf16:** 22/100 and 18/100 prompts identical over 128 greedy tokens. Every first divergence occurs at a reference
  top-2 logit gap ≤ 0.375 (median 0.0 and 0.0625).

## 2. Numerical explanation
bf16 keeps 8 significant bits, so logits near magnitude 32 are spaced 0.25 apart. The engine computes the same math
with different matrix shapes (many sequences per matmul, chunked prefill), and GPU GEMM kernels pick different tiling
and accumulation orders for different shapes. Results therefore differ in the last bf16 bits. Where two candidate tokens
are within that rounding distance, argmax can flip. After one flip, greedy decoding conditions on a different token, so
the sequences diverge for the rest of the generation. That is error *propagation*, not error *accumulation*: a single
rounding-level flip changes everything after it. Token identity under different batch shapes is therefore expected to
fail in bf16 even for a correct engine, and it says nothing about whether output quality changed.

## 3. Questions M5 must answer
1. **Per-step fidelity:** how far are bf16 engine logits from an fp32 reference *at the same context*?
2. **Batching effect:** is the bf16 engine any less accurate than bf16 unbatched decoding (does batching add error beyond bf16 itself)?
3. **Task quality:** does bf16 serving change task-level outcomes, and by how much, with uncertainty?

## 4. Methods
Teacher forcing removes divergence propagation: every configuration is fed the **same** token sequence (the fp32
reference's greedy output), so each step compares like with like.

| Configuration | Purpose |
|---|---|
| A: fp32 unbatched (reference) | Ground truth |
| B: bf16 unbatched | Precision effect alone |
| C: bf16 batched engine (same batch mix as production-like load) | Precision + batching |
| D: fp32 batched engine | Control: must equal A (re-verifies M3) |

**Metrics** (per step, teacher-forced, on the 100 M1 prompts × 128 tokens, both models):
- Top-1 agreement with A (fraction of steps whose argmax equals A's).
- KL(A ‖ X) of next-token distributions; max absolute logit error.
- Agreement **conditional on margin**: top-1 agreement for steps where A's top-2 gap > 1 bf16 ulp. Expected ≈ 100%; any
  disagreement there would point to an error beyond rounding.
- **Pre-registered hypothesis for question 2:** the distributions of KL(A‖B) and KL(A‖C) are equivalent. Tested with a
  paired comparison over steps (Wilcoxon signed-rank on per-prompt mean KL) and reported as an effect size with a
  bootstrap 95% CI, not only a p-value.

**Task quality** (questions 3):
- Held-out perplexity on a fixed public text sample (e.g. a WikiText-103 test subset) for A, B, C: paired bootstrap CI of the
  perplexity ratio.
- Task accuracy where the models are capable enough to be informative (Qwen2.5-0.5B-Instruct on a fixed, seeded
  subset of a public benchmark such as GSM8K or ARC-Easy, greedy, exact-match scoring): accuracy for A, B, C with
  paired comparison (McNemar's exact test on per-item correctness) and the difference with a 95% CI.
  The sample size is fixed in advance from a power estimate. If the CI is too wide to conclude, that is reported.
- SmolLM2-135M is too weak for most tasks: only fidelity metrics and perplexity are reported for it.

## 5. Acceptance (fixed before M5 runs)
- D == A token-identical (fp32 control), else engine regression.
- Margin-conditional top-1 agreement of C ≥ 99.9% at steps with gap > 1 ulp `[TARGET]`. Lower values trigger
  investigation, not reinterpretation.
- Batching effect: the 95% CI of mean KL difference (C − B) is reported. A material increase is a finding to explain.
- Task metrics: differences reported with CIs. No claim of "no quality loss" unless the CI excludes the stated tolerance
  (perplexity ratio within ±1%; accuracy difference within ±2 points).

## 6. Hardware tiers and cost
CPU (`[CPU-LAPTOP]`) for fp32 references and SmolLM2 fidelity. A16 (`[GPU:A16x1@vultr]`) for bf16 configurations.
Estimated GPU time is 30–60 min, so it runs as an approved benchmark campaign (Gate B). Results go to
`benchmarks/results/<date>-<label>-m5-precision-eval/` with env.json and the commit captured at start.

## 7. Threats to validity
Teacher forcing measures per-step fidelity, not free-running behavior (both are reported). Small models may
under-represent effects in larger ones. A fixed benchmark subset can be unrepresentative (report the subset seed and size).
GPU kernel choice can vary with driver/library versions (pinned via `uv.lock` and recorded in env.json).
