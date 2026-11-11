# M5 bf16 precision study: what the M3/M4 bf16 failures mean

`[GPU:A16x1@vultr]`, revision `8e790c1` (clean), pre-registered in `docs/evaluation-plan-m5.md` and
`benchmarks/methodology.md`. Raw data: `Qwen2.5-0.5B-Instruct.json`, `SmolLM2-135M-Instruct.json`.

The M3 (engine vs unbatched, bf16) and M4 (streamed vs non-streamed over HTTP, bf16) exact-match criteria **remain
recorded as failed**. This study replaces "are the tokens identical?" with questions that have meaningful answers in
bf16, using teacher forcing so every configuration is fed the same tokens.

| Configuration | |
|---|---|
| A | fp32, unbatched (reference) |
| B | bf16, unbatched: precision alone |
| C | bf16, continuous-batching engine in the M3 configuration (48 sequences, 256-token budget, pool at 25% of demand, **47 preemptions**): precision + batching |
| D | fp32, same engine configuration: control |

## 1. The engine is exact; bf16 is what changes tokens

| 100 prompts × 128 steps (12,800 steps) | Qwen2.5-0.5B | SmolLM2-135M |
|---|---|---|
| D (fp32 engine) top-1 agreement with A | **100%**, KL ≤ 1e-6, max logit error ≤ 6e-5 | **100%**, same |
| B (bf16 unbatched) top-1 agreement | 98.57% | 98.46% |
| C (bf16 engine) top-1 agreement | 98.64% | 98.53% |
| Per-logit error of B / C, median (p99) | 0.24 (0.84) / 0.25 (0.92) | 0.34 (0.75) / 0.34 (0.77) |
| Largest reference top-2 gap at a disagreement, B / C | 0.39 / 0.19 | 0.28 / 0.28 |

The fp32 engine reproduces the reference at every step through chunked prefill, batching and 47 preemptions, so
scheduling and paged KV add no error. In bf16, each logit is off by about a quarter of a unit (median) relative to
fp32, so argmax flips wherever the two best candidates are closer than that: 1.4–1.5% of steps, all at reference
gaps ≤ 0.39 (within the M1 bf16 rule, gap ≤ 1.0). Free-running decoding turns one such flip into a different
continuation, which is all the M3 and M4 exact-match failures showed.

## 2. Batching adds no material error
Pre-registered comparison: per-prompt mean KL(A‖C) vs KL(A‖B).

| | Qwen2.5-0.5B | SmolLM2-135M |
|---|---|---|
| mean KL, B / C | 6.98e-4 / 7.46e-4 | 7.05e-4 / 7.14e-4 |
| C − B, mean [95% bootstrap CI] | +4.8e-5 [−2.6e-5, +1.4e-4] | +8.1e-6 [−1.1e-5, +2.6e-5] |
| Wilcoxon signed-rank p | 0.005 | 0.28 |

For Qwen the paired test detects a shift in the per-prompt medians, but its size is 7% of bf16's own divergence and
the CI of the mean includes zero; C's top-1 agreement is slightly *higher* than B's. Batching changes which bf16 roundings
happen, not how much error there is.

## 3. Output quality is unchanged within the pre-registered tolerances

| | A fp32 | B bf16 | C bf16 engine | Tolerance | Verdict |
|---|---|---|---|---|---|
| WikiText-103 perplexity, Qwen2.5-0.5B (51,200 scored tokens) | 14.785 | 14.793 | 14.797 | ratio within ±1% | C/A = 1.0009 [1.0005, 1.0012] ✔ |
| WikiText-103 perplexity, SmolLM2-135M | 18.339 | 18.351 | 18.348 | ±1% | C/A = 1.0005 [1.0000, 1.0009] ✔ |
| ARC-Easy accuracy, Qwen2.5-0.5B (all 2,376 items) | 65.57% | 65.74% | 65.61% | ±2 points | C−A = +0.04 [−0.38, +0.46] pts, McNemar p = 1.0 ✔ |

bf16 raises perplexity by 0.05–0.09% (measurable, with CIs excluding 1.0 for C/A, i.e. real but tiny) and does not
change ARC accuracy (11 vs 10 discordant items between A and C). Both CIs lie inside the pre-registered tolerances,
so "no quality loss within ±1% perplexity and ±2 accuracy points" is supported for these models and tasks.

## 4. Pre-registered target missed: margin-conditional agreement
Target: top-1 agreement of C ≥ 99.9% at steps whose reference top-2 gap exceeds one bf16 ulp `[TARGET]`.

| | Qwen2.5-0.5B | SmolLM2-135M |
|---|---|---|
| C (engine) | **99.845%** (19 of 12,259 steps disagree) | **99.828%** (21 of 12,220) |
| B (unbatched bf16) | 99.837% (20) | 99.804% (24) |

**Missed for both models**, and missed equally by B, so the cause is bf16 arithmetic, not the engine. The
investigation the plan requires: the target assumed per-logit error of about one ulp, but the measured error is
several ulps (median 0.24–0.34, p99 up to 0.92, versus an ulp of 0.0625–0.25 for top logits of magnitude 8–64),
accumulated across 24–30 layers of bf16 matrix products. Steps with gaps of 1–2 ulps can therefore still flip. The
target is kept as written and recorded as missed. In this data every disagreement has a reference gap ≤ 0.39, so
agreement is 100% above that, well inside the M1 rule (gap ≤ 1.0).

## Threats to validity
Teacher forcing measures per-step fidelity, not free-running behaviour (M3/M4 recorded that). Two small models; larger
models accumulate error over more layers. One GPU type and one kernel library version (pinned, `env.json`).
ARC log-likelihood scoring with a plain prompt format; absolute accuracy depends on the format, the A/B/C comparison
does not.
