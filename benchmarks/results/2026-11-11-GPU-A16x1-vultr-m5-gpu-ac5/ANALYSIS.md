# AC5 rerun: overload behaviour at ≥ 2× saturation

`[GPU:A16x1@vultr]`, Qwen2.5-0.5B bf16, native engine only, revision `3b98d95` (clean),
`benchmarks/plans/m5-gpu-ac5.json`, 3 repetitions, 2026-11-11 15:12–15:49 UTC (37 min of A16 time).
Why a rerun: the campaign's overload test ran at ~0.5–1× of true saturation (biased saturation rule, fixed in
`3685485`) with an admission predictor that over-predicted 10–50× (fixed in `939e134`); see
`../2026-11-11-GPU-A16x1-vultr-m5-gpu/ANALYSIS.md` §5. Criteria unchanged from the pre-registration.

## Saturation (open loop, log-normal lengths, corrected rule)

| Offered req/s | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| Output tok/s (median of 3) | 169 | 213 | 243 | 255 | 265 |
| TTFT p50 / p99 | 114 / 295 ms | 173 / 432 ms | 327 / 800 ms | 465 / 1388 ms | 1052 ms / 44.7 s |
| TPOT p50 | 47 ms | 94 ms | 205 ms | 268 ms | 404 ms |

The queue-growth rule flagged 6 req/s, so 4 was the highest "unsaturated" point and the overload test offered
**8 req/s**. Throughput, however, plateaus from ~2 req/s (213 → 265 tok/s while the offered load triples), and per-token
latency rises 4×: with up to 128 concurrent sequences, continuous batching absorbs overload into slower decode
rather than into a waiting queue, which a TTFT-trend rule sees late. By the throughput criterion saturation is about
**2 req/s**, so the test ran at **~4× saturation**, which satisfies, conservatively, AC5's "2× saturation" condition. The
saturation rule's blind spot is recorded as a known issue (a throughput-plateau or TPOT-growth criterion is the fix).

## AC5 at 8 req/s

| | with `X-Deadline-Ms: 2000` | control, no deadline |
|---|---|---|
| Completed | 49.7% (157–182 of 320 per rep) | 100% |
| Output tok/s | 257 [256–260] | 274 [274–276] |
| TTFT p50 / **p99** (completed) | 749 ms / **1,981 ms** | 18.4 s / 90.1 s |
| Rejected up front (`deadline_unattainable`) | 97–108 per rep, answered in **2–16 ms** | – |
| Admitted, then expired while queued (`deadline_exceeded`) | 41–56 per rep, answered after **2.00–2.03 s** | – |
| Rejection latency p99 (all rejections) | **2,011 ms** | – |
| Crash / OOM / hang | none (peak 13.67 GiB) | none |

| Criterion (pre-registered) | Verdict |
|---|---|
| No crash or OOM at ≥ 2× saturation | ✔ |
| Admitted requests' TTFT P99 ≤ 2 s `[TARGET]` | ✔ 1.98 s, against 90 s without admission control |
| Rejections answered in < 50 ms | **✘** P99 2.01 s: up-front rejections take ≤ 16 ms, but 30–34% of rejected requests were admitted and then expired in the queue |

**AC5: failed on rejection latency**, met on stability and on the admitted-TTFT SLO.

## What this shows
- Deadline-aware admission does what the spec's hypothesis O2 predicts: at ~4× saturation it keeps admitted requests'
  TTFT under the 2 s SLO (P99 1.98 s), where the same load without it queues for up to 90 s. Throughput is almost
  unchanged (257 vs 274 tok/s): the engine does the same amount of work, for requests that can still meet their SLO.
- The step-model predictor is now on the optimistic side under heavy overload: for some requests it predicted
  < 2 s but the first token would have come later (13–18% of requests). The step time it multiplies by is an average of recent steps, and
  each admission makes the next steps slower (TPOT rose 4× across the sweep); the deadline backstop then rejects
  those requests after 2 s, which is correct for the client but too slow for the < 50 ms criterion.
- Next fixes, not done in M5: account for admitted-but-not-started requests' effect on step time (marginal cost per
  running sequence, which is roughly linear on this engine, §1 of the campaign analysis), and a safety margin
  calibrated from measured prediction error. Both are measurable with this plan.

## Related
- Prediction accuracy of the M4 predictor (over-prediction): campaign `ANALYSIS.md` §5.
- Why step time grows with the batch: `../2026-11-11-GPU-A16x1-vultr-m5-engine-profile/`.
