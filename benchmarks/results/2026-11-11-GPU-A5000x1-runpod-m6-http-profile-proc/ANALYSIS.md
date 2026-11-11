# HTTP serving profile at c=64, engine in its own process

`[GPU:A5000x1@runpod]`, Qwen2.5-0.5B bf16, `--cuda-graphs --engine-process` (now `--engine process`), closed-fixed
256/128 at c=64, one repetition per variant, revision `9dcb396` + the profiling hook (env.json `-dirty`),
2026-11-11 22:57-22:58 UTC; plan `benchmarks/plans/m6-a5000-http-profile-c64.json`.

| variant | tok/s | engine process busy | gap between steps | decode / mixed step |
|---|---|---|---|---|
| instrument | 4,582 | 94% | 0.50 ms mean | 9.35 / 53.9 ms |
| + 2 ms stack sampler (API process only) | 4,427 | 95% | 0.50 ms mean | 9.45 / 63.4 ms |

The engine process now runs at in-process speed (9.41 / 52 ms) and is idle only ~5% of the time. The API
process's loop is mostly idle (62% of samples in `select`; reader thread 0.26 s CPU). The hook's
`engine_thread_calls` are empty here: it still looked for `_dispatch`, renamed `_deliver` by then (fixed in the
committed hook). Conclusions: `…-m6-serving-final/ANALYSIS.md`.
