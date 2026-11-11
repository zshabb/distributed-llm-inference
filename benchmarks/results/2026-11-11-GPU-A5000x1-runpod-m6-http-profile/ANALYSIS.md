# HTTP serving profile at c=64, threaded engine (input to the M6 serving work)

`[GPU:A5000x1@runpod]`, Qwen2.5-0.5B bf16, `--cuda-graphs`, closed-fixed 256/128 at c=64, one repetition per
variant, revision `23e2449` (env.json records `-dirty`: the hook was an untracked file then; it is now
`benchmarks/scripts/profiling/sitecustomize.py`), 2026-11-11 22:19:28-22:21:15 UTC. Plan: `plan.json` in each
`campaign/` directory. The engine ran on a thread of the API process (the only design at the time).

| variant | tok/s | engine busy in steps | decode step | mixed step | gaps between steps |
|---|---|---|---|---|---|
| instrument (step timing, thread CPU) | 2,595 | 78% | 11.5 ms | 119 ms | 2.57 s (20%) |
| + 2 ms stack sampler | 1,676 | 71% | 15.2 ms | 174 ms | 5.40 s |
| + GIL switch interval 0.5 ms | 3,287 | 86% | 10.2 ms | 94 ms | 1.26 s |

In-process (`…-m6-mixed-step/`): decode 9.41 ms, mixed 52 ms. The gaps were almost all per-token
`call_soon_threadsafe` wake-ups (32,896 calls, 74 us each); steps were slowed by waiting for the GIL (the switch
interval and the sampler moved throughput by tens of percent). The client was not limiting (36% of a core, 0 ms
loop lag) and batches were full (480 of 491 decode steps with >= 60 sequences). The hook itself costs throughput
(2,595 vs 3,171 tok/s uninstrumented), so shares are approximate. Follow-up and results:
`…-m6-serving-final/ANALYSIS.md`.
