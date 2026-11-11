# M6 HTTP serving overhead: what was tried, what was kept, combined result

`[GPU:A5000x1@runpod]`, Qwen2.5-0.5B-Instruct bf16, `--cuda-graphs`, closed-fixed (input 256, output 128, exact),
plan `benchmarks/plans/m6-a5000-serving.json` (block 16, 100,416 KV blocks, `max_num_seqs` 128,
`max_num_batched_tokens` 2048). **Three repetitions per point, server restarted per repetition**; tables show the
median and [min–max] of output tok/s. 2026-11-11, 22:27–23:09 UTC. Single runs of this configuration earlier in the
day spread by ±20%, so only differences larger than the repetition ranges are treated as real.

Starting point: HTTP reached 31–38% less than the engine in-process (known issue 38). The c=64 profile
(`…-m6-http-profile/`) attributed the loss to the engine thread and the asyncio loop sharing one GIL: per-token
loop wake-ups (2.4 s of a 12.6 s run) and steps slowed while waiting for the GIL (mixed 119 vs 52 ms).

## Changes, in order

| Step | Change | Revision | c=16 tok/s | c=64 tok/s | Verdict |
|---|---|---|---|---|---|
| 0 | baseline (`main`) | `23e2449` | 1,819 [1,719–1,910] | 2,550 [2,350–2,814] | — |
| A | one event-loop wake-up per step, not per token | `bbff347` | 1,786 [1,749–1,990] | **3,639 [3,531–3,670]** | **kept**: +43% at c=64 (ranges disjoint), neutral at c=16 |
| B | engine in its own process (same session as its threaded control) | `416424c` | **2,234 [2,165–2,314]** vs 1,857 | **4,727 [4,524–4,929]** vs 3,887 | **kept**: +20% / +22%, ranges disjoint |
| C | GIL switch interval 0.5 ms (on B) | `9dcb396` | 2,254 vs 2,297 | 4,633 vs 4,652 | **reverted** (`006d4e7`): no gain |
| E1 | httptools HTTP parser (on B) | `9dcb396` | 2,258 vs 2,297 | 4,768 vs 4,652 | **reverted** (`68f56d8`): within the range |
| E2 | httptools + uvloop (on B) | `9dcb396` | 2,236 vs 2,297 | 4,650 vs 4,652 | **reverted** (`68f56d8`): no gain |
| D | per-token work in the API process | — | — | — | **not implemented**: no headroom (below) |
| — | free-threaded Python | — | — | — | **not applicable** (below) |

Per-step result directories: `…-m6-serving-baseline/`, `…-m6-serving-A-batched-delivery/`,
`…-m6-serving-B-engine-process/`, `…-m6-serving-CE-stack-variants/` (the C/E runs, all on top of B in one session).

## Final configuration (this directory, revision `488dde2`, same session)

| closed-fixed | `--engine thread` (+A) | `--engine process` (+A, the default now) | change |
|---|---|---|---|
| **c=64 output tok/s** | 4,115 [3,983–4,148] | **4,756 [4,541–5,011]** | +16% |
| c=64 TTFT p50 / p99 | 435 / 796 ms | 233 / 599 ms | −46% / −25% |
| c=64 TPOT p50 · E2E p99 | 12 ms · 2,256 ms | 11 ms · 1,903 ms | |
| **c=16 output tok/s** | 2,016 [1,887–2,023] | **2,268 [2,068–2,346]** | +12% |
| c=16 TTFT p50 / p99 | 160 / 264 ms | 118 / 187 ms | −26% / −29% |

**Combined against the baseline** (different sessions, so approximate: the threaded+A configuration itself measured
3,639, 3,887 and 4,115 tok/s at c=64 in three sessions):
- **c=64: 2,550 → 4,756 tok/s (≈1.9×)**; TTFT p50 419 → 233 ms.
- **c=16: 1,819 → 2,268 tok/s (≈1.25×)**; TTFT p50 184 → 118 ms.
- Against the engine in-process (`…-m6-mixed-step/`, 5,086 and 2,485 tok/s): HTTP now reaches **≈94% at c=64 and
  ≈91% at c=16** (was 62% and 69%).

## Why D and free-threading were not pursued
Profile of the engine-process server at c=64 (`…-m6-http-profile-proc/`): the engine process is **busy 94–95% of
the wall time**, with 0.5 ms between steps; its decode (9.35 ms) and mixed (54 ms) steps match in-process (9.41 /
52 ms). The API process's event loop spends most of its time idle (62% of stack samples in `select`; the reader
thread used 0.26 s of CPU). Cutting per-token work in the API process cannot raise throughput by more than the
engine's ~5% idle time, which is inside the repetition spread; it was therefore not implemented. Free-threaded
Python would remove a GIL that no hot path shares any more (the engine process is single-threaded, the API loop is
mostly idle) and would mean leaving the pinned Python 3.12 build; not applicable.

## Correctness (revision `488dde2`)
- CPU: 366 tests, including the whole HTTP API and metrics/admin e2e suites run a second time with the engine in a
  child process, engine crash / stop / SIGTERM drain with no orphaned engine process, and the drain test with both
  engines.
- GPU (`-m gpu`): 15 passed, including the engine process with CUDA graphs matching the threaded server.
- Real checkpoints over HTTP, fp32 on CUDA, `--cuda-graphs`, engine process (`…-m6-proc-http-parity-float32/`):
  **24/24 identical texts** to the unbatched reference for Qwen2.5-0.5B and SmolLM2-135M, streamed = non-streamed,
  finish reasons and usage correct, ~99 graph replays each.

## Limits
Closed-fixed only (no open-loop, long-context or multi-turn runs); one model; 3 repetitions per point.
