# A5000 baseline for M6 (focused re-run of the M5 measurements)

`[GPU:A5000x1@runpod]` RunPod RTX A5000 (24 GB, sm_86), container quota 7.65 CPUs (7 threads), bf16, revision
`b07963d` (clean), `benchmarks/plans/m5-a5000-baseline.json`, 3 rotated repetitions, 2026-11-11 17:45–18:49 UTC.
Native KV pools equal vLLM's capacity (pilot attempt 2). vLLM 0.32.0 runs from the hash-pinned venv with
`VLLM_USE_FLASHINFER_SAMPLER=0` (the A16's sampler). Generated tables: `SUMMARY.md`. **Compare only with other
`[GPU:A5000x1@runpod]` results**; the A16 results stand on their own hardware.

## Closed loop (input 256 / output 128), output tok/s, median [min–max]

| Model | Concurrency | native engine | vLLM | HF static | vLLM ÷ native | native ÷ HF |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 1 | 43.7 | 389.1 | 51.1 | 8.9× | 0.86× |
| Qwen2.5-0.5B | 16 | 163.9 [151.9–178.5] | 3,493.4 | 559.5 | 21× | 0.29× |
| Qwen2.5-0.5B | 64 | 198.1 [188.3–202.9] | 7,544.3 | 736.2 | 38× | 0.27× |
| Qwen2.5-1.5B | 16 / 64 | 135.8 / 168.5 | 1,821.9 / 4,410.8 | 389.3 / 589.7 | 13× / 26× | 0.35× / 0.29× |
| SmolLM2-1.7B | 16 / 64 | 198.1 / 252.9 | 1,572.5 / 2,990.2 | – | 7.9× / 12× | – |

Single-stream TPOT: native 22 ms, vLLM 2 ms.

## Open loop, Qwen2.5-0.5B (log-normal lengths)
The native engine is stable to ~2–4 req/s (TPOT 40 → 410 ms across 0.5 → 4 req/s); vLLM holds TTFT ≤ 46 ms and
TPOT ≤ 4 ms up to 16 req/s and saturates near 64 req/s.

## What the faster GPU changed, and what it did not
vLLM gains 2.7–2.9× over the A16 (Qwen2.5-0.5B c=16: 1,279 → 3,493; c=64: 2,566 → 7,544 tok/s). HF static batching
does not gain (c=16: 677 → 560; c=64: 755 → 736), consistent with its Python generate loop also being CPU-bound on
this pod. The native engine got *slower*
(c=16: 224.6 → 163.9 tok/s; per-sequence cost 3.15 → 4.75 ms per step): its decode is launch-bound
(13,497 kernels per step at batch 64, GPU ~18% busy, `../2026-11-11-GPU-A5000x1-runpod-m5-engine-profile/`), so it
runs at the speed of the pod's CPU, not the GPU. This is the "before" for M6.

## Notes
- vLLM's closed-loop c=64 TPOT p50 is 5 ms, but its client lag P99 reached 167 ms in the open-loop 64 req/s point: at
  that rate the load generator (sharing 7.65 CPUs with the server) starts to fall behind; that point's latencies are
  upper bounds.
- Qwen2.5-7B-AWQ, shared-prefix, multi-turn, AC5 and the precision study were not re-run (A16 results stand).
