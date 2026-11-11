# M6 step 1: batched decode attention vs the per-sequence path

`[GPU:A5000x1@runpod]`, Qwen2.5-0.5B bf16, revision `870d809` (clean), 2026-11-11 18:51–18:59 UTC.
`benchmarks/scripts/engine_profile.py --attention {reference,batched}`: the engine driven directly (no HTTP), batch of
B requests (input 256, output 128), warm-up past all prefills, then 20 **unprofiled** decode steps (timed) and 20
profiled steps (kernel counts, operator times). Raw: `Qwen2.5-0.5B-Instruct-{reference,batched}.json`.
One run per configuration: a short A/B as approved, not a repeated campaign.

## Result

| Steady-state decode | reference (per-sequence) | batched (M6) | change |
|---|---|---|---|
| B=16: step time, unprofiled | 87.4 ms | 24.2 ms | **3.6× faster** |
| B=16: output tok/s | 183 | 662 | 3.6× |
| B=64: step time, unprofiled | 379.7 ms | 22.8 ms | **16.7× faster** |
| B=64: output tok/s | 169 | 2,808 | 16.7× |
| Kernels per step, B=16 / B=64 | 4,113 / 13,497 | 1,204 / 1,276 | −71% / **−90.5%** |
| Attention calls per step (B=64) | 1,536 | 24 (one per layer) | |
| KV write, gather, GQA-expansion launches (B=64) | 3,072 + 3,074 + 3,072 | 48 + 48 + 0 | |

- **Step time no longer grows with the batch** (24.2 ms at B=16, 22.8 ms at B=64; one run each, so the 1.4 ms
  difference is within noise). That is the ~22 ms batch-independent cost the audit predicted from the reference path's
  linear fit: the per-sequence attention orchestration (4.75 ms per sequence) is gone.
- What remains is fixed per step: ~1,200 kernels for the dense layers, norms, RoPE and the 24 batched attention
  calls. GPU time is now small (attention 3.0 ms, GEMMs ~5 ms per step at B=64), so the step is still CPU/launch-bound,
  now at a constant ~1,200 launches instead of 8 per sequence per layer. That is the target of CUDA graphs (next).
- Attention now runs PyTorch's memory-efficient kernel (`fmha_cutlassF`, chosen because of the padding mask) instead of
  per-sequence flash attention; its device time is 3.0 ms per step at B=64 (reference: 17.8 ms of flash-attention
  kernels).

## Numerics
- **fp32**: CPU tests compare both paths step by step on the same requests (GQA, MHA, Qwen2 biases with chunked
  prefill, preemption, shared prefixes): identical tokens, every logit within 1e-4 (`tests/unit/model/
  test_batched_decode.py`). On the A5000 over HTTP (`../2026-11-11-GPU-A5000x1-runpod-m6-http-parity-float32/`):
  8/8 prompts identical to the unbatched reference for Qwen2.5-0.5B and SmolLM2-135M, streamed == non-streamed 8/8.
- **bf16** (`../...-m6-http-parity-bfloat16/`): 5/8 and 5/8 identical to the fp32-path reference (the reference path
  had 7/8 and 6/8 in the validation run). Every divergence is at a reference top-2 gap ≤ 0.125, i.e. a bf16 near-tie,
  as characterized in the M5 precision study: the batched path uses a different attention kernel and accumulation
  order, which moves rounding. Not a correctness regression; quality-level impact would need the M5 precision study
  re-run on this path (not done here).

## Regressions and limits found
- None in the test suites (294 passing, including all end-to-end server tests on the batched path).
- **Padding cost**: attention and the page gather cover every sequence up to the longest one in the batch. With very
  uneven lengths (one long context among short ones) the work and the transient gather memory scale with
  B × max length; for an MHA model at long context (SmolLM2-1.7B: 196 KB of KV per token) the per-layer transient can
  reach gigabytes at B=128, 4k tokens. Not measured here (the benchmark's lengths are uniform). The Triton
  paged-attention kernel removes the gather and the padding.
- Only all-decode steps are batched; steps with prefill chunks still take the per-sequence path (not profiled here).
- This A/B is engine-level steady-state decode. HTTP end-to-end throughput (prefill, detokenization, streaming) was
  not re-measured, so these numbers are not comparable with the closed-loop HTTP tables of the A5000 baseline.
