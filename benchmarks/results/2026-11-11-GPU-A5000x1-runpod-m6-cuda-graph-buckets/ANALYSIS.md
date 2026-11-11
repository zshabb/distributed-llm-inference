# M6 decode CUDA graph buckets: batch 16/64 x 24 blocks at benchmark context lengths

`[GPU:A5000x1@runpod]`, Qwen2.5-0.5B bf16, revision `11421fb` (clean, synced with `SYNC_FROM=head`),
2026-11-11 20:17:39–20:18:41 UTC (62 s of A5000 time). Direct engine, no HTTP:
`benchmarks/scripts/engine_profile.py` with its defaults: input 256 tokens, output 128, block 16, warm-up past
prefill, then 20 **unprofiled** decode steps (timed) and 20 profiled ones. The decode contexts measured here are
~330 tokens (21 blocks), inside the closed-fixed benchmark's range (256–383 tokens, ≤ 24 blocks). One run per
configuration, as approved: a short comparison, not a campaign.

Buckets (`BENCHMARK_SHAPES`): **(16 rows, 24 blocks)** and **(64 rows, 24 blocks)**, captured together into one
shared memory pool; a step replays the smallest bucket it fits into, or runs eagerly.

## Result

| Steady-state decode, contexts ~330 tokens | eager batched | CUDA graphs | change |
|---|---|---|---|
| **Batch 64**: step time, unprofiled | 23.09 ms | **10.03 ms** | **2.3× faster** |
| Batch 64: output tok/s | 2,772 | **6,381** | 2.3× |
| Batch 64, next to the full 19.7 GB KV pool (100,416 blocks) | – | 9.58 ms / 6,679 tok/s | captured (3,985 MiB free) |
| Batch 16: step time, unprofiled | 20.43 ms | **5.53 ms** | 3.7× faster |
| Batch 16: output tok/s | 783 | **2,891** | 3.7× |
| Decode steps replayed / eager fallbacks | – | 71 / 0 (B=64), 77 / 0 (B=16) | every decode step after capture |
| Capture, both buckets (first decode step) | – | 0.25–0.33 s | |

Graph usage: at batch 64 every step replayed the (64, 24) graph; at batch 16 every step replayed (16, 24). No
step fell back (counts cover warm-up, timed and profiled decode steps). The tests cover a batch shrinking from the
64-row into the 16-row bucket; this run does not, because all requests advance together.

**Where the batch-64 step goes now.** The replay is asynchronous, so the profiled "forward" (2.4 ms) is only
the host side; the GPU work surfaces in "postprocess" (8.3 ms), where sampling waits for the logits. The step is
now bounded by GPU execution (~8 ms for 1,277 kernels at B=64), no longer by launch overhead. That is the expected
next limit; reducing it (fused kernels, less gather padding) is out of scope for this step.

Not comparable with the proof of concept's 5.20 ms at B=16: that run used shorter contexts (96–176 tokens) and a
narrower graph (16 blocks).

## Memory
- Eager activation peak per forward: 4.7 MiB (16 x 24) and 18.7 MiB (64 x 24); the memory gate needs
  2× that + 256 MiB free. Free before capture: ~22.3–22.7 GiB with the default pool, **3.9 GiB next to the full
  19.7 GB pool**: both buckets captured in every configuration.
- **Graph memory overhead: 84 MiB** reserved (static buffers + the shared private pool, both buckets), measured
  after emptying the allocator cache, so it no longer comes out negative as in the proof of concept. Driver-level
  free memory dropped by 84–122 MiB (adds non-PyTorch allocations such as cuBLAS workspaces for the captured
  stream). The pool holds both graphs' output logits (64 x 151,936 bf16 = 18.5 MiB for the large one) and
  allocator segment rounding, not only activations.
- Total reserved memory, eager vs graphs: 1,550 vs 1,558 MiB (B=64), 1,190 vs 1,208 MiB (B=16).
- The scratch block adds one KV block (12 KiB per layer).

## Correctness (A5000, `-m gpu`, 13 tests, 7.7 s; CPU suite 309 passed)
- Replay is **bitwise equal** to the eager forward on the same padded inputs, in both buckets.
- Alternating buckets on the shared pool (64, 16, 64) leaves the large graph's results bitwise unchanged.
- Full engine runs with graphs vs eager batched (fp32, tiny GQA model, both buckets at tiny scale): identical tokens
  and every logit within 1e-4, with batches crossing from the 64-row into the 16-row bucket and with preemption.
- Padded replays in either bucket change only the written slots and the scratch block: no real KV is corrupted.
- More rows than the largest bucket and contexts wider than the bucket fall back to eager with identical outputs;
  a bucket that fails its memory gate is skipped while the other keeps serving; no free memory disables both.

## Limits
- Covers the closed-fixed benchmark (≤ 64 sequences, ≤ 24 blocks). The open-lognormal workload (inputs up to
  2,048 tokens) and shared-prefix workloads (1,024-token prefixes) exceed 24 blocks and run eagerly.
- Opt-in (`LLMEngine.for_model(cuda_graphs=BENCHMARK_SHAPES)`); the server does not expose it yet, so the HTTP
  benchmark numbers are unchanged. HTTP end-to-end throughput was not measured.
- Prefill and mixed steps are not graphed.
