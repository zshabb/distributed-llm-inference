# M6 CUDA graph proof of concept: one shape (batch 16 x 16 blocks)

`[GPU:A5000x1@runpod]`, Qwen2.5-0.5B bf16, revision `00ba311` (clean), 2026-11-11 19:31 UTC (43 s of A5000 time).
`benchmarks/scripts/engine_profile.py --batches 16 --input-len 96` (contexts 96–176 tokens, i.e. ≤ 11 blocks of 16,
inside the captured width): engine driven directly, warm-up past prefill, then 20 **unprofiled** steady-state decode
steps (timed) and 20 profiled steps. One run per configuration, as approved (a proof of concept, not a campaign).

**Question:** can a CUDA graph remove the ~23 ms fixed decode step cost left after batched attention, without
correctness or memory problems? **Answer for this shape: yes.**

## Result

| Batch 16, steady-state decode | eager batched (M6 step 1) | CUDA graph | change |
|---|---|---|---|
| Step time, unprofiled | 22.76 ms | **5.20 ms** | **4.4× faster** |
| Output tok/s | 703 | **3,078** | 4.4× |
| Same, next to the full 19.7 GB KV pool (100,416 blocks) | – | 5.27 ms / 3,037 tok/s | graph captured |
| Kernels per step (profiler) | 1,204 | 1,205 | same kernels, launched by one replay |
| Replays / fallbacks | – | 79 / 0 | every decode step after capture |
| Capture time (lazy, first eligible step) | – | 0.23–0.25 s | |

The graph replays the same ~1,200 kernels; what disappears is the CPU cost of launching them one by one. The remaining
~5.2 ms is about 3 ms of model forward (profiled) plus host work outside the graph: metadata construction, copying it
into the static buffers, sampling and scheduler bookkeeping.

## Memory
- One eager forward at this shape needs 4.7 MiB of activations; the capture gate required 2× that plus 256 MiB free.
- **Free device memory before capture: 22,695 MiB** with the small test pool and **3,955 MiB next to the full
  19.7 GB pool**: ample room in both cases (the gate would have disabled the graph below ~265 MiB).
- Overhead: total reserved memory 1,168 MiB with the graph vs 1,156 MiB eager (**+12 MiB**); peak allocated
  1,085 vs 1,103 MiB. The scratch block adds one KV block (12 KiB per layer here).
- **Measurement flaw found:** the recorded `graph_pool_mib` (reserved-memory delta across capture) came out as
  −8 MiB. The caching allocator rearranges reserved segments during capture, so that delta does not measure the
  graph's pool; the eager-vs-graph totals above are the meaningful comparison. To be fixed when graph coverage expands
  (measure with `torch.cuda.graph_pool_handle()` statistics or a dedicated pool).

## Correctness (A5000, `-m gpu`, 7 tests, 6.6 s)
- Replay is **bitwise equal** to the eager forward on the same padded inputs.
- Full engine runs with the graph vs eager batched (fp32, tiny GQA model): identical tokens and every logit within
  1e-4 over many steps with changing inputs, batches shrinking as requests finish (padding rows), and a tight pool
  with preemption.
- A padded replay (5 real rows, 11 padding rows) changes only the 5 written slots and the scratch block in every layer:
  no real KV block is corrupted.
- Unsupported shapes (more than 16 sequences, or a context over 16 blocks) and insufficient free memory fall back to
  eager with identical outputs.
- CPU suite (CI): graphs disable themselves on CPU; the scratch block is never allocated under pool pressure.

## Limits of this proof of concept
- One shape only: batch ≤ 16 and context ≤ 16 blocks (256 tokens at block 16). The M5/M6 benchmark workloads
  (256-token prompts + 128 output) exceed the width and would run eagerly; batch 64 is not captured.
- Opt-in (`LLMEngine.for_model(cuda_graph=(16, 16))`); the server does not expose it yet.
- Prefill and mixed steps are not graphed.
- HTTP end-to-end throughput was not measured.
