# ADR-0004: Benchmarking through the HTTP API, open-loop by default, equal KV capacity

- Status: Accepted
- Date: 2026-11-11

## Context
M5 compares the native engine with vLLM and with Hugging Face `transformers` static batching. Serving benchmarks are
easy to make flattering without lying: measure in-process instead of over HTTP, drive load closed-loop so a slow server
slows its own client, give one side more KV memory, use a weak baseline, or pick the workload after seeing results.
These decisions remove those degrees of freedom before any result exists (the rules were committed before the runs:
`benchmarks/methodology.md`).

## Decisions

1. **Measure every backend through the same OpenAI-compatible HTTP API with the same client** (`dli.bench.loadgen`).
   Engine-internal numbers (StepStats) explain results but are never compared with another system's client-side
   numbers. The HF baseline gets a thin server (`dli.bench.hf_server`) so it can be driven identically.
2. **Open-loop Poisson load for latency, latency measured from the scheduled arrival.** A closed loop hides queueing:
   when the server slows, the client sends less, so latency percentiles look better exactly when the system is
   overloaded (coordinated omission). Closed loop is used only for maximum throughput at a stated concurrency.
3. **Equal KV-cache capacity in tokens, not an equal memory fraction.** vLLM's pool is sized by
   `--gpu-memory-utilization` and its capacity is read from its log; the native pool is set to the same number of
   16-token blocks. Engines differ in activation memory and CUDA-graph workspace, so equal fractions would give one
   side more cache, which matters most for the KV-bound model (SmolLM2-1.7B, no GQA).
4. **vLLM runs in its normal optimized configuration** (CUDA graphs, torch.compile, prefix caching), from the
   official image pinned by digest. Detuning a baseline to look closer is a form of fabrication. The native engine
   runs as it is (eager); the comparison is explicitly system-to-system.
5. **A correct static-batching baseline, tested.** Left-padded batched `generate` must reproduce unbatched greedy
   decoding token for token (test), so a broken baseline cannot inflate the native engine's advantage.
6. **Fixed lengths via token-id prompts and `ignore_eos`.** Every backend does exactly the same work, independent of
   tokenizers, chat templates, or when a model happens to emit EOS. The multi-turn workload is the exception by
   design (real text and templates) and is compared only between engines that run the same templates.
7. **Sweeps stop one point after saturation.** Points beyond saturation only measure the growth of a queue and spend
   GPU time; the saturation rule is pre-registered.
8. **Calibration is separate from evidence.** A one-repetition pilot sets KV pool sizes and rate grids; its results are
   committed with a "not evidence" note and never quoted as measurements.

## Consequences
- The client shares the GPU host's 6 vCPUs with the server; client lateness is recorded so client saturation would be
  visible rather than silently inflating latency.
- The HF baseline cannot stream, so it is compared on throughput and end-to-end latency, not TTFT.
- Results are system comparisons on one GA107 GPU, not kernel comparisons and never multi-GPU evidence.
