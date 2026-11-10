# Continuous batching scheduler: chunked prefill, preemption, fairness

Implementation: `src/dli/engine/` (`scheduler.py`, `engine.py`, `runner.py`, `request.py`), `src/dli/sampling.py`,
batched attention in `src/dli/model/llama.py` (`BatchedKV` / `Segment`).
Tests: `tests/unit/engine/`, `tests/unit/model/test_engine_tiny.py`, `tests/unit/model/test_batched_forward.py`,
`tests/unit/test_sampling.py`, `tests/parity/test_engine_parity.py`.

## 1. Problem
Static batching waits for a whole batch of requests and then runs it until the *longest* finishes. Short requests
wait for long ones, GPU slots sit idle as requests finish, and new arrivals wait for the next batch. Serving needs to
change batch membership **every step** while bounding per-step latency, staying within KV memory, and keeping
outputs exactly what each request would get alone.

## 2. Why it matters
Batching is what turns a memory-bound decode step (one token per sequence) into useful work: the weights are
read once per step for every sequence in the batch. Throughput scales with batch size until compute or KV memory
saturates. Scheduling policy decides who waits, so it sets tail latency and fairness between tenants.

## 3. Existing solutions
Orca (iteration-level scheduling, OSDI 2022); vLLM (continuous batching over PagedAttention, recompute/swap
preemption, chunked prefill); Sarathi-Serve (chunked prefill with stall-free batching, OSDI 2024); TGI.

## 4. Their limitations (for this project's goals)
Production schedulers interleave many concerns (LoRA, spec decode, multi-modal), which makes policy behavior hard to
isolate and test. Fairness across tenants is usually absent or external. Goal here: a small scheduler whose policies
(budgeted chunking, LIFO recompute preemption, priority + aging + weighted fair queuing) are individually tested,
including by fault injection.

## 5. Design
Per step, under a token budget and a sequence cap:
1. Sort running requests by (effective priority, arrival). Serve each one its next decode token or next prompt chunk.
   If KV memory is short, preempt the newest running request not yet in this batch (LIFO). If none exists, the
   requester yields itself.
2. If nothing was preempted, admit waiting requests by key (effective priority, resumed-first, tenant virtual
   time, arrival), chunking the prompt to the remaining budget. Stop at the first one that does not fit in memory.
3. The runner flattens all chunks into one forward pass (`BatchedKV`). Dense layers run once, and attention runs
   per segment against that sequence's paged KV. Logits are computed only for rows that sample.
4. Sample per row with per-request generators. Apply stop/EOS/length rules. Finished requests free their KV.

Admission control by **capacity**: a request whose prompt + `max_tokens` exceeds the whole pool is rejected
immediately (`CAPACITY`). Deadline-aware admission (`X-Deadline-Ms`) and queue bounds belong to the API (M4).

## 6. Alternatives considered
| Alternative | Why not (now) |
|---|---|
| Prefill-first (admit before decoding) | New prompts stall running decodes, so inter-token latency spikes under load |
| Swap preemption (KV to host memory) | Pays off only when PCIe bandwidth beats recompute FLOPs; needs GPU to measure. Recompute is simpler and leverages the prefix cache. Deferred to GPU work |
| FIFO admission | One tenant's backlog starves others (shown by fault injection: FIFO fails the share tests) |
| Strict priority without aging | Batch work can starve forever (the no-aging mutant fails the starvation test) |
| Heap-ordered waiting queue | The key includes tenant virtual time, which changes whenever a tenant is served, so the heap would be constantly invalidated. Linear `min` is O(W) per admission (see §8) |

## 7. Implementation details
- `num_computed_tokens` on the request mirrors the block manager (the fake runner asserts every chunk starts exactly
  there). After preemption it resets to 0, and readmission sets it to the cached-prefix length.
- KV slots for newly sampled tokens are reserved at schedule time (`append_slots`), and copy-on-write ops are applied by
  the runner before the forward pass.
- No admission in a step that preempted (prevents preempt/readmit thrashing).
- A newly active tenant starts at the current virtual clock (smallest virtual time among active tenants), so idle
  time cannot be banked.

## 8. Complexity and tradeoffs
| Part | Cost |
|---|---|
| Scheduling a step | O(R log R) sort of running + O(R) reservation, plus O(W) per admitted request (`min` over waiting) |
| Admission burst of k requests from W waiting | O(k·W). Measured negligible at W = 512 (0.04 s total); would matter at ~10⁴ queued (known issue) |
| Batched forward | Dense layers once per step; attention per segment (Python loop, the reference path until the M6 kernel) |
| Recompute preemption | Wastes the victim's compute, partly recovered by prefix-cache hits on resume |

## 9. Correctness testing
- **Deterministic history-hash fake model** (`tests/unit/engine/fake.py`): the next token hashes the entire history,
  so dropped, duplicated or reordered tokens, or lost progress across preemption, change outputs. Every test compares
  against the same rule run per request in isolation.
- Unit tests for continuous batching, chunking, prefix reuse, stop reasons, capacity, abort (`test_continuous_batching.py`);
  preemption (`test_preemption.py`); fairness (`test_fairness.py`).
- **Randomized adversarial scenarios** (`test_scheduler_properties.py`): Hypothesis-generated workloads with hostile
  limits. Per-step invariants plus liveness, exact outputs, and zero leaks. Focused profile: 2,000 scenarios, passed.
- **Fault injection:** FIFO admission, no aging, credit banking, dropping preempted requests, over-admission, and ignoring
  the cached prefix on readmission are each detected. A non-reset inside `_preempt` is an equivalent mutant.
- **Real tensors:** tiny Llama/Qwen2 engine outputs equal the unbatched path under roomy, heavily chunked, and
  preempting configurations; seeded sampling is identical alone and in a busy batch.
- **Real checkpoints (milestone):** see §10 and `benchmarks/results/*-m3-engine-parity-*`.

## 10. Performance measurements
- **Engine overhead** (null runner, no model work): median 28–37 µs per sequence per decode step, flat from 8 to 256
  concurrent sequences `[CPU-LAPTOP]`, code `380a1fa`. An unrecorded dry run measured ~14 µs, so absolute values vary ~2×
  between runs on this laptop; only the flat shape is claimed. Largest component: argmax over `[batch, vocab]` logits
  (cProfile in the result directory).
- **Batching gain** (SmolLM2-135M, 24 prompts × 32 tokens, interleaved ×3): 107.2 vs 20.8 output tok/s (5.2×); engine
  overhead 0.58% of step time `[CPU-LAPTOP]`.
- **Correctness at scale:** 100 concurrent prompts × 128 tokens with 47 preemptions and chunked prefill: 100/100 identical
  to the unbatched path for both models in fp32, on CPU and on the A16. In bf16 most outputs diverge at bf16 near-ties
  (pre-set criterion failed, analysis in the result directory).
- **Not measured yet:** TTFT/ITL percentiles under open-loop load, GPU throughput, comparison with vLLM (M5).

## 11. Remaining limitations
Single-threaded engine loop (the async API arrives in M4); no swap preemption; no deadline-aware admission yet (M4);
O(W) admission scan; requests admitted in the same step cannot share a prefix (blocks become shareable only once
computed); attention is a per-segment Python loop until the M6 kernel; stop strings need detokenization (M4).
