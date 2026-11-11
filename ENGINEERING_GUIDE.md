# Engineering Guide

How the system works, for an engineer who needs to maintain or extend it.
**Scope of this revision: milestones M0–M4** (model, loader, paged KV memory with prefix caching, continuous-batching
engine with chunked prefill, preemption, fairness, sampling, OpenAI-compatible async HTTP server, tooling).
Sections for kernels, tensor parallelism, and the router are added as those land.
Nothing described as *planned* exists yet (see `docs/implementation-plan.md`).

## 1. End-to-end flow today (M1)

```
token ids ──► greedy_generate (dli/generate.py)
                 │ allocates ContiguousKVCache(len(prompt)+max_new_tokens)
                 ▼
            LlamaForCausalLM.forward(input_ids[n], positions[n], cache, last_token_only)
                 │ embed → for each layer: RMSNorm → Attention(RoPE, GQA, cache write) → +res
                 │                         RMSNorm → SwiGLU MLP → +res
                 ▼ final RMSNorm → logits (LM head or tied embedding)
            argmax → append → next step feeds 1 token at position len-1
```

- **Prefill** is the first call: all prompt tokens at once, causal attention, writes K/V for every position.
- **Decode** calls feed one token. Attention reads the whole cached prefix, and no mask is needed because a single
  query may see everything before it.
- A **chunk appended to a non-empty cache** (n > 1, past > 0) needs an explicit mask, because SDPA's `is_causal` aligns
  to the top-left corner. This path is not used by greedy decoding today, but is tested
  (`test_chunked_prefill_into_nonempty_cache_matches_single_pass`) because chunked prefill (M3) will depend on it.

## 1a. Paged KV flow (M2)

```
paged_greedy_generate(model, BlockManager, PagedKVStore, seq_id, prompt)
  BlockManager.allocate(seq, prompt)        → block table; leading full blocks reused from the prefix cache
  PagedSequence.prefill(prompt)             → forward only prompt[num_cached_tokens:]
     model(..., PagedSequenceView)           write: K/V rows → store slots (slot_mapping)
                                             read:  gather the sequence's pages in position order
     view.advance(n) → mark_computed         completed full blocks become shareable
  loop: BlockManager.append_slots(seq,[t])  → CopyOps (copy-on-write if the last block is shared)
        PagedKVStore.copy_blocks(ops); forward 1 token through the view
  BlockManager.free(seq)                    → cached blocks stay CACHED-IDLE (revivable), others FREE
```

The block manager owns *metadata* (where KV lives, who shares it, what is cached). The store owns *bytes*. The model sees
only the `KVCacheView` protocol (`length`, `write`, `advance`), so contiguous and paged caches are interchangeable and
the attention math did not change. Design and invariants: `docs/algorithms/paged-kv-cache.md`. Eviction policies:
`docs/algorithms/prefix-cache-eviction.md`. Decisions: ADR-0003.

## 1b. Engine step (M3)

```
LLMEngine.step()
  Scheduler.schedule()                       budget B tokens, cap S sequences
    running, sorted (eff. priority, arrival):  next decode token or next prompt chunk
       reserve KV slot (append_slots) ── no memory? preempt newest unscheduled (LIFO, recompute)
    if nothing preempted: admit waiting by (eff. priority, resumed-first, tenant vtime, arrival)
       allocate prompt (prefix reuse) → chunk = min(remaining prompt, remaining budget)
  ModelRunner.execute()                      apply COW copies; flatten chunks → ONE forward
    model(ids, positions, BatchedKV(segments), logits_indices=rows that sample)
       dense layers once over all tokens; attention per segment over its PagedSequenceView
  sample(logits, params, per-request generators)
  append token → stop/EOS/length → Scheduler.finish() frees KV
  StepStats: schedule / forward / postprocess seconds
```

Invariants: a chunk always starts at the request's `num_computed_tokens` (the block manager's count). The oldest
running request is never preempted. Every admitted request fits alone (capacity rejection), so the engine always
makes progress. Policy rationale and complexity: `docs/algorithms/continuous-batching-scheduler.md`.

## 1c. HTTP request (M4)

```
uvicorn (asyncio)                                                      engine thread (AsyncEngine)
 BodySizeLimit (413) → AuthMiddleware (401, Argon2 in a worker thread)
 route: pydantic + unsupported() (400) → tokenize / sandboxed chat template → length checks
 AdmissionController.check(snapshot) → 503 draining | 429 busy/queue/deadline + Retry-After
 _text_stream ─ generate(rid, ids, params) ──── _Add ──────────────► queue → step() loop
      ◄────────── EngineEvent(token_ids, finished) via call_soon_threadsafe ◄──┘
   _before(deadline) → _guard_dead (503) → IncrementalDetokenizer → StopStringMatcher
 _first_event (under _unless_disconnected; X-Deadline-Ms → 429 if no first token in time)
   stream=false: _collect (under _unless_disconnected; 504 on timeout) → JSON + usage + x_timing
   stream=true : _sse → role chunk, deltas, final chunk, [usage chunk], [DONE]; errors in-band
 any exit path closes the generator → finally: _Abort ─────────────► scheduler.abort → KV freed
```

Rules that keep this correct:
- **The engine thread is the only code that touches the engine.** Handlers communicate by command queue and read the
  immutable `EngineSnapshot`; there are no locks around engine state.
- **One exit path.** Generators are always wrapped in `aclosing()`, so disconnect, stop string, timeout, deadline,
  error, and shutdown all reach the same `finally` that aborts the request.
- **Time limits are per await, not `asyncio.timeout()`,** because generators are started in one task and iterated in
  another (QnA.md, question 45).
- **Nothing is sent before the first token,** so pre-token failures are real HTTP statuses (QnA.md, question 44).

## 2. Components

| Component | File | Responsibility | Key invariants |
|---|---|---|---|
| Config | `src/dli/model/config.py` | Parse HF `config.json` (Hub and transformers-v5 formats) into a frozen dataclass | Unsupported features raise `UnsupportedModelError` and are never ignored |
| RoPE | `src/dli/model/rope.py` | Inverse frequencies (default/linear/llama3), cos/sin tables, rotation | Angles computed in fp32; rotate-half layout |
| Model | `src/dli/model/llama.py` | Decoder graph; parameter names mirror checkpoints | Layers never see requests; positions are explicit |
| KV cache (M1) | `src/dli/model/kv_cache.py` | Contiguous per-sequence K/V | Writes must be contiguous and within `max_len`; `length` advances only after all layers wrote |
| Loader | `src/dli/model/loader.py` | safetensors → meta-built model via `assign=True` | No pickle; exact name/shape match; tied-head consistency |
| Hub | `src/dli/model/hub.py` | Download at pinned commit hashes | Only ungated Apache-2.0 models; `*.bin` never downloaded |
| Generation | `src/dli/generate.py` | Greedy loop, cached or recompute | Lowest-index argmax tie-break, same as the reference |
| Env manifest | `src/dli/envinfo.py` | Hardware/software manifest per result | Never raises on missing tools; revision captured at run start; untracked files count as dirty |
| Block hashing | `src/dli/kv/hashing.py` | SHA-256 chain identity of full blocks | Equal hash ⇒ equal prefix (same namespace) |
| Block manager | `src/dli/kv/block_manager.py` | Allocation, ref counts, COW, prefix index, eviction driving | `check_invariants()`; failures raise before mutating; thread-confined |
| Eviction policies | `src/dli/kv/eviction.py` | LRU and cost-aware GDSF over leaf candidates | Only unused leaves are candidates |
| Paged store | `src/dli/kv/paged_store.py` | KV pool tensors, slot scatter, page gather, COW copies | Layout `[layers, blocks, B, kv_heads, head_dim]` |
| Paged generation | `src/dli/generate.py` (`PagedSequence`, `paged_greedy_generate`) | One sequence over paged KV | Equal to the contiguous path |
| Batched attention | `src/dli/model/llama.py`, `kv_cache.py` (`BatchedKV`, `Segment`) | Many sequences in one forward pass | Segments tile the batch; each attends only to its own KV |
| Sampler | `src/dli/sampling.py` | Greedy, temperature, top-k, top-p; per-request generators | One draw per sampled token per request (batch-invariant) |
| Scheduler | `src/dli/engine/scheduler.py` | Budgeted chunking, LIFO recompute preemption, priority + aging + weighted fair queuing, capacity rejection, abort | Budget/cap never exceeded; chunks continue computed KV; no admission in a preempting step |
| Runner / engine | `src/dli/engine/runner.py`, `engine.py` | One forward per step; step loop, stop rules, StepStats | Thread-confined single loop |
| Async engine | `src/dli/server/async_engine.py` | Engine on its own thread; add/abort/stop commands; per-request event queues; `EngineSnapshot` | Only the engine thread touches the engine; a crash marks it unhealthy and fails every open request |
| HTTP app | `src/dli/server/app.py`, `schemas.py` | Routes, validation, termination, SSE, timeouts, disconnect handling | Every request path aborts its engine request on exit; in-flight counter always released |
| Text | `src/dli/server/text.py` | Tokenizer, sandboxed chat template, incremental detokenizer, stop strings | Never emits a partial UTF-8 character or a partial stop string |
| Admission | `src/dli/server/admission.py` | In-flight/queue caps, deadline feasibility, draining | Decisions use only the published snapshot |
| Auth / limits | `src/dli/server/auth.py`, `limits.py` | API keys (Argon2id), body-size limit | Both run before the body is parsed |
| Metrics / admin | `src/dli/server/metrics.py`, `admin.py` | Prometheus registry, admin listener (state, drain) | Bounded label cardinality; separate port |
| Entry point | `src/dli/server/__main__.py`, `factory.py` | CLI (`dli-serve`), safe defaults, graceful drain on SIGTERM | Refuses unauthenticated public binds |

## 3. Data structures and memory

- **Parameters** are loaded straight into their final dtype/device. Building on `meta` avoids allocating and randomly
  initializing a throwaway copy, so peak host memory is about one copy of the checkpoint (plus the safetensors mmap).
- **KV cache (M1):** `2 × layers × kv_heads × max_len × head_dim × bytes`. For Qwen2.5-0.5B in fp32
  (24 layers, 2 KV heads, head_dim 64) that is 24,576 bytes per token; at 4096 tokens, 96 MiB per sequence,
  reserved up front whether used or not. That waste, multiplied by concurrent sequences, is exactly what the
  paged block manager (M2) removes.
- **GQA expansion:** K/V are expanded with `repeat_interleave` before SDPA. That costs memory bandwidth proportional
  to the fan-out (3× for SmolLM2's 9 query / 3 KV heads; 7× for Qwen2.5-0.5B's 14 / 2). The paged kernel (M6)
  will index KV heads directly instead of materializing the expansion.

## 4. Numerical design decisions

| Decision | Why |
|---|---|
| RMSNorm in fp32, weight multiply after cast back | bf16 sum-of-squares over hundreds of channels loses precision; identical cast placement is required for bit-level parity |
| RoPE angles in fp32 | Angle error in bf16 grows with position; rotation tables are cast only at the end |
| Keys cached post-rotation | Absolute positions never change, so a cached key never needs re-rotation (required for prefix sharing in M2) |
| Same SDPA entry point as the reference | Keeps fp32 results within float reordering noise of `transformers` |
| Explicit `positions` argument | Continuous batching packs tokens at different positions into one vector |

## 5. Verification strategy

1. **Fast tiny-model parity** (`tests/unit/model/test_reference_parity.py`, ~6 s): random 3-layer models from
   `transformers` saved as real safetensors. Covers 7 variants (untied/tied, Llama biases, llama3 and linear RoPE, MHA,
   Qwen2), positions past the original context, chunked prefill, cache vs recompute, EOS handling.
2. **Real-checkpoint parity** (`tests/parity/`, `-m parity`): 100 prompts × 128 greedy tokens on two pinned models.
   Results land in `benchmarks/results/<date>-<label>-m1-greedy-parity-*` with `env.json`. Current results: 100/100
   identical for both models in fp32 `[CPU-LAPTOP]` and bf16 `[GPU:A16x1@vultr]` (see README *Verified results*).
   Long runs capture the code revision at start (`envinfo.current_revision`) because HEAD can move during a 70-minute run.
3. **Rejection paths:** every loader and config error has a test that triggers it.
4. **Server (M4), all over real TCP** (`tests/e2e/`): a uvicorn server backed by the real engine and a tiny random
   model with a character tokenizer, so every layer runs in milliseconds. Covers streaming, disconnects (stream,
   non-stream, queued), overload and deadlines, timeouts, malformed input, auth, body limits, metrics, admin, the real
   `dli-serve` process (refusals and SIGTERM drain), contract C1 schema conformance of every response, and the official
   `openai` SDK. On real checkpoints (`tests/parity/test_http_real_checkpoint.py`): prompt ids equal `transformers`'
   for 100 prompts, and concurrent HTTP output equals `transformers`' decode of the unbatched reference tokens.

Why exact token equality is meaningful in fp32: both implementations perform the same operations in the same
order, so per-step logit differences are ~1e-6, far below typical top-2 gaps. A token flip in fp32 would point to a
real bug or a pathological tie, and the report records the reference's top-2 gap at any divergence to tell the two apart.

## 6. Technology choices (M0–M1)

See `docs/design-decisions/`: ADR-0001 (Python/PyTorch engine + Go router, layering) and ADR-0002 (torch 2.14.1 pinned
for CPU and CUDA 13.0 builds in one lockfile, matching the A16 host driver).

## 7. Debugging procedures

| Symptom | Likely cause | How to check |
|---|---|---|
| Fluent but wrong text | RoPE layout or theta/scaling mismatch | `test_positions_beyond_original_context_match`; compare `inverse_frequencies` against the reference |
| Garbage from the first token | Wrong GQA grouping or tied/untied head | Logit parity on a tiny variant with the same head counts |
| Cached output ≠ recompute | Cache `length` advanced early, or wrong mask branch | `test_kv_cache_path_equals_recompute_path`; check `past` in `Attention.forward` |
| `CheckpointError: missing tensors` | Architecture flag not mapped (e.g. biases) | Compare `model.state_dict().keys()` with `safetensors` keys |
| GPU numbers look throttled | Unlicensed vGPU or another tenant's load | `env.json` → `gpu.details["License Status"]`; `nvidia-smi -q` |
| Paged output ≠ contiguous | Shared block written (missing COW), hit on uncomputed block, slot mapping off-by-one | `check_invariants()`; run the stateful test with `HYPOTHESIS_PROFILE=focused`; `scripts/dev/mutation_check_block_manager.sh` confirms the tests still catch these faults |
| `NoFreeBlocksError` with apparently free memory | Watermark reserve at admission, or idle blocks pinned by in-use children | `stats()`; `check_invariants()` asserts idle blocks are evictable (liveness) |
| Blocks leak over time | A sequence never freed (scheduler bug) | `stats().used_blocks` vs live sequences; stress soak pattern in `tests/stress/` |
| Batched output ≠ unbatched | Chunk not continuing computed KV, wrong segment offsets, preemption losing progress | Fake history-hash runner (asserts chunk continuity); `test_batched_forward.py`; engine parity harness records the first divergence and the reference logit gap |
| A request never finishes | Livelock or starvation | `has_unfinished()` loops with a step bound in tests; aging_steps; check that the request is not over capacity |
| Throughput low despite many requests | Budget/cap too small, memory-bound preemption churn | `StepStats` (tokens per step, preemptions); `engine.bm.stats()` |
| Many 429s | In-flight/queue caps, or deadlines tighter than the predicted TTFT | `dli_admission_rejections_total{reason}`; `/admin/state` (queue, prefill rate) |
| KV usage does not return to 0 when idle | A request path that does not close its generator | `dli_kv_blocks_used`; `/admin/state`; the disconnect tests in `tests/e2e/test_http_api.py` show the pattern |
| Streams cut with an in-band error | Request timeout, or engine failure (`dli_engine_healthy` = 0) | `dli_request_errors_total{type}`; server log |
| `/readyz` 503 | Draining or engine dead | `/admin/state` (`draining`, `engine.healthy`, `engine.error`) |

## 8. Scaling, failure handling, limitations

One worker process serves one model on one device. Within it: continuous batching and paged KV (M2–M3) set capacity;
admission control turns overload into fast 429s instead of unbounded queues; cancellation returns memory as soon as a
client leaves; an engine crash marks the worker unready so an orchestrator stops routing to it. Across workers
(routing, retries, worker loss) is M8–M9. Performance under load is not yet measured (M5). Current limitations:
`docs/known-issues.md`.
