# QnA: implementation-specific technical questions

> **Scope: milestones M0–M3 (38 questions).** The release target is ≥ 30 questions across architecture, implementation,
> scalability, reliability, security, performance, debugging, and system design (master prompt §17).
> Questions about components that do not exist yet (API server, kernels, TP, router) are
> added when those components land, because answering them now would describe unbuilt software.
> Every number carries an evidence label and a result directory.

## Architecture

### 1. Why write the model from scratch instead of loading it with `transformers`?
The engine's original components need to own the model's inner operations. Paged attention (M2/M6) replaces how
attention reads K/V, continuous batching (M3) changes the tensor layout to flattened tokens from many sequences, and
tensor parallelism (M7) replaces `nn.Linear` with sharded linears. With `transformers` each of those would be a
monkey-patch against an API we don't control. `src/dli/model/llama.py` keeps the checkpoint's parameter names so loading
stays a 1:1, strictly checked mapping (`src/dli/model/loader.py`). `transformers` remains a test-only dependency used as
the numerical reference (`pyproject.toml`, `[dependency-groups].dev`).

### 2. Why does `forward` take flattened 1-D `input_ids` and explicit `positions` instead of a `[batch, seq]` tensor?
Continuous batching mixes prefill chunks and single decode tokens from different sequences in one step, at different
absolute positions. A padded `[batch, seq]` layout wastes compute on padding and assumes aligned positions. The 1-D
layout with explicit positions (`LlamaForCausalLM.forward`, `rope_cos_sin`) is what the runner will feed in M3. Only the
attention metadata changes; the per-layer math (`DecoderLayer.forward`) stays the same.

### 3. Where are the component boundaries in the engine, and why that layering?
`api → scheduler → kv → runner → model → kernels` with downward-only dependencies (ADR-0001,
`docs/architecture.md`). The model layer has no notion of requests, and the scheduler never touches tensors. That keeps the
original algorithms (block manager, scheduler) testable as pure data structures with property tests, and the model
testable against a reference without a scheduler. That is how M1 was validated (`tests/unit/model/`).

### 4. Why one lockfile with two mutually exclusive PyTorch extras?
CPU (laptop/CI) and GPU (Vultr A16) runs must use the same library versions, or a CPU-vs-GPU comparison would be
confounded. `pyproject.toml` declares `cpu` and `cu130` extras as a uv conflict pair, so `uv.lock` pins `torch==2.14.1+cpu`
and `torch==2.14.1+cu130` together. 2.14.1 is the newest torch built for CUDA 13.0, which is exactly what the host
driver (580.95.05) reports. torch 2.15 ships only for CUDA 13.2+, and Triton JIT paths are where CUDA minor-version
compatibility breaks (ADR-0002).

## Implementation

### 5. How is RoPE implemented, and what goes wrong if the layout is wrong?
`src/dli/model/rope.py` uses the rotate-half layout: dimension *i* is paired with *i + d/2*. Angles are
`positions × inv_freq` computed in fp32, duplicated to the full head dimension, and cast to the activation dtype only at
the end. HF Llama/Qwen checkpoints were trained with this layout. Using the interleaved (adjacent-pair) layout instead
produces text that is grammatical for a few tokens and then drifts, which is why the tiny-model parity test also runs at
200 positions (`test_positions_beyond_original_context_match`).

### 6. How does Llama 3 RoPE scaling work in your code?
`inverse_frequencies` with `rope_type="llama3"` computes each frequency's wavelength. Wavelengths shorter than
`original_ctx / high_freq_factor` (high frequencies, local position detail) are unchanged. Wavelengths longer than
`original_ctx / low_freq_factor` are divided by `factor`. The band in between is linearly interpolated with
`smooth = (original_ctx/wavelen − low) / (high − low)`. The tiny-model test sets `original_max_position_embeddings=32`
and evaluates 200 positions so all three bands are exercised against the reference.

### 7. How is grouped-query attention handled?
Query head *h* reads KV head *h // fan_out*, with contiguous groups as stored in the checkpoints. `Attention.forward`
expands K/V with `repeat_interleave(fan_out, dim=0)` before SDPA. It is simple and matches the reference, but it
materializes `fan_out×` more K/V traffic (7× for Qwen2.5-0.5B). The M6 Triton kernel will index the shared KV head
directly. The `mha` tiny variant (fan-out 1) and the default variant (fan-out 4) are both parity-tested.

### 8. Why are there three attention branches?
In `Attention.forward`: (1) one new token needs no mask, because it may attend to every cached position; (2) a fresh
prompt with no cache uses `is_causal=True`; (3) a multi-token chunk appended to a non-empty cache needs an explicit
boolean mask `k_pos <= q_pos`, because SDPA's `is_causal` aligns the causal triangle to the top-left and would hide
legitimate cached keys from the chunk. Branch 3 is not exercised by greedy decoding today, so it has a dedicated test
(`test_chunked_prefill_into_nonempty_cache_matches_single_pass`). Chunked prefill in M3 depends on it.

### 9. Why load weights onto the `meta` device first?
Constructing `nn.Linear` normally allocates and randomly initializes every parameter, which wastes time and holds two
copies of the model in memory during load. `load_model` builds the module under `torch.device("meta")` (shapes only),
loads tensors from safetensors, validates names and shapes, then calls `load_state_dict(..., assign=True)` so the
checkpoint tensors become the parameters. The non-persistent RoPE buffer is then rebuilt on the real device (`reset_rope`).

### 10. How are errors propagated from checkpoint and config problems?
Two typed exceptions: `UnsupportedModelError` (architecture features we do not implement, raised by
`ModelConfig.from_hf_dict`) and `CheckpointError` (pickle-only weights, missing/unexpected tensors, shape mismatch,
inconsistent tied head, raised by `load_model`). Both fail at load time instead of serving wrong output. Each path has a
test in `tests/unit/model/test_config_and_loader.py`.

## Security

### 11. What is the attack surface of model loading, and how is it reduced?
Weights come from the internet. PyTorch `.bin`/`.pt` files are pickles, and unpickling can execute arbitrary code. The loader
refuses directories that only contain pickle formats (`_safetensor_files`), and `hub.download` never requests `*.bin`
(`_ALLOW_PATTERNS`). Revisions are pinned commit hashes (`PINNED_REVISIONS`), so a compromised or changed `main` branch
cannot silently change what is loaded. Remaining gap: no per-file digest check against a signed manifest yet. The
portfolio's model-manifest contract (C8) adds that.

### 12. How are credentials kept off the GPU host?
`scripts/gpu/common.sh` authenticates through the local ssh-agent. It passes only the public key to `ssh -i` with
`IdentitiesOnly=yes` and sets `ForwardAgent=no`, so a compromised host cannot use the agent. `sync.sh` sends only files
from `git ls-files` (no `.git`, no ignored files such as `scripts/gpu/host.env`) plus a `REVISION` marker. Models are
ungated, so no Hugging Face token is needed on the host (`docs/operations.md` §1).

### 13. Why did the security workflow need to strip `+cpu`/`+cu130` from versions?
`pip-audit` resolves packages against PyPI. The PyTorch builds come from download.pytorch.org with local version labels,
so `torch 2.14.1+cpu` cannot be audited and the run fails. Advisories are published against the upstream version, so
`security.yml` strips the label and audits both locked dependency sets with `--no-deps` (the lock already pins the
full closure). Running it locally found a real issue: transitive `setuptools 81.0.0` had PYSEC-2026-3447 and
GHSA-xmj3-9gf7-h52w. A `constraint-dependencies` floor in `pyproject.toml` raised it to 84.2.1 (commit "security: raise
transitive setuptools").

## Debugging

### 14. Generated text is fluent but wrong after a few dozen tokens. How do you investigate?
That pattern points at position handling rather than weights: RoPE theta/scaling/layout, or KV-cache positions.
(1) Compare full-sequence logits at long positions against the reference
(`test_positions_beyond_original_context_match`). (2) Compare cached vs recompute generation
(`test_kv_cache_path_equals_recompute_path`). A mismatch there isolates the cache or mask branch from the model math.
(3) Check `ModelConfig.rope` against the checkpoint's `config.json`, including which format (Hub `rope_scaling` vs
v5 `rope_parameters`) it came from.

### 15. Our tokens diverge from the reference at step k. Bug or numerics?
Look at the reference's top-2 logit gap at step k, which the parity report records (`ref_top2_gap_at_divergence`). In fp32
both implementations agree to about 1e-6, so a divergence with a gap ≫ 1e-5 is a bug. In bf16 the logit resolution near
magnitude 32 is 0.25, so flips at gaps of that order are rounding, not bugs. The bf16 acceptance rule in
`tests/parity/test_hf_greedy_parity.py` encodes this.

## Performance

### 16. What has been measured so far, and what has not?
Correctness has been measured. Performance has not. The parity reports include wall-clock fields, but they come from
sequential, unbatched, single-prompt loops with warm-up included, so they are not benchmarks, and I don't compare engines
with them. The A16 validation includes smoke timings (≈ 15 TFLOP/s bf16 matmul, 168 GB/s device copy
`[GPU:A16x1@vultr]`), which are labeled as smoke checks for detecting a broken or throttled device, not results. The
benchmark harness (open-loop load, baselines, `benchmarks/methodology.md`) is milestone M5. GPU campaigns over 15
minutes need approval (Gate B).

### 17. Both parity suites show 100/100. What does that prove, and what doesn't it prove?
It proves that for two real checkpoints (Llama architecture via SmolLM2, Qwen2 architecture), the native model, loader,
RoPE, GQA, tied head, and KV-cached decoding reproduce `transformers` exactly: 25,600 greedy steps per precision, in
fp32 on CPU `[CPU-LAPTOP]` and bf16 on GPU `[GPU:A16x1@vultr]`. The fp32 run is also informative about margins: the
smallest reference top-2 logit gap across all steps was 1.2e-4, about 100× above fp32 implementation noise, so a single
flipped token would have pointed at a real bug.
It does **not** prove: Llama-3 RoPE scaling on real weights (gated checkpoints; verified only on random tiny models),
correctness of sampling (not implemented), behavior beyond ~180 total tokens, or anything about batching or performance.

### 18. In bf16 the reference had exact logit ties (top-2 gap 0.0), yet every token matched. Why?
In bf16, distinct logits often collapse onto the same representable value. Logits near magnitude 32 are spaced 0.25
apart, so exact ties are common over 25,600 steps. A tie is resolved by the argmax rule, and both implementations take
the lowest token id (`torch.argmax` semantics, documented in `greedy_generate`). Matching across ties therefore also
shows that our bf16 logits round to the same values as the reference's at those steps. Had the tie-break rules
differed, the suite would have reported a divergence at a gap of 0.0, which the acceptance rule treats as rounding.

## Debugging (continued)

### 19. Describe a real defect found during M1 and how it was handled.
A **provenance bug** in the evidence tooling: the first full CPU parity run started from commit `d676e49`, but its
`env.json` reported a later revision (`…-dirty`), because `dli.envinfo` read `git describe` when results were *written*
(70 minutes later, after unrelated commits). Handling: (1) verified the measurement was still valid with
`git diff d676e49 46f6359^ -- src tests pyproject.toml uv.lock` (empty); (2) fixed the cause in `46f6359`: the revision is
captured at run start (`current_revision()`) and passed to `collect(revision=…)`, and the parity suite also stores it as
`code_revision`; (3) added a regression test that fails if a moved HEAD is reported; (4) left the raw `env.json` unedited and
recorded the correction in `PROVENANCE.md` beside it. Lesson: evidence metadata has to be pinned at the start of a run,
just like inputs.

### 20. A GPU result says `revision: …-dirty`. Can you trust it?
Not on its own. "Dirty" means uncommitted files were present, so the exact code is unknown. That happened on the first A16
validation, and it was rerun after `scripts/gpu/sync.sh` gained `SYNC_FROM=head`, which ships `git archive HEAD` and writes
the exact commit to `REVISION`. Evidence runs use HEAD sync. Worktree sync is for iteration only (`docs/operations.md` §3).

## Paged KV cache (M2)

### 21. Why page the KV cache, and what exactly does the block manager own?
A contiguous cache reserves `max_len` slots per sequence up front (`ContiguousKVCache`). For Qwen2.5-0.5B in bf16 that
is 12 KiB per token, 48 MiB for a 4K context, whether used or not. Reserved-but-unused KV is what caps concurrency.
Paging allocates 16-token blocks on demand, so waste is under one block per sequence. The `BlockManager`
(`src/dli/kv/block_manager.py`) owns only **metadata**: block tables, reference counts, the prefix index, and eviction
order. `PagedKVStore` (`paged_store.py`) owns the bytes. The model sees a `KVCacheView` (`length`/`write`/`advance`), so
the attention math did not change between M1 and M2.

### 22. What states can a block be in, and why is "cached-idle" a separate state?
FREE (on the free list), USED (`ref > 0`), and CACHED-IDLE (`ref == 0` but registered in the prefix index). Cached-idle
blocks hold valid KV that a later request can revive with a prefix hit, so freeing a finished request must *not* discard
them. They still count as available memory, because they can be evicted on demand. `check_invariants()` asserts that every
block is in exactly one state, and the stress soak checks conservation (`free + idle + used == pool`) after every operation.

### 23. How is a block identified for prefix sharing, and why SHA-256?
A block's KV depends on every earlier token, so identity is a hash chain:
`h_i = SHA-256(len(ns) ‖ ns ‖ h_{i-1} ‖ int64 tokens_i)` (`hashing.py`). A collision would make one request attend to
another's cached KV: wrong output *and* a data leak. Python's 64-bit tuple hash is not built to resist crafted
collisions, and hash-keyed prefix caches are a known vulnerability class. SHA-256 costs about a microsecond per 16-token
block. The namespace is length-prefixed so `("ab", …)` cannot alias `("a", …)` (tested).

### 24. Why can a block only be shared after `mark_computed`, and why is the last prompt block always recomputed?
Allocation reserves slots before the forward pass writes them. If blocks were registered at allocation, a request
admitted in the same step could "hit" a block whose KV is still garbage. That is the first fault the mutation script
injects, and the shadow-KV property test catches it. The last-block rule exists because a fully cached prompt would
leave zero tokens to run through the model, and then there are no logits to sample from. `_lookup` only considers
blocks covering positions `< len(prompt) - 1`. The fault-injection run showed my first property test did not check this
(content was still correct), so the rule is now asserted explicitly.

### 25. How does copy-on-write work, and how did you prove it?
`fork(parent, child)` increments the reference count of every block. When either sequence appends into a *partial*
last block with `ref > 1`, `append_slots` allocates a private block, returns `CopyOp(src, dst)`, and moves the sequence
to the copy. The caller applies the copy across all layers (`PagedKVStore.copy_blocks`) before writing. Full shared
blocks are never written, because appends only touch the last partial block or a new one. Evidence:
`test_fork_with_copy_on_write_matches_independent_runs` teacher-forces different continuations into a parent and its fork
and matches every step's logits against independent full recomputation, with exactly one COW copy. Removing COW is
mutant #1, caught as "KV overwritten".

### 26. How do you guarantee an allocation failure never leaks blocks?
Every mutating operation computes its full requirement first: new blocks needed, minus prefix hits, with revived
idle hits subtracted from what is evictable, plus one block for a pending COW. It raises `NoFreeBlocksError` before
changing anything. The scheduler (M3) can then preempt and retry safely. Tests assert byte-for-byte unchanged state after
failures (`test_allocation_failure_is_atomic`, `test_append_failure_is_atomic`). The million-operation soak hits about
27,000 out-of-space events per policy and still ends with zero leaked blocks.

### 27. Describe the liveness bug you found in your own design.
With duplicate content (two identical prompts admitted before either was computed), my first `_register` linked a
block under whichever block the index held for the parent *hash*, possibly another sequence's copy. That other parent
could then be idle while owning an in-use child. It counted as "available" but could never be evicted (only leaves are
candidates), so the allocation pre-check could pass and the subsequent eviction fail. Fix: register a block only if the
cached parent is the exact block preceding it in *its own* table. `check_invariants()` now asserts that an in-use cached
block's parent is in use, which makes the availability count exact. Found in self-review before the first test run;
recorded in the commit message.

### 28. Is the block manager thread-safe?
No, deliberately. It is **thread-confined** to the scheduler loop, the single writer of KV metadata in this design
(ADR-0001). Locking every append would add overhead without benefit. Accidental cross-thread use is detected, not
tolerated: with `debug_checks=True` every public method checks the caller's thread id and raises
(`test_cross_thread_use_is_rejected`). If the engine ever moves to multiple scheduler threads, this would need a lock
or sharding per pool, and the stateful tests would need a concurrent variant.

### 29. How do you know your tests would actually catch the bugs that matter?
Three layers. (1) A Hypothesis state machine with a *shadow KV store* that records which token prefix each slot holds
and asserts, after every step, that every live sequence still owns its computed KV and every prefix hit returns matching
content. (2) Fault injection: `scripts/dev/mutation_check_block_manager.sh` plants four realistic bugs, and the suite must
fail on each. All four are caught; the first run caught a gap in the tests themselves. (3) End-to-end equality: paged
decoding must equal the contiguous path, which M1 tied to `transformers`, on tiny models across block sizes, sharing,
forks and eviction, and on SmolLM2-135M (40/40 runs `[CPU-LAPTOP]`).

### 30. Did the cost-aware eviction policy work?
Partly, and the negative half is published. On synthetic traces replayed through the real `BlockManager`
(`benchmarks/results/2026-11-10-SIM-m2-prefix-eviction/`, `[SIM]`, hit rate only), GDSF beat LRU on
**shared-system-prompt** traffic in every seed (+2.5 to +4.1 hit-rate points under memory pressure). It **lost** on
**multi-turn** sessions in every seed (down to −10.9 points with a 16K-token pool) and was slightly worse with scan
pollution. That refutes the spec's hypothesis for multi-turn traffic. The likely mechanism, not yet isolated: early blocks
of a session get hit every turn, accumulate frequency, and keep pinning memory after the session ends, while aging is
slow when evictions are rare. Decision: LRU stays the default and cost-aware is opt-in. I deliberately did not tune
the policy against the same synthetic traces used to evaluate it. Follow-ups (frequency decay, ARC-style adaptation)
must be judged with the same paired method and, more importantly, with latency measurements once the engine exists (M5).

## Continuous batching and scheduling (M3)

### 31. Walk through one engine step.
`LLMEngine.step()` (`src/dli/engine/engine.py`) calls `Scheduler.schedule()`, which builds one batch under a token
budget and a sequence cap. Running requests come first, sorted by (effective priority, arrival): each gets its next
decode token or its next prompt chunk, after reserving a KV slot. Then, if nothing was preempted, waiting requests are
admitted by (effective priority, resumed-first, tenant virtual time, arrival), with prompts chunked to the remaining
budget. `ModelRunner.execute()` applies copy-on-write copies and runs **one** forward pass over a `BatchedKV`.
Embedding, projections, MLP, and the LM head run once over all flattened tokens, and attention runs per segment against
each sequence's paged KV. Logits are computed only for rows that sample. Then `sample()` runs per request, stop rules are
applied, and finished requests free their KV. `StepStats` times each phase.

### 32. Why serve running requests before admitting new ones, and why chunk prefills?
Prefill-first lets a burst of new prompts stall every in-flight decode for a whole step: inter-token latency spikes
exactly under load. Running-first keeps decodes flowing, and chunking caps any prompt's share of a step at the remaining
budget, so one 40K-token prompt cannot freeze the batch. `test_long_prompt_is_chunked_and_running_decodes_are_not_stalled`
shows a running request decoding every step while a 40-token prompt is processed in ≤ 7-token chunks. The cost is that
a long prompt's time to first token grows with the number of chunks. That trade-off is tunable via the budget.

### 33. How does preemption work, and why can't the system livelock?
When a running request needs a KV block and none is available (`_reserve` fails), the scheduler preempts the **newest
running request not already in this step's batch** (`_pick_victim`). It frees the victim's blocks (its full blocks stay
prefix-cached), keeps its generated tokens, and puts it at the front of the waiting queue. On readmission it recomputes
`prompt + outputs`, reusing any cached prefix. Liveness: victims are always newer than the requester, so the request at
the front of the running order is never preempted. Capacity rejection (`Scheduler.add`) guarantees any admitted request
fits in the pool alone, so the front request always completes. No admission happens in a step that preempted, which
prevents preempt/readmit thrashing. The randomized scenarios enforce a step bound, and the over-admission mutant
(admitting one block too large) is caught as a livelock.

### 34. Why recompute instead of swapping preempted KV to host memory?
Recompute needs no extra memory or transfer path and combines with the prefix cache: the victim's full blocks
usually survive as cached-idle, so resume recomputes only part of the sequence (`test_resumed_request_reuses_its_cached_prefix`).
Swap wins when PCIe transfer of the KV is cheaper than recomputing it, which depends on model size, context length, and
the GPU. That has to be measured on GPU hardware, so swap is deferred (known issue 17), not ruled out.

### 35. How does tenant fairness work, and how did you show it does something?
Each tenant has a virtual time that advances by tokens served divided by its weight (`_charge`). Admission picks the
tenant furthest behind. A tenant that becomes active starts at the current virtual clock (the minimum among active
tenants), so idle time cannot be banked as credit. Priority classes order INTERACTIVE before BATCH, and aging
(`aging_steps`) promotes long-waiting work, which bounds starvation. Tests show equal shares despite one tenant's 20-request
head start, a 2:1 weighted share, no credit banking, and a batch request admitted within `aging_steps + 3` under a
continuous interactive stream. Fault injection shows each test is necessary: FIFO admission fails the share tests, removing
aging fails the starvation test, and allowing banking fails the banking test.

### 36. How do you test a scheduler without a real model, and why trust the result?
`tests/unit/engine/fake.py` is a deterministic model whose next token hashes the **entire** token history. Any
dropped, duplicated, or reordered token, or progress lost across preemption, changes every later token. Each request is
compared with the same rule run alone. The fake runner also asserts that every chunk starts exactly at the block manager's
computed count. On top of that, Hypothesis generates adversarial scenarios (1-token budgets, single-sequence caps, pools
barely larger than requests, cancellations at every lifecycle stage) and checks budget, cap, chunk continuity, and
block invariants every step, plus liveness, exact outputs, and zero leaks. The focused profile (2,000 scenarios) passed,
and three injected scheduler faults were detected. Real tensors are then covered by tiny-model engine tests and the
real-checkpoint parity runs.

### 37. Your bf16 engine parity check failed. What happened and what did you conclude?
Batched vs unbatched on the A16 in bf16, 100 prompts × 128 tokens: only 22/100 (SmolLM2) and 18/100 (Qwen2.5)
were identical, failing my pre-set "≥ 98% identical over 64 tokens" rule copied from M1. I recorded the failure unedited,
then analyzed it. Every divergence occurred at a reference top-2 gap ≤ 0.375 (median 0.0 for SmolLM2: exact bf16 ties),
none beyond bf16 rounding, and preempted requests diverged no more often than others. The decisive test was the same
engine, configuration, and GPU in **fp32**: 100/100 identical for both models, with the same 47 preemptions, and CPU fp32
agrees. So the engine is exact, and the bf16 differences come from batched GEMMs accumulating in a different order. My
error was the acceptance rule: token identity across different batch shapes is only a valid criterion in fp32. For bf16,
the right checks are "no divergence beyond rounding" plus task-level quality (M5).
(`benchmarks/results/2026-11-10-GPU-A16x1-vultr-m3-engine-parity-bfloat16/ANALYSIS.md`)

### 38. How much does the engine itself cost, and what limits throughput at 10× load?
With a null runner (no model), engine overhead was flat at roughly 28–37 µs per sequence per step from 8 to 256
concurrent sequences `[CPU-LAPTOP]` (≈ 2× run-to-run variance on this laptop, so only the flat shape is claimed). The
largest component is the sampler's argmax over `[batch, vocab]`. On SmolLM2-135M the engine produced 107 output tok/s vs
21 tok/s for sequential decoding (5.2×, median of 3 interleaved runs), with overhead at 0.6% of step time. At 10× load
the limits are, in order: the per-segment Python attention loop (replaced by the M6 kernel), KV memory (more preemption;
the pool size and prefix-hit rate decide recompute waste), the single-threaded loop (M4 adds async ingress), and
eventually the O(W) admission scan (known issue 16) at ~10⁴ queued requests.

## OpenAI-compatible server (M4)

### 39. Walk through one HTTP request.
`BodySizeLimit` bounds the body (413) and `AuthMiddleware` checks the bearer key (401) before anything is parsed.
The handler validates the request (pydantic, plus an explicit list of unsupported OpenAI features that get 400 instead
of being silently ignored), tokenizes or renders the chat template in a sandboxed Jinja environment, and checks lengths
against the context and the KV pool. Admission control then accepts or rejects (429/503) from the latest engine snapshot.
An accepted request becomes an async generator over `AsyncEngine.generate()`: token ids arrive from the engine thread,
pass through the incremental detokenizer and the stop-string matcher, and leave as one JSON body or SSE chunks.

### 40. Why a dedicated engine thread rather than running the engine on the event loop?
A step is a blocking forward pass (milliseconds to hundreds of milliseconds). On the loop it would stall every
connection, including health checks and disconnect detection. A thread pool would break the engine's single-owner
invariant (the block manager is thread-confined since M2). So one thread owns the engine and receives typed commands
(add, abort, stop) through a queue; results go back with `loop.call_soon_threadsafe`. After each step the thread
publishes an immutable `EngineSnapshot` that admission, readiness, and metrics read without locks.

### 41. How does cancellation work, and how do you know memory is freed?
Every request is an async generator wrapped in `aclosing()`; whatever ends it (client disconnect, stop string, timeout,
deadline miss, shutdown) runs its `finally`, which sends an abort to the engine thread, which frees the KV blocks. Tests
check the outcome, not the mechanism: after a client drops a stream, `kv_used_blocks` returns to 0 and the generated-token
count stops short of `max_tokens`. Writing those tests for the non-streaming and still-queued cases found a real gap:
disconnects were noticed only when a chunk was written, so an abandoned 3000-token non-streaming request ran to the end.
A watcher on the ASGI `http.disconnect` message now cancels the work (commit 3eb47cb).

### 42. What does admission control reject, and why reject rather than queue?
Four reasons: draining (503), too many in-flight requests or too many queued requests (429 + `Retry-After`), and a
deadline that cannot be met: with `X-Deadline-Ms`, predicted time to first token = queued prefill tokens ÷ measured prefill
rate (EWMA). A request still queued when its deadline passes is aborted and gets 429 before any byte is sent. Queuing
work that will miss its deadline wastes capacity and gives the client no signal; a fast 429 lets a router or client
retry elsewhere. The predictor is deliberately simple and ignores decode contention (known issue #26).

### 43. How do stop strings work when a stop string spans tokens?
Stop strings are matched on decoded text. `StopStringMatcher` holds back any suffix of the output that is a prefix of
a stop string, so a partial match is never streamed and then "taken back". On a full match the text before it is
emitted, `finish_reason` is `stop`, and the generator is closed, which aborts the engine request at once. The
incremental detokenizer never emits a partial UTF-8 character (it waits while the decoded text ends in U+FFFD).

### 44. Why wait for the first token before sending response headers?
Once a 200 and SSE headers are sent, an error can only be reported in-band. Waiting for the first event means anything
that fails before then (deadline missed while queued, context or capacity errors, a dead engine) is a real HTTP status
that clients and load balancers understand. After the first token, failures are a `data: {"error": ...}` event followed
by `[DONE]`, as OpenAI does.

### 45. Describe a concurrency bug you found in your own async code.
The request timeout used `asyncio.timeout()` inside the request generator. That context manager cancels the task that
*entered* it, but the generator is entered in one task and iterated in another (Starlette iterates streams in its own
task). For streams, the timeout therefore never fired. It surfaced when the disconnect watcher moved the first-token
wait into its own task and the 504 contract test started failing. The fix bounds each wait separately with
`asyncio.wait_for` against a fixed deadline, in whichever task is iterating, and a regression test covers both forms.

### 46. Which security controls does the server have?
Safe defaults: loopback bind, auth required (refuses to start without keys, refuses `--no-auth` on a public address).
API keys `dli_<id>_<secret>`: only Argon2id hashes are stored; lookup by id, a dummy verification for unknown ids
(no timing oracle on key existence), verification off the event loop, a SHA-256 cache of verified tokens. Auth and the
body limit run before parsing. The tenant comes from the key; a client's `X-Tenant-Id` is ignored. Chat templates run in
an immutable sandboxed Jinja environment. Metrics and drain controls are on a separate loopback listener with an
optional constant-time-compared token. Limits on `max_tokens`, stop strings, and request time bound resource use.

### 47. How do you know it is actually OpenAI-compatible?
Three layers, all against the real server over TCP: (1) every response's status must be documented in contract C1 and
its body must validate against the contract schema, stream events included (writing these found five undocumented
statuses); (2) the official `openai` Python SDK, unchanged, for both endpoints, streaming, stop sequences, exception
mapping (401/400/404/429/503) and async cancellation; (3) on real checkpoints, prompt ids equal `transformers`' tokenizer
and chat template for all 100 prompts, and concurrent HTTP output equals `transformers`' decode of the unbatched
reference tokens.

### 48. Why are metrics on a separate port, and what do they tell an operator?
Metrics reveal traffic and tenant activity, and drain is a control action: neither belongs on the public port. They
show requests by endpoint and status, errors by type (including in-band stream failures and client disconnects),
rejections by reason, prompt and generated tokens, TTFT / inter-token / end-to-end latency histograms, and engine state
from the snapshot (queue depth, KV blocks used/free/cached, prefill rate, step time, preemptions, prefix hit ratio,
health), plus process and GPU memory.

### 49. What has M4 not shown?
Performance. The tests prove behaviour under overload, cancellation and failure, not throughput or latency; the HTTP and
detokenization overhead is unmeasured. Open-loop load with TTFT/ITL percentiles and comparison to vLLM are M5.

## Benchmarking and evaluation (M5)

### 50. How did you make the benchmark comparison fair?
The rules were committed before any run (ADR-0004, `benchmarks/methodology.md`). Every backend is driven through the
same OpenAI-compatible API by the same client; requests are seeded token-id prompts with `ignore_eos`, so every backend
does identical work. Both engines get the same scheduler limits and the **same KV capacity in tokens**: vLLM sizes its
pool from a memory fraction and logs the capacity, and the native pool is set to the same number of blocks. vLLM runs
its normal optimized configuration from a digest-pinned image; detuning a baseline would be a form of fabrication. The
static-batching baseline is tested to produce exactly the tokens of unbatched generation. One server is on the GPU at a
time and the backend order rotates each repetition.

### 51. Why open-loop load, and how is latency measured?
In a closed loop, a slow server slows its own client, so latency percentiles look best exactly when the system is
overloaded (coordinated omission). The load generator schedules Poisson arrivals in advance and measures every latency
from the scheduled arrival, not from when the request was sent; its own lateness is recorded (P99 ≤ 24 ms in every GPU
run, so the client was never the bottleneck). Closed loop is used only for maximum throughput at a stated concurrency.

### 52. What did the benchmarks show?
On the A16 the native engine reaches 225 output tok/s at concurrency 16 on Qwen2.5-0.5B, vLLM 1,279 and HF static
batching 677. vLLM leads by 1.7-9.2x across the model ladder, and the engine is 0.33-0.49x of static batching, so
AC3 (≥ 5x static batching) failed, as did its CPU part (0.73x vs 2x). The engine does win on latency under load:
sub-second TTFT where static batching queues for 10-30 s. All of it is published, including the failures.

### 53. Why is your engine slower than plain `transformers` with static batching?
The profile answers it. Each decode step loops over every sequence in every layer: gather that sequence's KV pages,
expand 2 KV heads to 14 by copying, call attention. That is ~170 kernel launches per sequence per step: 11,888 per
step at batch 64, ~35x the memory-bandwidth floor, with the GPU ~49% busy. Static batching makes one padded attention
call per layer, so its kernel count does not grow with the batch. The scheduler is not the problem (0.1-0.4 ms per
step). M6 replaces the loop with one batched, GQA-aware paged-attention kernel per layer, then adds CUDA graphs.

### 54. What did the bf16 study conclude about the M3/M4 failures?
Teacher forcing fed every configuration the same tokens. The fp32 engine matched the fp32 reference at 100% of
12,800 steps through 47 preemptions, so the engine adds no error. bf16 changes 1.4-1.5% of argmax decisions, all at
reference top-2 gaps ≤ 0.39, equally with or without batching (98.57% unbatched vs 98.64% engine). Perplexity rose
0.05-0.09% and ARC-Easy accuracy moved +0.04 points [−0.38, +0.46]: inside the pre-registered tolerances. One target
was missed: agreement at margins above one bf16 ulp was 99.85% vs ≥ 99.9%, equally for unbatched bf16, because
measured per-logit error is several ulps. The exact-match failures stay recorded; they meant rounding, not bugs.

### 55. What went wrong in your own benchmark harness, and how did you handle it?
Two defects surfaced in the data. The saturation rule divided completed requests by a span that included the drain
after the last arrival, so stable points with long outputs looked saturated: sweeps stopped early and the AC5 overload
test ran at about 0.5-1x of true saturation instead of 2x. And the admission predictor, estimating prefill speed from
decode-dominated steps, over-predicted TTFT 10-50x and rejected half of the requests that would have met their
deadline. Both were reproduced in regression tests and fixed; the original results stay, with the explanation.
AC5 was re-measured at ~4x saturation: no crash, admitted TTFT P99 1.98 s (vs 90 s without admission control), but
rejection latency P99 2.01 s against < 50 ms, because the corrected predictor is now optimistic for 13-18% of
requests, which are admitted and then expire in the queue. AC5 is recorded as failed on that criterion.

### 56. Does prefix caching help?
On prompts dominated by long shared prefixes, yes: median TTFT 60 vs 157 ms at 2 req/s, and the engine stays stable at
a load where the ablation saturates. On short multi-turn conversations it saves 32% of KV memory (hit ratio 0.69) but
no latency, because this engine's step time is dominated by decode overhead, not by the prefill it avoids. That
should change once M6 removes the decode overhead, and will be re-measured.

## Results (labeled evidence)

| Claim | Label | Evidence |
|---|---|---|
| 100/100 prompts × 128 tokens identical to `transformers`, SmolLM2-135M and Qwen2.5-0.5B, fp32 | `[CPU-LAPTOP]` | `benchmarks/results/2026-11-10-CPU-LAPTOP-m1-greedy-parity-float32/` (+ `PROVENANCE.md`) |
| Same, bf16, both implementations on one GPU | `[GPU:A16x1@vultr]` | `benchmarks/results/2026-11-10-GPU-A16x1-vultr-m1-greedy-parity-bfloat16/` |
| A16 host usable: fp32/SDPA numerics, Triton JIT, container GPU access | `[GPU:A16x1@vultr]` | `benchmarks/results/2026-11-10-GPU-A16x1-vultr-env-validation/` |
| Tiny-model logit parity ≤ 1e-4 on 7 architecture variants | `[CPU-LAPTOP]` | `tests/unit/model/test_reference_parity.py` (44-test fast suite) |
| Paged decoding == contiguous decoding, 40/40 runs (cold + shared-prefix), SmolLM2-135M, block 16 | `[CPU-LAPTOP]` | `benchmarks/results/2026-11-10-CPU-LAPTOP-m2-paged-parity/` |
| Block manager: 10^6 random ops per policy, 0 invariant violations, 0 leaks; 10k-request soak, 0 leaks | `[CPU-LAPTOP]` | `tests/stress/test_block_manager_soak.py` (run output in commit aeacae5) |
| Eviction policy hit rates (cost-aware wins shared-system, loses multi-turn) | `[SIM]` | `benchmarks/results/2026-11-10-SIM-m2-prefix-eviction/` |
| Engine == unbatched path, 100/100 prompts × 128 tokens, both models, fp32, with 47 preemptions and chunked prefill | `[CPU-LAPTOP]` | `benchmarks/results/2026-11-10-CPU-LAPTOP-m3-engine-parity-float32/` |
| Same, fp32 on GPU | `[GPU:A16x1@vultr]` | `benchmarks/results/2026-11-10-GPU-A16x1-vultr-m3-engine-parity-float32/` |
| bf16 engine vs unbatched: 22/100 and 18/100 identical (criterion failed); all divergences at bf16 near-ties | `[GPU:A16x1@vultr]` | `benchmarks/results/2026-11-10-GPU-A16x1-vultr-m3-engine-parity-bfloat16/ANALYSIS.md` |
| HTTP server (OpenAI SDK, 48 concurrent requests) on real checkpoints: output identical to `transformers`' decode of the unbatched reference 24/24 + 24/24, stream == non-stream 24/24, prompt ids equal `transformers`' for 100/100 prompts, fp32 | `[CPU-LAPTOP]` · `[GPU:A16x1@vultr]` | `benchmarks/results/2026-11-11-CPU-LAPTOP-m4-http-parity-float32/`, `benchmarks/results/2026-11-11-GPU-A16x1-vultr-m4-http-parity-float32/` |
| Same in bf16: stream == non-stream criterion **failed** (11/24, 15/24); every divergence at a near-tie | `[GPU:A16x1@vultr]` | `benchmarks/results/2026-11-11-GPU-A16x1-vultr-m4-http-parity-bfloat16/ANALYSIS.md` |
| Batched engine 5.2× sequential output tok/s on SmolLM2 (107 vs 21, median of 3 interleaved); engine overhead 0.6% of step time | `[CPU-LAPTOP]` | `benchmarks/results/2026-11-10-CPU-LAPTOP-m3-scheduler-overhead/` |
