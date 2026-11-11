# Progress log

Per-session record (ENGINEERING_STANDARDS §13). Newest first.

## 2026-11-10/11 — Session 4: M4 OpenAI-compatible async server (complete, branch `feat/m4-openai-api`)

### Completed
- M3 follow-ups: path-disclosure record (`SECURITY.md`, known issue #22 closed per owner decision), M5 bf16 evaluation
  plan, 5.2× figure labeled as a preliminary CPU observation.
- **M4:** `dli.server`: text layer (sandboxed chat templates, incremental detokenizer, stop strings), `AsyncEngine`,
  OpenAI routes with SSE, admission control and drain, API-key auth and body limits, Prometheus metrics and admin
  listener, `dli-serve` entry point with graceful drain; contract C1 extended (additive).

### Tests executed (results)
| Suite | Result | Label |
|---|---|---|
| Fast suite (incl. 63 real-HTTP e2e tests) | 231 passed (~45 s) | `[CPU-LAPTOP]` |
| HTTP parity fp32, 24 prompts × 48 tokens × 2 models, 48 concurrent SDK requests | 24/24 + 24/24 on every check | `[CPU-LAPTOP]` code `7463dd4` |
| Same, fp32 | 24/24 + 24/24 on every check | `[GPU:A16x1@vultr]` code `7463dd4` |
| Same, bf16 | stream == non-stream 11/24 + 15/24: **pre-set criterion failed**; divergences at near-ties | `[GPU:A16x1@vultr]` code `7463dd4` |
| Prompt tokenization vs `transformers`, 100 prompts × 2 models | 100/100 + 100/100 | all runs |
| pip-audit (cpu + cu130), gitleaks full history | no known vulnerabilities; no leaks | — |

### Defects and corrections found
1. Non-streaming and queued requests were not cancelled on client disconnect (found while writing the KV-leak tests;
   fixed in `3eb47cb` with regression tests).
2. The request timeout never fired for streams: `asyncio.timeout()` inside a generator binds to the task that entered
   it (fixed in `f073e64`; regression test for both forms).
3. **Process error:** `3eb47cb` was pushed with one test failing; `pytest | tail` hid the exit code. Fixed by
   `f073e64`; test commands now run with `pipefail`.
4. Contract C1 lacked five statuses the server returns and had no stream-event schema (found by the conformance tests).
5. uvicorn installs its own signal handlers, which would skip draining; overridden via `capture_signals`.
6. A real `dli-serve` run logged no uvicorn startup or access lines: the admin listener's uvicorn config reset the
   process-wide loggers. Found by smoke-testing the README; fixed in `de754c2` with a regression assertion.
7. **Acceptance-rule error (again bf16):** the HTTP test asserted stream == non-stream in every dtype although M3 had
   shown bf16 is batch-composition dependent. Recorded as failed with an analysis; assertion now fp32-only (`e85c572`).

### Four-perspective review
- **Staff:** the engine stays synchronous and single-owner; the server talks to it only through a command queue and an
  immutable snapshot. Every request path ends in one `aclosing()` exit. Debt: deadline predictor ignores decode load
  (#26); keys load at startup (#25).
- **Security:** auth on by default and before parsing; Argon2id hashes; no timing oracle on key ids; bounded bodies and
  generation; sandboxed templates; admin surface off the public port. Dependencies audited. No TLS in-process (#24).
- **SRE:** readiness reflects drain and engine health; SIGTERM drains; metrics cover the golden signals plus KV and
  queue state. No alerts or dashboards yet (M9).
- **Interviewer:** behaviour proven over real TCP with the official SDK and the contract; two real concurrency bugs
  found by tests; a second honest negative bf16 result. No performance claims (M5).

### GPU usage
| Start (UTC) | End (UTC) | Duration | Purpose |
|---|---|---|---|
| 2026-11-11 00:06:04 | 00:08:18 | 134 s | M4 HTTP parity, bf16 + fp32 (incl. dependency sync), ≈ $0.018 |

### Next steps (awaiting approval)
M5: benchmark harness and baselines (HF static batching, vLLM on the A16), open-loop TTFT/ITL percentiles, bf16
evaluation per `docs/evaluation-plan-m5.md`.

## 2026-11-10 — Session 3: publication + M3 continuous batching (complete, branch `feat/m3-continuous-batching`)

### Completed
- Repository **published** (public, owner-approved) with its full history; first GitHub CI + security runs passed.
- Portfolio standards: commit-history-as-engineering-journal, development visibility, **no AI co-author trailers**
  (applied from commit `8d26e41` onward; earlier published commits left unchanged).
- **M3:** `BatchedKV` batched attention (refactor), `dli.sampling`, `dli.engine` (request lifecycle, scheduler with
  chunked prefill, LIFO recompute preemption, capacity rejection, abort, priority + aging + weighted tenant fairness,
  `ModelRunner`, `LLMEngine` with `StepStats`), overhead benchmark, milestone parity harness.

### Tests executed (results)
| Suite | Result | Label |
|---|---|---|
| Fast suite | 140 passed (< 15 s) | `[CPU-LAPTOP]` |
| Randomized scheduler scenarios, focused (2,000) | passed | 12 s |
| Fault injection: fairness (FIFO, no aging, banking) and scheduler (lost request, over-admission, ignored cached prefix) | 6/6 detected (+1 equivalent mutant explained) | — |
| Engine parity fp32, 100 prompts × 128 tokens × 2 models, 47 preemptions each | 100/100 + 100/100 | `[CPU-LAPTOP]` code `6132814` |
| Same, fp32 | 100/100 + 100/100 | `[GPU:A16x1@vultr]` code `5b4254c` |
| Same, bf16 | 22/100 + 18/100: **pre-set criterion failed**; all divergences at bf16 near-ties | `[GPU:A16x1@vultr]` code `6132814` |
| Overhead / batching gain | 28–37 µs/seq/step (flat); 5.2× tok/s on SmolLM2; overhead 0.58% | `[CPU-LAPTOP]` code `380a1fa` |

### Defects and corrections found
1. Test-design errors in early preemption tests (a request sized over capacity; an "oldest" request used as the
   intended victim; an under-sized scenario that never preempted). Fixed. Each one revealed a true property: capacity
   rejection, and the oldest request is never preempted.
2. Shared-prefix test assumed same-step requests share blocks. They cannot by design (rule 2); documented (known issue 19).
3. **Acceptance-rule error:** the bf16 exact-match rule copied from M1 is invalid for batched-vs-unbatched comparisons.
   Recorded the failure, proved engine exactness in fp32, and moved bf16 quality to M5 evaluation.
4. Push failed once on a transient WSL DNS error; retried successfully.

### Four-perspective review
- **Staff:** clean boundaries (scheduler is tensor-free; runner owns tensors; sampler stateless). Debt: O(W) admission
  scan, per-segment attention loop, `generate.py` bridge APIs kept as test references.
- **Security:** `abort` is idempotent; capacity rejection prevents a single request from wedging the engine. Open (M4):
  request size limits, deadline/queue bounds, tenant identity from authentication (today tenant is a caller argument).
- **SRE:** `StepStats` gives per-step timing and preemption counts. Prometheus export and alerts arrive in M4.
  Preemption churn under sustained overload has not been characterized yet (M5 load tests).
- **Interviewer:** the bf16 episode (failed criterion → root cause → decisive fp32 control) and the fault-injected
  fairness tests are the strongest evidence; all numbers are labeled and tied to commits.
No critical findings open.

### GPU usage (M3)
| Start (UTC) | End (UTC) | Duration | Purpose |
|---|---|---|---|
| 22:07:06 | 22:07:08 | 2 s | Sync + environment refresh |
| 22:07:20 | 22:16:05 | 525 s | bf16 engine parity |
| 22:17:24 | 22:26:14 | 529 s | fp32 engine parity (decisive check) |

### Next (awaiting approval)
M4: OpenAI-compatible async API server (SSE streaming, cancellation on disconnect, deadline-aware admission,
backpressure, Prometheus metrics, stop strings).

## 2026-11-10 — Session 2: portfolio policy consolidation + M2 paged KV cache (complete, branch `feat/m2-paged-kv`)

### Completed
- Owner policies consolidated (master prompt §25, standards §4/§6.1/§8.1/§14, CLAUDE.md files); M1 follow-ups
  (scoped parity claims, defense-in-depth wording, verified vs untested recovery steps, workflow deviation recorded).
- **M2:** `dli.kv`: SHA-256 hash chains, LRU and cost-aware (GDSF) eviction, `BlockManager` (ref counting, COW,
  watermark, atomic failure, register-after-compute, last-block recompute, prefix-closed leaf-first eviction, thread
  confinement, `check_invariants`), `PagedKVStore`, `PagedSequenceView`, `KVCacheView` protocol, `PagedSequence` and
  `paged_greedy_generate`.
- Docs: `docs/algorithms/paged-kv-cache.md`, `docs/algorithms/prefix-cache-eviction.md`, ADR-0003,
  `benchmarks/methodology.md`, ENGINEERING_GUIDE §1a/components/debugging, architecture, QnA (now 30), README.

### Tests executed (results)
| Suite | Result | Label / time |
|---|---|---|
| Fast suite | 95 passed (6 deselected: parity, stress) | `[CPU-LAPTOP]` 9–14 s |
| Stateful property tests, focused profile (2,000 × 80, both policies) | passed | 68 s |
| Fault injection (`scripts/dev/mutation_check_block_manager.sh`) | 4/4 mutants killed (after fixing a test gap) | ~10 s |
| Stress: 10⁶ ops × 2 policies; 10k-request soak | 0 violations, 0 leaks | 21 s |
| Paged-vs-contiguous on SmolLM2-135M (40 runs) | 40/40 identical | `[CPU-LAPTOP]` 131 s, code `c2e6800` |
| Eviction-policy trace simulation (120 replays) | cost-aware wins shared-system, loses multi-turn | `[SIM]` ~2 min, code `26b2bc0` |

**Not rerun (decision):** M1 real-checkpoint parity suites. M2 changed `src/dli/model/` only by a type annotation
(`KVCacheView`). `src/dli/generate.py`'s contiguous path, `uv.lock`, and the parity harness are unchanged, so no
numerics-relevant change. The paged path's equality to the contiguous path is the new evidence.

### Defects found and fixed
1. **Liveness bug (design, pre-test):** a duplicate-content block could be linked under another sequence's cached
   parent, leaving an idle-but-unevictable parent. Fixed by requiring the cached parent to be the sequence's own preceding
   block, plus a new invariant.
2. **Unbounded heap growth (pre-test):** GDSF lazy deletion leaked stale entries under revive/release churn. Fixed by compaction + test.
3. **Test gap (found by fault injection):** the property test did not enforce "never serve the whole prompt from cache". Assertion added.
4. **Provenance gap:** a parity run of an untracked (uncommitted) test file was labeled clean. Result discarded and rerun
   from a commit; `current_revision()` now treats untracked files as dirty (+ regression test).
5. Test bugs (not product bugs): an over-small pool in one eviction test; out-of-vocabulary token ids in another.

### Four-perspective review (ENGINEERING_STANDARDS §12)
- **Staff engineer:** clean layering (`dli.kv` depends only on `dli.model.config`; the model sees only `KVCacheView`).
  Debt: `can_allocate` + `allocate` hash the prompt twice (cache in M3 if profiling shows it matters); `generate.py`
  holds two bridge APIs that M3's runner replaces.
- **Security:** collision-resistant identities; namespace for model/tenant isolation; the prefix-hit timing side channel is
  documented. Open: a single very long prompt can evict the whole cache. Admission limits arrive with the API (M4).
- **SRE:** failures are atomic and observable via `stats()`. No exported metrics yet (Prometheus in M4). The capacity
  sizing helper (`num_blocks_for_budget`) exists but is unused until the runner sizes pools from free GPU memory.
- **Interviewer:** three original mechanisms with tests that demonstrably catch faults; a published negative result
  (multi-turn) with a stated, untested mechanism hypothesis; evidence labels and tiers respected (no GPU work in M2).
No critical findings open.

### GPU usage
None. M2 is CPU-only and the Vultr host was not used (idle since 2026-11-10 20:06:41Z).

### Next steps (awaiting approval)
M3: continuous-batching scheduler + batched runner over the paged KV cache (chunked prefill, preemption, cancellation),
including the engine-level soak moved from M2.

## 2026-11-10 — Session 1: M0 bootstrap + M1 model core (complete)

### Completed
- **M0:** uv project with one lockfile pinning torch 2.14.1 for `cpu` and `cu130` (ADR-0002); ruff, mypy --strict,
  pytest/hypothesis; pre-commit (format, lint, gitleaks, 1 MiB guard); ADR-0001/0002; architecture, plan, known issues;
  inference contract C1 (OpenAPI 3.1, validated); CI + security workflows (SHA-pinned; not yet executed); GPU host
  tooling and runbook; `dli.envinfo`.
- **Security:** pip-audit found transitive setuptools 81.0.0 advisories, raised to 84.2.1; both builds audit clean.
- **GPU host (Vultr A16):** inspected (A16 in PCI pass-through, Docker + NVIDIA toolkit preinstalled); user-level env from
  `uv.lock` in 25 s; Docker hardened by merging `"ip": "127.0.0.1"` into `daemon.json` (idempotent); validation passed.
- **M1:** native Llama/Qwen2 model, RoPE (default/linear/llama3), strict safetensors loader, pinned downloads, greedy
  generation; tiny-model parity tests; real-checkpoint parity suite.
- **Defects found and fixed:** setup_docker would have skipped an existing `daemon.json` (public port default);
  envinfo provenance bug (revision read at write time); an editor rewrite dropped executable bits on two scripts.

### Tests executed (results)
| Suite | Result | Label |
|---|---|---|
| Fast suite (unit, tiny-model parity, contract) | 44 passed | `[CPU-LAPTOP]` |
| fp32 parity, SmolLM2-135M + Qwen2.5-0.5B, 100 × 128 | 100/100 + 100/100 identical (70 min) | `[CPU-LAPTOP]` |
| bf16 parity, same suite | 100/100 + 100/100 identical (15 min) | `[GPU:A16x1@vultr]` |
| GPU host validation (numerics, Triton, container) | all checks pass | `[GPU:A16x1@vultr]` |
| pip-audit (cpu + cu130 locks) | no known vulnerabilities after fix | — |

### GPU sessions
| Start (UTC) | End (UTC) | Duration | Purpose |
|---|---|---|---|
| 19:48:19 | 19:51:19 | ~3 min | Connectivity, inspection, bootstrap, Docker config, validation (×2) |
| 19:51:31 | 20:06:41 | 910 s | bf16 parity suite |

### Status
M0 and M1 acceptance criteria met (`docs/implementation-plan.md`). CI is written but has never executed (unpublished).

### Known defects / open items
`docs/known-issues.md` #1–#3, #7–#9 (no open defects in shipped code).

### Next steps (awaiting approval)
M2: paged KV-cache block manager with prefix sharing (CPU, property-tested).
