# Progress log

Per-session record (ENGINEERING_STANDARDS §13). Newest first.

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
