# Changelog

All notable changes are documented here, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/). Nothing has been released yet.

## [Unreleased]

### Added
- Project scaffold with one lockfile pinning torch 2.14.1 for CPU and CUDA 13.0 builds (ADR-0002).
- `dli.envinfo` environment manifests for reproducible evidence.
- Remote GPU host tooling (`scripts/gpu/`) and operations runbook.
- Inference API contract C1 v1 (OpenAPI 3.1) with validation tests.
- CI and security workflows (not yet executed; the repository is unpublished).
- Native Llama/Qwen2 model, strict safetensors loader, pinned hub downloads, greedy generation (M1).
- Parity suites: 100/100 token-identical vs `transformers` for two models in fp32 `[CPU-LAPTOP]` and bf16 `[GPU:A16x1@vultr]`.
- GPU host validation evidence and `SYNC_FROM=head` evidence-sync mode.

- Paged KV cache (M2): `BlockManager` (ref counting, copy-on-write, watermark, atomic failure, thread confinement),
  SHA-256 hash-chained prefix cache with leaf-first eviction, LRU and cost-aware (GDSF) policies, `PagedKVStore`,
  `PagedSequence` / `paged_greedy_generate`, the `KVCacheView` protocol.
- Property-based (stateful, shadow-KV) tests, fault-injection script, stress tier (`-m stress`), paged parity spot check,
  prefix-cache eviction trace simulation `[SIM]`.

- Continuous-batching engine (M3): batched attention over `BatchedKV` segments, `LLMEngine` step loop, `ModelRunner`,
  scheduler with chunked prefill, LIFO recompute preemption, capacity rejection, abort, priority classes with aging, and
  weighted per-tenant fair queuing; batched sampler with per-request seeded generators (`dli.sampling`).
- Randomized adversarial scheduler tests, fault-injected fairness/preemption checks, engine overhead benchmark,
  milestone engine-vs-unbatched parity harness.

- OpenAI-compatible async server (M4, `dli.server`): `/v1/completions`, `/v1/chat/completions`, `/v1/models`, SSE
  streaming with `include_usage`, stop strings, sandboxed chat templates, incremental detokenization; `AsyncEngine`
  (engine on a dedicated thread); deadline/capacity admission with 429 + `Retry-After`; client-disconnect cancellation;
  request timeouts; API-key auth (Argon2id); body-size limit; Prometheus metrics and an admin listener (state, drain);
  `dli-serve` entry point with safe defaults and graceful SIGTERM drain; `python -m dli.server.auth new <tenant>`.
- Contract C1: documented 401/404/413/500/504 responses and the `CompletionChunk` stream event schema (additive).
- Tests: real-HTTP e2e suite, contract conformance of live responses, official `openai` SDK compatibility,
  real-checkpoint HTTP parity (`-m parity`).

### Fixed
- `envinfo` recorded the revision at write time instead of run start (provenance).
- `envinfo` reported a clean revision when untracked files were present.
- Server (found by M4 tests before release): a client disconnect during a non-streaming request, or while queued, did
  not abort generation; the request timeout never fired for streams (it was bound to the wrong asyncio task); the
  admin listener's logging config silenced the API server's startup and access logs.
- `setup_docker.sh` now merges the localhost bind into an existing `daemon.json` instead of skipping it.

### Security
- Raised transitive `setuptools` to >= 84.1.0 (PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w found by pip-audit).
