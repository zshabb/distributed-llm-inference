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

### Fixed
- `envinfo` recorded the revision at write time instead of run start (provenance).
- `envinfo` reported a clean revision when untracked files were present.
- `setup_docker.sh` now merges the localhost bind into an existing `daemon.json` instead of skipping it.

### Security
- Raised transitive `setuptools` to >= 84.1.0 (PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w found by pip-audit).
