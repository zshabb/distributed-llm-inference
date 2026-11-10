# Progress log

Per-session record (ENGINEERING_STANDARDS §13). Newest first.

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
