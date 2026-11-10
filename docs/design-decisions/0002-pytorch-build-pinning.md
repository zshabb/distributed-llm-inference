# ADR-0002: One lockfile, two PyTorch builds (cpu and cu130), pinned to torch 2.14.1

- Status: Accepted
- Date: 2026-11-10

## Context

Development and CI run on CPU-only machines; GPU validation runs on a Vultr A16 host whose driver (580.95.05) reports
CUDA 13.0 capability. We need (a) identical library versions on both, so CPU and GPU results compare like for like, and
(b) a fully pinned GPU environment that can be rebuilt on a fresh host.

Wheel availability on download.pytorch.org at decision time (checked 2026-11-10):

| Index | Newest torch for cp312 / x86_64 |
|---|---|
| `cpu` | 2.15.0 |
| `cu128` | 2.11.0 |
| `cu130` | 2.14.1 |
| `cu132` / `cu134` | 2.15.0 |

## Decision

Pin `torch==2.14.1`, exposed as two mutually exclusive extras in `pyproject.toml`: `cpu` (CPU index) and `cu130` (CUDA
13.0 index), declared as a uv conflict pair so `uv.lock` resolves and pins both. Machines select one:
`uv sync --frozen --extra cpu` (laptop/CI) or `--extra cu130` (GPU host).

## Rationale

- **Exact driver match:** cu130 wheels target exactly the CUDA version the host driver advertises. torch 2.15 only ships
  for CUDA 13.2/13.4. Those may run on a 13.0 driver through CUDA minor-version compatibility, but JIT paths (Triton
  kernels, PTX) are where that compatibility breaks, and custom Triton kernels are central to this project.
- **Same version on CPU and GPU:** avoids "the CPU test passed on 2.15, the GPU run used 2.14" ambiguity.
- **One lockfile:** a single reviewed artifact defines both environments, and `--frozen` makes drift impossible.

## Alternatives considered

- torch 2.15.0 + cu132: newer, but a CUDA runtime newer than the driver. Revisit if the driver is upgraded (owner action).
- torch 2.11.0 + cu128: older, with no benefit over cu130.
- Separate requirement files per machine: two sources of truth, easy to drift.

## Consequences

Upgrading torch is an explicit change to this ADR plus `uv.lock`. A driver upgrade on the host would allow moving
to a newer CUDA build.
