#!/usr/bin/env bash
# Regenerate benchmarks/baselines/vllm/requirements.txt (hash-pinned) from requirements.in.
# Runs LOCALLY. The vLLM baseline lives in its own environment: vLLM pins its own torch, which
# must not leak into the project lock (uv.lock).
#   * torch comes from the PyTorch CUDA 13.0 index, as in the pinned official Docker image
#     (torch 2.13.0+cu130); everything else from PyPI.
#   * --exclude-newer applies the supply-chain policy (only releases at least two weeks old when
#     the lock was made, 2026-11-11) and makes re-locking reproducible.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
dir=benchmarks/baselines/vllm
# --no-config: this environment is resolved on its own, not under the project's [tool.uv] settings
# (vLLM 0.32.0 needs setuptools<81, below the project's security floor; see benchmarks/baselines/vllm/README.md).
uv pip compile --no-config "$dir/requirements.in" \
  --output-file "$dir/requirements.txt" \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  --exclude-newer 2026-10-28T00:00:00Z \
  --generate-hashes \
  --no-header
echo "locked: $(grep -cE '^[a-zA-Z0-9]' "$dir/requirements.txt") packages; torch: $(grep -E '^torch==' "$dir/requirements.txt")"
