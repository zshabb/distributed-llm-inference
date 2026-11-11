#!/usr/bin/env bash
# Idempotent USER-LEVEL setup on the GPU host. Runs on the host from the synced repo directory.
#   * installs a pinned, checksum-verified uv into ~/.local/bin (no apt, no system Python changes)
#   * installs the pinned Python and syncs the locked environment with the CUDA 13.0 torch build
# Re-running is safe and fast; the environment is fully determined by uv.lock.
set -euo pipefail
UV_VERSION="0.13.5"   # keep in sync with docs/operations.md
UV_ASSET="uv-x86_64-unknown-linux-gnu.tar.gz"
export PATH="$HOME/.local/bin:$PATH"
# Same interpreter build on every host: uv-managed CPython at the exact version in .python-version,
# never whatever the image ships (the RunPod image's system 3.12.3 would otherwise be picked).
export UV_PYTHON_PREFERENCE=only-managed

if [[ "$(uv --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]]; then
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  base="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
  curl -fsSL --retry 3 -o "$tmp/$UV_ASSET" "$base/$UV_ASSET"
  curl -fsSL --retry 3 -o "$tmp/$UV_ASSET.sha256" "$base/$UV_ASSET.sha256"
  (cd "$tmp" && sha256sum -c "$UV_ASSET.sha256")
  tar -xzf "$tmp/$UV_ASSET" -C "$tmp"
  mkdir -p "$HOME/.local/bin"
  install -m 0755 "$tmp/uv-x86_64-unknown-linux-gnu/uv" "$tmp/uv-x86_64-unknown-linux-gnu/uvx" "$HOME/.local/bin/"
fi
uv --version

# Persistent storage (profiles that set DLI_PERSIST_DIR, e.g. a RunPod network volume): results,
# logs and the Hugging Face cache live there and survive a pod restart; the repo and the Python
# environment stay on fast local disk (a network filesystem is slow for thousands of small files),
# and are rebuilt by this script after a restart.
if [[ -n "${DLI_PERSIST_DIR:-}" ]]; then
  mkdir -p "$DLI_PERSIST_DIR/results" "$DLI_PERSIST_DIR/logs" "$DLI_PERSIST_DIR/hf"
  if [[ -d benchmarks/results && ! -L benchmarks/results ]]; then
    # keep anything produced locally before persistence was configured, never overwrite
    rsync -a --ignore-existing benchmarks/results/ "$DLI_PERSIST_DIR/results/" && rm -rf benchmarks/results
  fi
  ln -sfn "$DLI_PERSIST_DIR/results" benchmarks/results
  echo "persistent results: benchmarks/results -> $DLI_PERSIST_DIR/results; HF cache: ${HF_HOME:-unset}"
fi

# --frozen: fail instead of re-resolving, so the host runs exactly what was locked and reviewed.
uv sync --frozen --extra cu130
uv run --frozen python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"

if ! command -v gcc >/dev/null && ! command -v cc >/dev/null; then
  echo "WARNING: no C compiler found. Triton JIT needs one at runtime (apt package build-essential)."
  echo "         Installing it is a system change and requires owner approval; see docs/known-issues.md."
fi
