#!/usr/bin/env bash
# Idempotent USER-LEVEL setup on the GPU host. Runs on the host from the synced repo directory.
#   * installs a pinned, checksum-verified uv into ~/.local/bin (no apt, no system Python changes)
#   * installs the pinned Python and syncs the locked environment with the CUDA 13.0 torch build
# Re-running is safe and fast; the environment is fully determined by uv.lock.
set -euo pipefail
UV_VERSION="0.13.5"   # keep in sync with docs/operations.md
UV_ASSET="uv-x86_64-unknown-linux-gnu.tar.gz"
export PATH="$HOME/.local/bin:$PATH"

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

# --frozen: fail instead of re-resolving, so the host runs exactly what was locked and reviewed.
uv sync --frozen --extra cu130
uv run --frozen python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"

if ! command -v gcc >/dev/null && ! command -v cc >/dev/null; then
  echo "WARNING: no C compiler found. Triton JIT needs one at runtime (apt package build-essential)."
  echo "         Installing it is a system change and requires owner approval; see docs/known-issues.md."
fi
