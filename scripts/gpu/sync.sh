#!/usr/bin/env bash
# Push the working tree (tracked + untracked-but-not-ignored files) to the GPU host.
# The host receives source only: no .git directory, no credentials, no ignored files.
# A REVISION file records exactly which code ran, so remote results stay attributable.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
gpu_preflight

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cd "$REPO_ROOT"
git ls-files -z --cached --others --exclude-standard | rsync -a --from0 --files-from=- ./ "$stage/"
git describe --always --dirty --abbrev=12 > "$stage/REVISION"

gpu_ssh "mkdir -p '$GPU_REMOTE_DIR'"
# --delete keeps the remote tree identical to the local one; .venv and results produced on the host
# are excluded so a sync never destroys the environment or unsaved measurements.
gpu_rsync -az --delete --exclude '.venv/' --exclude 'benchmarks/results/' "$stage/" "$GPU_HOST:$GPU_REMOTE_DIR/"
echo "synced $(cat "$stage/REVISION") -> $GPU_HOST:$GPU_REMOTE_DIR"
