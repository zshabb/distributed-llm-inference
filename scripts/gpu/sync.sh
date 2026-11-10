#!/usr/bin/env bash
# Push code to the GPU host. The host receives source only: no .git directory, no credentials,
# no ignored files. A REVISION file records exactly which code ran, so remote results stay attributable.
#
#   SYNC_FROM=worktree (default)  tracked + untracked-but-not-ignored files, for development iteration;
#                                 REVISION carries "-dirty" if anything is uncommitted.
#   SYNC_FROM=head                exactly the committed HEAD (git archive). Use this for every run whose
#                                 results are kept as evidence, so they map to a precise commit.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
gpu_preflight

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cd "$REPO_ROOT"
case "${SYNC_FROM:-worktree}" in
  head)
    git archive --format=tar HEAD | tar -x -C "$stage"
    git rev-parse --short=12 HEAD > "$stage/REVISION"
    ;;
  worktree)
    git ls-files -z --cached --others --exclude-standard | rsync -a --from0 --files-from=- ./ "$stage/"
    git describe --always --dirty --abbrev=12 > "$stage/REVISION"
    ;;
  *) echo "SYNC_FROM must be 'head' or 'worktree'" >&2; exit 64 ;;
esac

gpu_ssh "mkdir -p '$GPU_REMOTE_DIR'"
# --delete keeps the remote tree identical to the local one; .venv and results produced on the host
# are excluded so a sync never destroys the environment or unsaved measurements.
gpu_rsync -az --delete --exclude '.venv/' --exclude 'benchmarks/results/' "$stage/" "$GPU_HOST:$GPU_REMOTE_DIR/"
echo "synced $(cat "$stage/REVISION") (${SYNC_FROM:-worktree}) -> $GPU_HOST:$GPU_REMOTE_DIR"
