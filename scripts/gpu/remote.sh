#!/usr/bin/env bash
# Run one bounded command in the synced repo on the GPU host, then pull results back.
#   scripts/gpu/remote.sh <command...>        (GPU_TIMEOUT seconds, default 900)
# Prints session start/end/duration and an estimated cost for PROJECT_STATUS.md bookkeeping.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
gpu_preflight
[[ $# -gt 0 ]] || { echo "usage: $0 <command...>" >&2; exit 64; }

timeout_s="${GPU_TIMEOUT:-900}"
start_epoch=$(date -u +%s)
echo "[gpu-session] start=$(date -u +%FT%TZ) host=$GPU_HOST timeout=${timeout_s}s cmd=$*"

# printf %q quoting keeps arguments intact across the remote shell.
remote_cmd="cd '$GPU_REMOTE_DIR' && export PATH=\"\$HOME/.local/bin:\$PATH\" && timeout --signal=TERM --kill-after=30 $timeout_s $(printf '%q ' "$@")"
set +e
gpu_ssh "$remote_cmd"
rc=$?
set -e

mkdir -p "$REPO_ROOT/benchmarks/results"
gpu_rsync -az "$GPU_HOST:$GPU_REMOTE_DIR/benchmarks/results/" "$REPO_ROOT/benchmarks/results/" 2>/dev/null || true

end_epoch=$(date -u +%s)
dur=$((end_epoch - start_epoch))
cost=$(awk -v s="$dur" -v r="${GPU_HOURLY_USD:-0.471}" 'BEGIN { printf "%.3f", s / 3600 * r }')
echo "[gpu-session] end=$(date -u +%FT%TZ) duration=${dur}s exit=$rc est_cost_usd=$cost (incremental share of standing cost)"
exit "$rc"
