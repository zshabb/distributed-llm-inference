#!/usr/bin/env bash
# Run one bounded command in the synced repo on the GPU host, then pull results back.
#   [GPU_PROFILE=<name>] scripts/gpu/remote.sh <command...>     (GPU_TIMEOUT seconds, default 900)
# The profile's GPU_REMOTE_ENV is applied first; CPU thread pools are capped to the container's
# real CPU quota (cpu_threads.sh) unless the profile sets OMP_NUM_THREADS itself.
# Prints session start/end/duration and an estimated cost for PROJECT_STATUS.md bookkeeping.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
gpu_preflight
[[ $# -gt 0 ]] || { echo "usage: $0 <command...>" >&2; exit 64; }

timeout_s="${GPU_TIMEOUT:-900}"
start_epoch=$(date -u +%s)
echo "[gpu-session] start=$(date -u +%FT%TZ) host=$GPU_HOST label=$GPU_RESULT_LABEL timeout=${timeout_s}s cmd=$*"

# printf %q quoting keeps arguments intact across the remote shell.
remote_env="export PATH=\"\$HOME/.local/bin:\$PATH\" GPU_RESULT_LABEL=$(printf %q "$GPU_RESULT_LABEL"); ${GPU_REMOTE_ENV}
: \"\${OMP_NUM_THREADS:=\$(bash scripts/gpu/cpu_threads.sh)}\"
export OMP_NUM_THREADS MKL_NUM_THREADS=\"\${MKL_NUM_THREADS:-\$OMP_NUM_THREADS}\" OPENBLAS_NUM_THREADS=\"\${OPENBLAS_NUM_THREADS:-\$OMP_NUM_THREADS}\""
remote_cmd="cd '$GPU_REMOTE_DIR' && $remote_env
timeout --signal=TERM --kill-after=30 $timeout_s $(printf '%q ' "$@")"
set +e
gpu_ssh "$remote_cmd"
rc=$?
set -e

# Committed results are never overwritten by the host's copies (see pull_results.sh).
RSYNC_RSH="ssh ${SSH_OPTS[*]}" "$GPU_SCRIPT_DIR/pull_results.sh" \
  "$GPU_HOST:$GPU_RESULTS_DIR" "$REPO_ROOT" || echo "warning: pulling results failed" >&2

end_epoch=$(date -u +%s)
dur=$((end_epoch - start_epoch))
cost=$(awk -v s="$dur" -v r="$GPU_HOURLY_USD" 'BEGIN { printf "%.3f", s / 3600 * r }')
echo "[gpu-session] end=$(date -u +%FT%TZ) duration=${dur}s exit=$rc est_cost_usd=$cost (incremental share of standing cost)"
exit "$rc"
