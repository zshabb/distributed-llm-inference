#!/usr/bin/env bash
# Shared settings for scripts that run LOCALLY (WSL2) and drive the remote GPU host over SSH.
#
# Host selection (provider-independent):
#   GPU_PROFILE=<name>  loads scripts/gpu/hosts/<name>.env (untracked; see hosts/*.env.example)
#   (unset)             loads scripts/gpu/host.env (untracked), the original single-host file
# A profile describes one machine: how to reach it, where code/caches/results live on it, its
# hourly price, evidence label, and capabilities. Scripts branch on capabilities, never on the
# provider's name, so adding a provider means writing one profile.
#
# Security model (see docs/operations.md):
#   * The private key never leaves the laptop. We authenticate through a local ssh-agent and pass
#     only the PUBLIC key to `-i` (or an ssh_config alias with IdentitiesOnly) so ssh picks the
#     matching agent identity.
#   * Agent forwarding is explicitly disabled: a compromised GPU host must not be able to use the key.
#   * Host keys are verified (StrictHostKeyChecking=yes); never trust-on-first-use in scripts.
#   * No credentials, tokens, or Git remotes are ever copied to the host.
set -euo pipefail

GPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GPU_SCRIPT_DIR/../.." && pwd)"
GPU_PROFILE_DIR="${GPU_PROFILE_DIR:-$GPU_SCRIPT_DIR/hosts}"

if [[ -n "${GPU_PROFILE:-}" ]]; then
  profile="$GPU_PROFILE_DIR/$GPU_PROFILE.env"
  [[ -f "$profile" ]] || { echo "error: no host profile $profile (copy hosts/<name>.env.example)" >&2; exit 64; }
  # shellcheck source=/dev/null
  source "$profile"
elif [[ -f "$GPU_SCRIPT_DIR/host.env" ]]; then
  # shellcheck source=/dev/null
  source "$GPU_SCRIPT_DIR/host.env"
fi
: "${GPU_HOST:?set GPU_HOST (user@address, or an ssh_config alias) in the host profile}"
: "${GPU_SSH_PUBKEY:=}"                                   # empty: identity comes from ssh_config
: "${GPU_REMOTE_DIR:=work/distributed-llm-inference}"   # relative to the remote $HOME
: "${GPU_RESULTS_DIR:=$GPU_REMOTE_DIR/benchmarks/results}"  # where the host keeps results
: "${GPU_LABEL:=GPU:A16x1@vultr}"                       # evidence label as written in docs
: "${GPU_RESULT_LABEL:=GPU-A16x1-vultr}"                # path-safe form used in result directory names
: "${GPU_HOURLY_USD:=0.471}"
: "${GPU_REMOTE_ENV:=}"                                  # shell exports applied before remote commands
: "${SSH_AUTH_SOCK:=/run/user/$(id -u)/openssh_agent}"
export SSH_AUTH_SOCK GPU_LABEL GPU_RESULT_LABEL

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -o IdentitiesOnly=yes
  -o ForwardAgent=no
  -o ForwardX11=no
  -o StrictHostKeyChecking=yes
)
if [[ -n "$GPU_SSH_PUBKEY" ]]; then
  SSH_OPTS+=(-i "$GPU_SSH_PUBKEY")
fi

gpu_preflight() {
  if ! ssh-add -l >/dev/null 2>&1; then
    echo "error: no identities in ssh-agent at $SSH_AUTH_SOCK." >&2
    echo "Load the host's key yourself (the passphrase is never handled by these scripts):" >&2
    echo "  SSH_AUTH_SOCK=$SSH_AUTH_SOCK ssh-add -t 8h ~/.ssh/<key>" >&2
    exit 2
  fi
}

gpu_ssh() { ssh "${SSH_OPTS[@]}" "$GPU_HOST" "$@"; }
gpu_rsync() { rsync -e "ssh ${SSH_OPTS[*]}" "$@"; }
