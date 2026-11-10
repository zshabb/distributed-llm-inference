#!/usr/bin/env bash
# Shared settings for scripts that run LOCALLY (WSL2) and drive the remote GPU host over SSH.
#
# Security model (see docs/operations.md):
#   * The private key never leaves the laptop. We authenticate through a local ssh-agent and pass
#     only the PUBLIC key to `-i` so ssh picks the matching agent identity (IdentitiesOnly=yes).
#   * Agent forwarding is explicitly disabled: a compromised GPU host must not be able to use the key.
#   * No credentials, tokens, or Git remotes are ever copied to the host.
set -euo pipefail

GPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GPU_SCRIPT_DIR/../.." && pwd)"

# host.env is untracked (it contains the host address). Copy host.env.example to create it.
if [[ -f "$GPU_SCRIPT_DIR/host.env" ]]; then
  # shellcheck source=/dev/null
  source "$GPU_SCRIPT_DIR/host.env"
fi
: "${GPU_HOST:?set GPU_HOST (e.g. root@203.0.113.10) in scripts/gpu/host.env}"
: "${GPU_SSH_PUBKEY:?set GPU_SSH_PUBKEY (public key matching the agent identity) in scripts/gpu/host.env}"
: "${GPU_REMOTE_DIR:=work/distributed-llm-inference}"   # relative to the remote $HOME
: "${SSH_AUTH_SOCK:=/run/user/$(id -u)/openssh_agent}"
export SSH_AUTH_SOCK

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -o IdentitiesOnly=yes
  -o ForwardAgent=no
  -o ForwardX11=no
  -o StrictHostKeyChecking=yes
  -i "$GPU_SSH_PUBKEY"
)

gpu_preflight() {
  if ! ssh-add -l >/dev/null 2>&1; then
    echo "error: no identities in ssh-agent at $SSH_AUTH_SOCK." >&2
    echo "Load the key yourself (the passphrase is never handled by these scripts):" >&2
    echo "  SSH_AUTH_SOCK=$SSH_AUTH_SOCK ssh-add -t 8h ${GPU_SSH_PUBKEY%.pub}" >&2
    exit 2
  fi
}

gpu_ssh() { ssh "${SSH_OPTS[@]}" "$GPU_HOST" "$@"; }
gpu_rsync() { rsync -e "ssh ${SSH_OPTS[*]}" "$@"; }
