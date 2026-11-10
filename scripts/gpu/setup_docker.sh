#!/usr/bin/env bash
# SYSTEM-LEVEL: Docker Engine + NVIDIA Container Toolkit on the GPU host (owner-approved 2026-11-10).
# Runs on the host as root. Idempotent and conservative:
#   * inspects first; never replaces an existing Docker/containerd/toolkit installation
#   * never touches sshd, ufw/iptables rules we did not create, or the cloud firewall
#   * published container ports default to 127.0.0.1 so nothing is exposed publicly by accident
#   * the Docker API stays on the local unix socket only (no TCP listener)
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "must run as root" >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
. /etc/os-release

echo "--- pre-state"
command -v docker && docker --version || echo "docker: absent"
dpkg -s nvidia-container-toolkit >/dev/null 2>&1 && echo "nvidia-container-toolkit: present" || echo "nvidia-container-toolkit: absent"
for pkg in docker.io podman containerd; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "found distro package '$pkg': not replacing it automatically. Resolve manually." >&2
    exit 3
  fi
done

if ! command -v docker >/dev/null; then
  echo "--- installing Docker Engine from download.docker.com ($VERSION_CODENAME)"
  apt-get update -qq
  apt-get install -y -qq ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin
fi

if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  echo "--- installing NVIDIA Container Toolkit from nvidia.github.io"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit
fi

# Default bind address for published ports. Only written if no daemon.json exists (never overwrite).
if [[ ! -f /etc/docker/daemon.json ]]; then
  echo '{ "ip": "127.0.0.1" }' > /etc/docker/daemon.json
fi
nvidia-ctk runtime configure --runtime=docker   # merges the nvidia runtime into daemon.json
systemctl enable --now docker >/dev/null
systemctl restart docker

echo "--- post-state"
docker --version
nvidia-ctk --version | head -1
cat /etc/docker/daemon.json
if ss -tln | grep -qE ':(2375|2376)\b'; then echo "ERROR: Docker API listening on TCP" >&2; exit 4; fi
echo "docker API: unix socket only (no TCP listener)"
