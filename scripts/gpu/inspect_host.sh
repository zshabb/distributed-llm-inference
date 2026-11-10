#!/usr/bin/env bash
# READ-ONLY inspection of the GPU host. Runs on the host. Changes nothing.
set -uo pipefail
section() { echo; echo "=== $*"; }

section "identity / OS"
whoami; hostname; grep PRETTY_NAME /etc/os-release; uname -r; echo "booted: $(uptime -s)"
section "CPU / memory / disk"
nproc; lscpu | grep -E "Model name|Hypervisor"; free -g | head -2; df -h / | tail -1
section "GPU"
nvidia-smi || echo "nvidia-smi FAILED"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,compute_cap,driver_version,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,power.limit --format=csv 2>&1
# Virtualization/licensing matter: an unlicensed vGPU is throttled after a grace period,
# which would silently invalidate benchmarks.
nvidia-smi -q 2>/dev/null | grep -iE "Product Name|Product Architecture|Virtualization Mode|Host VGPU Mode|vGPU Software Licensed|License Status|MIG Mode|Persistence Mode|Compute Mode|ECC Mode" -A1 | sed '/^--$/d'
section "CUDA toolkit / compilers (Triton needs a host C compiler at runtime)"
ls -d /usr/local/cuda* 2>/dev/null || echo "no /usr/local/cuda* (fine: PyTorch wheels bundle the CUDA runtime)"
for t in gcc cc nvcc; do printf "%s: %s\n" "$t" "$(command -v $t || echo missing)"; done
section "Python / uv"
python3 --version; command -v "$HOME/.local/bin/uv" && "$HOME/.local/bin/uv" --version || echo "uv: not installed"
section "containers"
command -v docker && docker --version || echo "docker: not installed"
dpkg -l 2>/dev/null | awk '/^ii/ && /nvidia-container|docker|containerd|podman/ {print $2, $3}'
[ -f /etc/docker/daemon.json ] && cat /etc/docker/daemon.json
section "network exposure (listening sockets) and host firewall (inspect only, never modified)"
ss -tlnp | awk 'NR==1 || /LISTEN/'
(command -v ufw >/dev/null && ufw status verbose) 2>/dev/null | head -12 || echo "ufw: not present"
section "sshd effective security settings (inspect only, never modified)"
sshd -T 2>/dev/null | grep -iE "^(passwordauthentication|permitrootlogin|pubkeyauthentication|kbdinteractiveauthentication) "
