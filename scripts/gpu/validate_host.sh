#!/usr/bin/env bash
# Bounded GPU environment validation. Runs ON the host from the synced repo directory.
# Produces benchmarks/results/<UTC date>-GPU-A16x1-vultr-env-validation/ with:
#   inspect.txt  env.json  smoke.json  container.txt  summary.txt
# remote.sh pulls the directory back to the laptop, where it is reviewed and committed.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
label="${GPU_RESULT_LABEL:-GPU-A16x1-vultr}"
out="benchmarks/results/$(date -u +%Y-%m-%d)-${label}-env-validation"
mkdir -p "$out"
status=0

bash scripts/gpu/inspect_host.sh > "$out/inspect.txt" 2>&1
uv run --frozen python -m dli.envinfo "$out/env.json" || status=1
uv run --frozen python scripts/gpu/gpu_smoke.py "$out/smoke.json" > /dev/null || status=1

{
  if command -v docker >/dev/null && dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    # Image tag pinned; matches the host's CUDA 13.0 driver capability.
    docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu22.04 \
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
      && echo "container GPU access: OK" || { echo "container GPU access: FAILED"; exit 1; }
  else
    echo "container GPU access: SKIPPED (docker or nvidia-container-toolkit not installed)"
  fi
} > "$out/container.txt" 2>&1 || status=1

{
  echo "revision: $(cat REVISION 2>/dev/null || echo unknown)"
  echo "smoke ok: $(python3 -c "import json;print(json.load(open('$out/smoke.json'))['ok'])" 2>/dev/null || echo unknown)"
  tail -1 "$out/container.txt"
  echo "overall exit status: $status"
} | tee "$out/summary.txt"
exit "$status"
