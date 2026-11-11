#!/usr/bin/env bash
# Idempotent setup of the vLLM baseline environment on hosts WITHOUT Docker. Runs on the host from
# the synced repo directory. Installs exactly the hash-pinned lock into its own venv (default
# ~/vllm-venv on local disk); the project environment (uv.lock) is untouched.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH" UV_PYTHON_PREFERENCE=only-managed
venv="${DLI_VLLM_VENV:-$HOME/vllm-venv}"
lock=benchmarks/baselines/vllm/requirements.txt

[[ -x "$venv/bin/python" ]] || uv venv --python "$(cat .python-version)" "$venv"
uv pip sync --python "$venv/bin/python" --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match "$lock"
"$venv/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available())"
