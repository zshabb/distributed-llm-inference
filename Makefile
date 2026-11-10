# Developer entry points. Every target is a thin wrapper so CI and humans run identical commands.
# TORCH selects the PyTorch build extra: cpu (laptop/CI) or cu130 (GPU host). See ADR-0002.
TORCH ?= cpu
UV ?= uv

.PHONY: dev lint format typecheck test test-cov parity check gpu-sync gpu-bootstrap gpu-validate

dev:            ## Create/refresh the locked environment
	$(UV) sync --frozen --extra $(TORCH)

lint:           ## Static checks (no changes)
	$(UV) run --frozen ruff check .
	$(UV) run --frozen ruff format --check .

format:         ## Apply formatting and safe lint fixes
	$(UV) run --frozen ruff format .
	$(UV) run --frozen ruff check --fix .

typecheck:
	$(UV) run --frozen mypy

test:           ## Fast tests (no downloads, no GPU)
	$(UV) run --frozen pytest -q

test-cov:
	$(UV) run --frozen pytest -q --cov --cov-report=term-missing

parity:         ## Slow numerical parity suite vs transformers (downloads pinned model revisions)
	$(UV) run --frozen pytest -q -m parity -s

check: lint typecheck test

# --- Remote GPU host (see docs/operations.md). Requires scripts/gpu/host.env and a loaded ssh-agent. ---
gpu-sync:
	scripts/gpu/sync.sh

gpu-bootstrap:  ## Idempotent user-level setup on the GPU host (uv, Python, locked cu130 env)
	scripts/gpu/sync.sh && scripts/gpu/remote.sh bash scripts/gpu/bootstrap_host.sh

gpu-validate:   ## Bounded GPU validation suite; results pulled into benchmarks/results/
	scripts/gpu/sync.sh && scripts/gpu/remote.sh bash scripts/gpu/validate_host.sh
