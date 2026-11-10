# distributed-llm-inference: instructions for Claude Code

This repository is intended to be **public**. Never commit host addresses, credentials, Terraform state, or raw host
inspection output (`scripts/gpu/host.env` and `benchmarks/results/**/inspect.txt` are git-ignored on purpose).

## Read first
`docs/progress.md` → `docs/known-issues.md` → `docs/implementation-plan.md` → `ENGINEERING_GUIDE.md`.
Portfolio-wide rules: `ENGINEERING_STANDARDS.md` in the local planning workspace (Git §4, tests §6.1, security §8.1, GPU §14).

## Commands (run inside WSL2 from the repo root; `make` may be unavailable)
| Purpose | Command | Tier / budget |
|---|---|---|
| Env | `uv sync --frozen --extra cpu` (GPU host: `--extra cu130`) | — |
| Lint + types | `uv run --frozen ruff check . && uv run --frozen ruff format --check . && uv run --frozen mypy` | fast |
| Fast tests | `uv run --frozen pytest -q` | fast, < 60 s |
| Stress / soak | `uv run --frozen pytest -q -m stress` | focused/broad, minutes |
| Real-checkpoint parity | `uv run --frozen pytest -m parity -s` | comprehensive (~70 min CPU): milestone/release only |
| GPU host | `SYNC_FROM=head scripts/gpu/sync.sh && scripts/gpu/remote.sh <cmd>` | see `docs/operations.md` |

## Rules specific to this repo
- Work on a `feat/*` branch per milestone; merge to `main` with `--no-ff` and a PR-template message.
- Do not rerun the M1 parity suites unless `src/dli/model/`, `src/dli/generate.py`, `uv.lock`, or the parity harness change
  in a way that affects numerics. Record the decision in `docs/progress.md`.
- Evidence directories: `benchmarks/results/<UTC date>-<label>-<name>/` with `env.json` whose revision is captured at
  run start (`envinfo.current_revision`). Never edit raw results; add a correction note beside them instead.
- Parity results are evidence for the tested models, prompts, precision modes, and greedy settings only.
