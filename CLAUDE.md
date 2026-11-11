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
- **No AI co-author trailers:** never add `Co-Authored-By: Claude` (or any automatic AI attribution) to commit messages.
  Commits use the configured identity `Zeav <95062713+zshabb@users.noreply.github.com>`. Earlier published commits that
  carry the trailer are left as they are: no amend, rewrite, or force-push.
- Git history is the engineering journal reviewers will read: commit each validated, independently understandable
  improvement separately with a Conventional Commit message explaining what and why (e.g.
  `fix(memory): prevent block leaks during request cancellation`). No bundling unrelated work, no trivial splitting, no
  manufactured activity. Local commits need no approval.
- **Published** (public, approved 2026-11-10): https://github.com/zshabb/distributed-llm-inference. Push validated
  progress regularly during development (feature branch pushes after validated commits), open PRs, and merge
  passing milestone branches with `--no-ff`, all without asking. Run the pre-push checks every time (fast suite, lint,
  types, `gitleaks git` on the pushed range, and `git grep -nIE "/home/|/Users/|<host IP>"` for personal paths and
  infrastructure details; tool output such as profiles must be path-redacted before committing). Never force-push, rewrite published history, delete remote branches, or change visibility.
- Push from WSL with the Windows GitHub CLI as a one-off credential helper (gh is not logged in inside WSL):
  `git -c credential.helper= -c 'credential.helper=!"/mnt/c/Program Files/GitHub CLI/gh.exe" auth git-credential' push`
- Do not rerun the M1 parity suites unless `src/dli/model/`, `src/dli/generate.py`, `uv.lock`, or the parity harness change
  in a way that affects numerics. Record the decision in `docs/progress.md`.
- **GPU-first testing** (owner policy 2026-11-11, portfolio standards §6.2): CPU runs correctness (unit tests, fp32
  parity, scheduler/block-manager properties, memory safety, API/e2e, targeted regressions); performance is measured
  on the A16. No CPU performance campaigns unless an acceptance criterion names CPU, and then only the smallest
  focused measurement (flag it). Targeted tests while developing (`pytest <path> -k ...`), one fast-suite run before a
  push, comprehensive suites and GPU campaigns at milestone checkpoints. Targets and unfavourable results are kept.
- Evidence directories: `benchmarks/results/<UTC date>-<label>-<name>/` with `env.json` whose revision is captured at
  run start (`envinfo.current_revision`). Never edit raw results; add a correction note beside them instead.
- Parity results are evidence for the tested models, prompts, precision modes, and greedy settings only.
