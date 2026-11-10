# Contributing

Portfolio-wide rules live in `ENGINEERING_STANDARDS.md` (portfolio planning repository). Repository-specific essentials:

## Environment
```bash
uv sync --frozen --extra cpu        # laptop / CI     (GPU host: --extra cu130, see docs/operations.md)
uv run pre-commit install
```

## Before every commit
```bash
uv run --frozen ruff check . && uv run --frozen ruff format --check .
uv run --frozen mypy
uv run --frozen pytest -q            # fast suite
uv run --frozen pytest -q -m parity  # when touching model math (downloads pinned weights)
```

## Rules
- Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `ci`, `security`, `build`, `chore`), one coherent change per commit.
- Bug fixes start with a failing regression test.
- Every reported number carries an evidence label (`[CPU-LAPTOP]`, `[GPU:A16x1@vultr]`, `[SIM]`, …) and a result
  directory with `env.json`. A single A16 is never evidence of multi-GPU behavior.
- `perf:` commits include before/after measurements and a case study in `docs/engineering-evolution/`.
- Model weights and datasets are downloaded by scripts at pinned revisions, never committed.
- Documentation changes ship in the same commit as the behavior they describe.
