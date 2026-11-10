# Provenance correction for this result set

**Code that actually ran: commit `d676e49`** ("test(parity): add real-checkpoint greedy parity suite…").

`env.json` in this directory reports `revision: 4848e8a935d0-dirty`. That value is **wrong** and is kept unedited
as raw output. The run started at 2026-11-10T19:44:04Z from `d676e49` (pytest imports all code at collection time),
but `dli.envinfo` read `git describe` when it wrote `env.json` at 20:54:29Z, after unrelated commits had moved HEAD.
The bug is fixed in `46f6359` (revision captured at run start) with a regression test.

Why the measurement itself is unaffected:

```bash
git diff --stat d676e49 46f6359^ -- src tests pyproject.toml uv.lock   # empty: no code under test changed
```

The commits between the start and `46f6359` changed only `scripts/gpu/`, `docs/`, `.gitignore`, and committed GPU
results. Even the later commits that touched `src/dli/envinfo.py` and `tests/` cannot affect this run, because the
process had already imported its modules at 19:44Z.

| Field | Value |
|---|---|
| Label | `[CPU-LAPTOP]` (i7-1355U, WSL2 Ubuntu 26.04, 15.5 GiB, torch 2.14.1+cpu, 6 threads) |
| Duration | 19:44:04Z → 20:54:31Z (4,225 s for both models, sequential) |
| Result | SmolLM2-135M-Instruct 100/100 identical · Qwen2.5-0.5B-Instruct 100/100 identical (128 tokens each) |
