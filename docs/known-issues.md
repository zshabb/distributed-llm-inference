# Known issues and limitations

| # | Issue | Impact | Status / plan |
|---|---|---|---|
| 1 | CI workflows had never run before publication | — | **Resolved** 2026-11-10: first GitHub runs passed (ci: lint, mypy --strict, 95 tests, 54 s; security: full-history gitleaks + pip-audit of both torch builds, 16 s), runs 42907583097 / 42907583166 |
| 2 | `make` is not installed in the WSL2 dev environment (needs `sudo apt install make`, owner action) | `Makefile` targets unusable locally; the underlying `uv run …` commands are used directly | Open |
| 3 | No C compiler in WSL2. Triton's runtime needs one on any machine that JIT-compiles kernels | No impact on CPU paths. The GPU host has gcc and Triton JIT is verified there | Open (laptop only) |
| 4 | M1 generation uses a contiguous per-sequence KV cache and greedy decoding only | Not the serving path; exists for numerical parity. Paged KV (M2) and sampling (M3/M4) replace it | By design |
| 5 | Llama-3.x weights are gated (HF token required, never placed on the GPU host) | The Llama architecture is verified with real SmolLM2 weights (`model_type: llama`), but Llama-3 RoPE scaling only against random-weight reference models | Revisit if the owner approves a token for local use |
| 6 | Provenance bug: `env.json` recorded the revision at write time, not run start | First fp32 CPU parity `env.json` shows a wrong revision; measurement unaffected (verified by `git diff`) | **Fixed** in `46f6359` + regression test; correction in the result's `PROVENANCE.md` |
| 7 | `sync.sh` prints "cannot delete non-empty directory: benchmarks" in HEAD mode | Cosmetic: rsync declines to delete the remote `benchmarks/` because its excluded `results/` subtree is protected | Accepted |
| 8 | `setup_docker.sh` install path (Docker absent) has never run: the Vultr image shipped Docker + toolkit | Rebuild on a different image could hit an unexercised path | Open; exercise on next rebuild |
| 9 | Parity covers prompts ≤ 53 tokens + 128 generated (≤ 181 positions) | Long-context behavior unverified on real weights | Extend in M3 when chunked prefill lands |
| 10 | Complete GPU server recreation has never been performed | Recovery time and the full procedure are unverified; only the user-level bootstrap (25 s) and validation are measured | Open; perform and time on the next approved rebuild (operations.md §4) |
| 11 | Process deviation: M0–M1 were committed directly to `main` instead of a feature branch with a `--no-ff` merge | Review record is the commit history itself rather than a merge commit | Branch workflow applies from M2 |
| 12 | Paged KV read path gathers the whole prefix per layer per decode step | Paged decoding is slower than the contiguous cache on CPU (correctness reference only) | By design until the M6 Triton paged-attention kernel reads pages in place |
| 13 | Prefix sharing is block-granular (16 tokens by default) | Prefixes shorter than a block, and the partial tail of longer ones, are not shared | Accepted (ADR-0003); radix/partial-block sharing is a possible extension |
| 14 | Identical prompts admitted before either is computed keep private copies | Temporary duplicate memory (counted in `duplicate_registrations`) | Accepted; merge-on-register is a possible optimization |
| 15 | Eviction-policy results are trace simulations of hit rate only | No latency/TTFT evidence for the policies yet | Measure with the engine in M3/M5 |
