# Known issues and limitations

| # | Issue | Impact | Status / plan |
|---|---|---|---|
| 1 | CI workflows exist but have never run: the repository is local-only until publication is approved | Lint/type/test gates are enforced by pre-commit and local runs only | Will run on first push (Gate P) |
| 2 | `make` is not installed in the WSL2 dev environment (needs `sudo apt install make`, owner action) | `Makefile` targets unusable locally; the underlying `uv run …` commands are used directly | Open |
| 3 | No C compiler in WSL2. Triton's runtime needs one on any machine that JIT-compiles kernels | No impact on CPU paths. The GPU host has gcc and Triton JIT is verified there | Open (laptop only) |
| 4 | M1 generation uses a contiguous per-sequence KV cache and greedy decoding only | Not the serving path; exists for numerical parity. Paged KV (M2) and sampling (M3/M4) replace it | By design |
| 5 | Llama-3.x weights are gated (HF token required, never placed on the GPU host) | The Llama architecture is verified with real SmolLM2 weights (`model_type: llama`), but Llama-3 RoPE scaling only against random-weight reference models | Revisit if the owner approves a token for local use |
| 6 | Provenance bug: `env.json` recorded the revision at write time, not run start | First fp32 CPU parity `env.json` shows a wrong revision; measurement unaffected (verified by `git diff`) | **Fixed** in `46f6359` + regression test; correction in the result's `PROVENANCE.md` |
| 7 | `sync.sh` prints "cannot delete non-empty directory: benchmarks" in HEAD mode | Cosmetic: rsync declines to delete the remote `benchmarks/` because its excluded `results/` subtree is protected | Accepted |
| 8 | `setup_docker.sh` install path (Docker absent) has never run: the Vultr image shipped Docker + toolkit | Rebuild on a different image could hit an unexercised path | Open; exercise on next rebuild |
| 9 | Parity covers prompts ≤ 53 tokens + 128 generated (≤ 181 positions) | Long-context behavior unverified on real weights | Extend in M3 when chunked prefill lands |
