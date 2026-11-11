# Engine step profile (M5 bottleneck evidence)

`benchmarks/scripts/engine_profile.py`, Qwen2.5-0.5B bf16 on the A16, revision `8e790c1`: torch.profiler over 20
steady-state decode steps at decode batch 16 and 64 (input 256, output 128). Raw: `Qwen2.5-0.5B-Instruct.json`.

The profiler itself adds overhead (profiled step 517 ms at batch 64 vs 217 ms TPOT measured unprofiled over HTTP),
so compare ratios and counts, not absolute times. Interpretation: `../2026-11-11-GPU-A16x1-vultr-m5-gpu/ANALYSIS.md` §1.
