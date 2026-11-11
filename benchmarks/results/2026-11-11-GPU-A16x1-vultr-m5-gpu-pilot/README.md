# Calibration pilot (not evidence)

One repetition, closed loop only, run to calibrate `benchmarks/plans/m5-gpu.json`:

1. every backend and model starts on the A16 (they did);
2. vLLM's KV-cache capacity at `--gpu-memory-utilization 0.85`, from its startup logs (`server-logs/vllm-*.log`):
   Qwen2.5-0.5B 946,960 tokens, Qwen2.5-1.5B 321,248, SmolLM2-1.7B 47,440, Qwen2.5-7B-AWQ 105,360.
   The main campaign gives the native engine the same number of 16-token blocks;
3. throughput ranges for the open-loop rate grids.

A single repetition is not a measurement under `ENGINEERING_STANDARDS` §7; use the main campaign's results.
