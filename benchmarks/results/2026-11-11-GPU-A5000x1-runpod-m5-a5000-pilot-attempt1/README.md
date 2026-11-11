# Calibration pilot, attempt 1 (failed, kept for the record)

Calibration only, not evidence. The native engine (Qwen2.5-0.5B, closed loop c=16/64) ran; vLLM then failed at
start-up: on this GPU (64 SMs) vLLM 0.32.0 switches to FlashInfer sampling, whose kernel it could not JIT-compile in the
pinned venv (no CUDA toolkit or ninja). See `server-logs/vllm-*.log` and `benchmarks/baselines/vllm/README.md`.
Attempt 2 (`../2026-11-11-GPU-A5000x1-runpod-m5-a5000-pilot/`) sets `VLLM_USE_FLASHINFER_SAMPLER=0`, the sampler vLLM
used in every A16 run.
