# A5000 calibration pilot, attempt 2 (not evidence)

One closed-loop repetition per backend and model on the RunPod RTX A5000 (revision 6bfa773), with vLLM from the
pinned venv and VLLM_USE_FLASHINFER_SAMPLER=0. Used to size the baseline: vLLM KV capacity at 0.85 of 24 GB was
1,606,656 / 604,496 / 88,848 tokens (Qwen2.5-0.5B / 1.5B / SmolLM2-1.7B); native pools are set to the same number of
16-token blocks in m5-a5000-baseline.json. A single repetition is not a measurement; see the baseline results.
