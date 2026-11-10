# Implementation plan

Source of truth for scope: portfolio `specs/01-distributed-llm-inference.md` and `PORTFOLIO_ROADMAP.md` §3.
Validation tiers: T0 CPU · T1 simulated · T2 single GPU (Vultr A16) · T3 multi-GPU (rental) · T4 multi-node.

| M | Milestone | Tiers | Status |
|---|---|---|---|
| M0 | Bootstrap: tooling, CI, ADRs, contract v1, GPU host scripts, environment validation | T0 · T2 smoke | in progress |
| M1 | Native Llama-family model + HF weight loading; greedy parity with `transformers` | T0 (fp32 exact) · T2 (bf16 tolerance) | in progress |
| M2 | Paged KV-cache block manager + prefix sharing | T0 | not started |
| M3 | Continuous batching, chunked prefill, preemption, cancellation | T0 · T2 | not started |
| M4 | OpenAI API server, SSE, admission control, backpressure, metrics | T0 · T2 | not started |
| M5 | Benchmark harness + baselines (HF static batching; vLLM on A16) | T0 · T2 (Gate B) | not started |
| M6 | Triton paged-attention decode kernel | T0 interpreter · T2 | not started |
| M7 | Tensor parallelism | T0 gloo · T3 rental | not started |
| M8 | Go router | T0 · K8S-KIND · T2 (logic) | not started |
| M9 | Helm/K8s, HPA, failure injection | K8S-KIND | not started |
| M10 | Rental validation (T3, FP8, 80 GB) | Gate G | not started |
| M11 | Case studies, hardening, docs, review, RC | — | not started |

## M0 acceptance

- [x] `uv.lock` pins CPU and CUDA builds; `ruff`, `mypy --strict`, `pytest` pass locally
- [x] pre-commit: format, lint, gitleaks, large-file guard
- [x] ADR-0001 (architecture), ADR-0002 (PyTorch pinning)
- [ ] OpenAPI contract `contracts/inference/v1` validated by a test
- [ ] CI workflows (lint, types, tests, security scan). They cannot run until the repository is published
- [x] GPU host scripts (sync, remote run, bootstrap, Docker setup, inspection, validation) + runbook
- [ ] GPU host validated: CUDA torch tensor check + container GPU access, results committed

## M1 acceptance (spec AC1, M1 scope)

- [ ] `ModelConfig` parses Llama and Qwen2 configs, rejects unsupported features
- [ ] Logits match `transformers` on randomly initialized tiny Llama/Qwen2 models (fp32, atol ≤ 1e-4), incl. tied/untied
      embeddings and llama3 RoPE scaling. Fast, no downloads
- [ ] Safetensors loader with strict name/shape checks; pinned model revisions
- [ ] Greedy generation with a KV cache equals greedy generation without one (recompute) on the same model
- [ ] **Parity suite:** 100 fixed prompts × 128 greedy tokens, token-identical to `transformers` fp32 on CPU for
      SmolLM2-135M-Instruct and Qwen2.5-0.5B-Instruct `[CPU-LAPTOP]`
- [ ] bf16 GPU check on the A16 with the spec's tolerance rule `[GPU:A16x1@vultr]`
- [ ] ENGINEERING_GUIDE.md, QnA.md (M1 scope), README updated with labeled results
