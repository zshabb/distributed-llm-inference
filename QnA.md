# QnA: implementation-specific technical questions

> **Scope: milestones M0–M1.** The release target is ≥ 30 questions across architecture, implementation,
> scalability, reliability, security, performance, debugging, and system design (master prompt §17).
> Questions about components that do not exist yet (scheduler, paged KV cache, API, kernels, TP, router) are
> added when those components land, because answering them now would describe unbuilt software.
> Every number carries an evidence label and a result directory.

## Architecture

### 1. Why write the model from scratch instead of loading it with `transformers`?
The engine's original components need to own the model's inner operations. Paged attention (M2/M6) replaces how
attention reads K/V, continuous batching (M3) changes the tensor layout to flattened tokens from many sequences, and
tensor parallelism (M7) replaces `nn.Linear` with sharded linears. With `transformers` each of those would be a
monkey-patch against an API we don't control. `src/dli/model/llama.py` keeps the checkpoint's parameter names so loading
stays a 1:1, strictly checked mapping (`src/dli/model/loader.py`). `transformers` remains a test-only dependency used as
the numerical reference (`pyproject.toml`, `[dependency-groups].dev`).

### 2. Why does `forward` take flattened 1-D `input_ids` and explicit `positions` instead of a `[batch, seq]` tensor?
Continuous batching mixes prefill chunks and single decode tokens from different sequences in one step, at different
absolute positions. A padded `[batch, seq]` layout wastes compute on padding and assumes aligned positions. The 1-D
layout with explicit positions (`LlamaForCausalLM.forward`, `rope_cos_sin`) is what the runner will feed in M3. Only the
attention metadata changes; the per-layer math (`DecoderLayer.forward`) stays the same.

### 3. Where are the component boundaries in the engine, and why that layering?
`api → scheduler → kv → runner → model → kernels` with downward-only dependencies (ADR-0001,
`docs/architecture.md`). The model layer has no notion of requests, and the scheduler never touches tensors. That keeps the
original algorithms (block manager, scheduler) testable as pure data structures with property tests, and the model
testable against a reference without a scheduler. That is how M1 was validated (`tests/unit/model/`).

### 4. Why one lockfile with two mutually exclusive PyTorch extras?
CPU (laptop/CI) and GPU (Vultr A16) runs must use the same library versions, or a CPU-vs-GPU comparison would be
confounded. `pyproject.toml` declares `cpu` and `cu130` extras as a uv conflict pair, so `uv.lock` pins `torch==2.14.1+cpu`
and `torch==2.14.1+cu130` together. 2.14.1 is the newest torch built for CUDA 13.0, which is exactly what the host
driver (580.95.05) reports. torch 2.15 ships only for CUDA 13.2+, and Triton JIT paths are where CUDA minor-version
compatibility breaks (ADR-0002).

## Implementation

### 5. How is RoPE implemented, and what goes wrong if the layout is wrong?
`src/dli/model/rope.py` uses the rotate-half layout: dimension *i* is paired with *i + d/2*. Angles are
`positions × inv_freq` computed in fp32, duplicated to the full head dimension, and cast to the activation dtype only at
the end. HF Llama/Qwen checkpoints were trained with this layout. Using the interleaved (adjacent-pair) layout instead
produces text that is grammatical for a few tokens and then drifts, which is why the tiny-model parity test also runs at
200 positions (`test_positions_beyond_original_context_match`).

### 6. How does Llama 3 RoPE scaling work in your code?
`inverse_frequencies` with `rope_type="llama3"` computes each frequency's wavelength. Wavelengths shorter than
`original_ctx / high_freq_factor` (high frequencies, local position detail) are unchanged. Wavelengths longer than
`original_ctx / low_freq_factor` are divided by `factor`. The band in between is linearly interpolated with
`smooth = (original_ctx/wavelen − low) / (high − low)`. The tiny-model test sets `original_max_position_embeddings=32`
and evaluates 200 positions so all three bands are exercised against the reference.

### 7. How is grouped-query attention handled?
Query head *h* reads KV head *h // fan_out*, with contiguous groups as stored in the checkpoints. `Attention.forward`
expands K/V with `repeat_interleave(fan_out, dim=0)` before SDPA. It is simple and matches the reference, but it
materializes `fan_out×` more K/V traffic (7× for Qwen2.5-0.5B). The M6 Triton kernel will index the shared KV head
directly. The `mha` tiny variant (fan-out 1) and the default variant (fan-out 4) are both parity-tested.

### 8. Why are there three attention branches?
In `Attention.forward`: (1) one new token needs no mask, because it may attend to every cached position; (2) a fresh
prompt with no cache uses `is_causal=True`; (3) a multi-token chunk appended to a non-empty cache needs an explicit
boolean mask `k_pos <= q_pos`, because SDPA's `is_causal` aligns the causal triangle to the top-left and would hide
legitimate cached keys from the chunk. Branch 3 is not exercised by greedy decoding today, so it has a dedicated test
(`test_chunked_prefill_into_nonempty_cache_matches_single_pass`). Chunked prefill in M3 depends on it.

### 9. Why load weights onto the `meta` device first?
Constructing `nn.Linear` normally allocates and randomly initializes every parameter, which wastes time and holds two
copies of the model in memory during load. `load_model` builds the module under `torch.device("meta")` (shapes only),
loads tensors from safetensors, validates names and shapes, then calls `load_state_dict(..., assign=True)` so the
checkpoint tensors become the parameters. The non-persistent RoPE buffer is then rebuilt on the real device (`reset_rope`).

### 10. How are errors propagated from checkpoint and config problems?
Two typed exceptions: `UnsupportedModelError` (architecture features we do not implement, raised by
`ModelConfig.from_hf_dict`) and `CheckpointError` (pickle-only weights, missing/unexpected tensors, shape mismatch,
inconsistent tied head, raised by `load_model`). Both fail at load time instead of serving wrong output. Each path has a
test in `tests/unit/model/test_config_and_loader.py`.

## Security

### 11. What is the attack surface of model loading, and how is it reduced?
Weights come from the internet. PyTorch `.bin`/`.pt` files are pickles, and unpickling can execute arbitrary code. The loader
refuses directories that only contain pickle formats (`_safetensor_files`), and `hub.download` never requests `*.bin`
(`_ALLOW_PATTERNS`). Revisions are pinned commit hashes (`PINNED_REVISIONS`), so a compromised or changed `main` branch
cannot silently change what is loaded. Remaining gap: no per-file digest check against a signed manifest yet. The
portfolio's model-manifest contract (C8) adds that.

### 12. How are credentials kept off the GPU host?
`scripts/gpu/common.sh` authenticates through the local ssh-agent. It passes only the public key to `ssh -i` with
`IdentitiesOnly=yes` and sets `ForwardAgent=no`, so a compromised host cannot use the agent. `sync.sh` sends only files
from `git ls-files` (no `.git`, no ignored files such as `scripts/gpu/host.env`) plus a `REVISION` marker. Models are
ungated, so no Hugging Face token is needed on the host (`docs/operations.md` §1).

### 13. Why did the security workflow need to strip `+cpu`/`+cu130` from versions?
`pip-audit` resolves packages against PyPI. The PyTorch builds come from download.pytorch.org with local version labels,
so `torch 2.14.1+cpu` cannot be audited and the run fails. Advisories are published against the upstream version, so
`security.yml` strips the label and audits both locked dependency sets with `--no-deps` (the lock already pins the
full closure). Running it locally found a real issue: transitive `setuptools 81.0.0` had PYSEC-2026-3447 and
GHSA-xmj3-9gf7-h52w. A `constraint-dependencies` floor in `pyproject.toml` raised it to 84.2.1 (commit "security: raise
transitive setuptools").

## Debugging

### 14. Generated text is fluent but wrong after a few dozen tokens. How do you investigate?
That pattern points at position handling rather than weights: RoPE theta/scaling/layout, or KV-cache positions.
(1) Compare full-sequence logits at long positions against the reference
(`test_positions_beyond_original_context_match`). (2) Compare cached vs recompute generation
(`test_kv_cache_path_equals_recompute_path`). A mismatch there isolates the cache or mask branch from the model math.
(3) Check `ModelConfig.rope` against the checkpoint's `config.json`, including which format (Hub `rope_scaling` vs
v5 `rope_parameters`) it came from.

### 15. Our tokens diverge from the reference at step k. Bug or numerics?
Look at the reference's top-2 logit gap at step k, which the parity report records (`ref_top2_gap_at_divergence`). In fp32
both implementations agree to about 1e-6, so a divergence with a gap ≫ 1e-5 is a bug. In bf16 the logit resolution near
magnitude 32 is 0.25, so flips at gaps of that order are rounding, not bugs. The bf16 acceptance rule in
`tests/parity/test_hf_greedy_parity.py` encodes this.

## Performance

### 16. What has been measured so far, and what has not?
Correctness has been measured. Performance has not. The parity reports include wall-clock fields, but they come from
sequential, unbatched, single-prompt loops with warm-up included, so they are not benchmarks, and I don't compare engines
with them. The A16 validation includes smoke timings (≈ 15 TFLOP/s bf16 matmul, 168 GB/s device copy
`[GPU:A16x1@vultr]`), which are labeled as smoke checks for detecting a broken or throttled device, not results. The
benchmark harness (open-loop load, baselines, `benchmarks/methodology.md`) is milestone M5. GPU campaigns over 15
minutes need approval (Gate B).

### 17. Both parity suites show 100/100. What does that prove, and what doesn't it prove?
It proves that for two real checkpoints (Llama architecture via SmolLM2, Qwen2 architecture), the native model, loader,
RoPE, GQA, tied head, and KV-cached decoding reproduce `transformers` exactly: 25,600 greedy steps per precision, in
fp32 on CPU `[CPU-LAPTOP]` and bf16 on GPU `[GPU:A16x1@vultr]`. The fp32 run is also informative about margins: the
smallest reference top-2 logit gap across all steps was 1.2e-4, about 100× above fp32 implementation noise, so a single
flipped token would have pointed at a real bug.
It does **not** prove: Llama-3 RoPE scaling on real weights (gated checkpoints; verified only on random tiny models),
correctness of sampling (not implemented), behavior beyond ~180 total tokens, or anything about batching or performance.

### 18. In bf16 the reference had exact logit ties (top-2 gap 0.0), yet every token matched. Why?
In bf16, distinct logits often collapse onto the same representable value. Logits near magnitude 32 are spaced 0.25
apart, so exact ties are common over 25,600 steps. A tie is resolved by the argmax rule, and both implementations take
the lowest token id (`torch.argmax` semantics, documented in `greedy_generate`). Matching across ties therefore also
shows that our bf16 logits round to the same values as the reference's at those steps. Had the tie-break rules
differed, the suite would have reported a divergence at a gap of 0.0, which the acceptance rule treats as rounding.

## Debugging (continued)

### 19. Describe a real defect found during M1 and how it was handled.
A **provenance bug** in the evidence tooling: the first full CPU parity run started from commit `d676e49`, but its
`env.json` reported a later revision (`…-dirty`), because `dli.envinfo` read `git describe` when results were *written*
(70 minutes later, after unrelated commits). Handling: (1) verified the measurement was still valid with
`git diff d676e49 46f6359^ -- src tests pyproject.toml uv.lock` (empty); (2) fixed the cause in `46f6359`: the revision is
captured at run start (`current_revision()`) and passed to `collect(revision=…)`, and the parity suite also stores it as
`code_revision`; (3) added a regression test that fails if a moved HEAD is reported; (4) left the raw `env.json` unedited and
recorded the correction in `PROVENANCE.md` beside it. Lesson: evidence metadata has to be pinned at the start of a run,
just like inputs.

### 20. A GPU result says `revision: …-dirty`. Can you trust it?
Not on its own. "Dirty" means uncommitted files were present, so the exact code is unknown. That happened on the first A16
validation, and it was rerun after `scripts/gpu/sync.sh` gained `SYNC_FROM=head`, which ships `git archive HEAD` and writes
the exact commit to `REVISION`. Evidence runs use HEAD sync. Worktree sync is for iteration only (`docs/operations.md` §3).

## Results (labeled evidence)

| Claim | Label | Evidence |
|---|---|---|
| 100/100 prompts × 128 tokens identical to `transformers`, SmolLM2-135M and Qwen2.5-0.5B, fp32 | `[CPU-LAPTOP]` | `benchmarks/results/2026-11-10-CPU-LAPTOP-m1-greedy-parity-float32/` (+ `PROVENANCE.md`) |
| Same, bf16, both implementations on one GPU | `[GPU:A16x1@vultr]` | `benchmarks/results/2026-11-10-GPU-A16x1-vultr-m1-greedy-parity-bfloat16/` |
| A16 host usable: fp32/SDPA numerics, Triton JIT, container GPU access | `[GPU:A16x1@vultr]` | `benchmarks/results/2026-11-10-GPU-A16x1-vultr-env-validation/` |
| Tiny-model logit parity ≤ 1e-4 on 7 architecture variants | `[CPU-LAPTOP]` | `tests/unit/model/test_reference_parity.py` (44-test fast suite) |
