# Engineering Guide

How the system works, for an engineer who needs to maintain or extend it.
**Scope of this revision: milestones M0–M2** (model, loader, single-sequence generation, paged KV memory with
prefix caching, tooling).
Sections for the scheduler, paged KV cache, API, kernels, tensor parallelism, and router are added as those land.
Nothing described as *planned* exists yet (see `docs/implementation-plan.md`).

## 1. End-to-end flow today (M1)

```
token ids ──► greedy_generate (dli/generate.py)
                 │ allocates ContiguousKVCache(len(prompt)+max_new_tokens)
                 ▼
            LlamaForCausalLM.forward(input_ids[n], positions[n], cache, last_token_only)
                 │ embed → for each layer: RMSNorm → Attention(RoPE, GQA, cache write) → +res
                 │                         RMSNorm → SwiGLU MLP → +res
                 ▼ final RMSNorm → logits (LM head or tied embedding)
            argmax → append → next step feeds 1 token at position len-1
```

- **Prefill** is the first call: all prompt tokens at once, causal attention, writes K/V for every position.
- **Decode** calls feed one token. Attention reads the whole cached prefix, and no mask is needed because a single
  query may see everything before it.
- A **chunk appended to a non-empty cache** (n > 1, past > 0) needs an explicit mask, because SDPA's `is_causal` aligns
  to the top-left corner. This path is not used by greedy decoding today, but is tested
  (`test_chunked_prefill_into_nonempty_cache_matches_single_pass`) because chunked prefill (M3) will depend on it.

## 1a. Paged KV flow (M2)

```
paged_greedy_generate(model, BlockManager, PagedKVStore, seq_id, prompt)
  BlockManager.allocate(seq, prompt)        → block table; leading full blocks reused from the prefix cache
  PagedSequence.prefill(prompt)             → forward only prompt[num_cached_tokens:]
     model(..., PagedSequenceView)           write: K/V rows → store slots (slot_mapping)
                                             read:  gather the sequence's pages in position order
     view.advance(n) → mark_computed         completed full blocks become shareable
  loop: BlockManager.append_slots(seq,[t])  → CopyOps (copy-on-write if the last block is shared)
        PagedKVStore.copy_blocks(ops); forward 1 token through the view
  BlockManager.free(seq)                    → cached blocks stay CACHED-IDLE (revivable), others FREE
```

The block manager owns *metadata* (where KV lives, who shares it, what is cached). The store owns *bytes*. The model sees
only the `KVCacheView` protocol (`length`, `write`, `advance`), so contiguous and paged caches are interchangeable and
the attention math did not change. Design and invariants: `docs/algorithms/paged-kv-cache.md`. Eviction policies:
`docs/algorithms/prefix-cache-eviction.md`. Decisions: ADR-0003.

## 2. Components

| Component | File | Responsibility | Key invariants |
|---|---|---|---|
| Config | `src/dli/model/config.py` | Parse HF `config.json` (Hub and transformers-v5 formats) into a frozen dataclass | Unsupported features raise `UnsupportedModelError` and are never ignored |
| RoPE | `src/dli/model/rope.py` | Inverse frequencies (default/linear/llama3), cos/sin tables, rotation | Angles computed in fp32; rotate-half layout |
| Model | `src/dli/model/llama.py` | Decoder graph; parameter names mirror checkpoints | Layers never see requests; positions are explicit |
| KV cache (M1) | `src/dli/model/kv_cache.py` | Contiguous per-sequence K/V | Writes must be contiguous and within `max_len`; `length` advances only after all layers wrote |
| Loader | `src/dli/model/loader.py` | safetensors → meta-built model via `assign=True` | No pickle; exact name/shape match; tied-head consistency |
| Hub | `src/dli/model/hub.py` | Download at pinned commit hashes | Only ungated Apache-2.0 models; `*.bin` never downloaded |
| Generation | `src/dli/generate.py` | Greedy loop, cached or recompute | Lowest-index argmax tie-break, same as the reference |
| Env manifest | `src/dli/envinfo.py` | Hardware/software manifest per result | Never raises on missing tools; revision captured at run start; untracked files count as dirty |
| Block hashing | `src/dli/kv/hashing.py` | SHA-256 chain identity of full blocks | Equal hash ⇒ equal prefix (same namespace) |
| Block manager | `src/dli/kv/block_manager.py` | Allocation, ref counts, COW, prefix index, eviction driving | `check_invariants()`; failures raise before mutating; thread-confined |
| Eviction policies | `src/dli/kv/eviction.py` | LRU and cost-aware GDSF over leaf candidates | Only unused leaves are candidates |
| Paged store | `src/dli/kv/paged_store.py` | KV pool tensors, slot scatter, page gather, COW copies | Layout `[layers, blocks, B, kv_heads, head_dim]` |
| Paged generation | `src/dli/generate.py` (`PagedSequence`, `paged_greedy_generate`) | One sequence over paged KV | Equal to the contiguous path |

## 3. Data structures and memory

- **Parameters** are loaded straight into their final dtype/device. Building on `meta` avoids allocating and randomly
  initializing a throwaway copy, so peak host memory is about one copy of the checkpoint (plus the safetensors mmap).
- **KV cache (M1):** `2 × layers × kv_heads × max_len × head_dim × bytes`. For Qwen2.5-0.5B in fp32
  (24 layers, 2 KV heads, head_dim 64) that is 24,576 bytes per token; at 4096 tokens, 96 MiB per sequence,
  reserved up front whether used or not. That waste, multiplied by concurrent sequences, is exactly what the
  paged block manager (M2) removes.
- **GQA expansion:** K/V are expanded with `repeat_interleave` before SDPA. That costs memory bandwidth proportional
  to the fan-out (3× for SmolLM2's 9 query / 3 KV heads; 7× for Qwen2.5-0.5B's 14 / 2). The paged kernel (M6)
  will index KV heads directly instead of materializing the expansion.

## 4. Numerical design decisions

| Decision | Why |
|---|---|
| RMSNorm in fp32, weight multiply after cast back | bf16 sum-of-squares over hundreds of channels loses precision; identical cast placement is required for bit-level parity |
| RoPE angles in fp32 | Angle error in bf16 grows with position; rotation tables are cast only at the end |
| Keys cached post-rotation | Absolute positions never change, so a cached key never needs re-rotation (required for prefix sharing in M2) |
| Same SDPA entry point as the reference | Keeps fp32 results within float reordering noise of `transformers` |
| Explicit `positions` argument | Continuous batching packs tokens at different positions into one vector |

## 5. Verification strategy

1. **Fast tiny-model parity** (`tests/unit/model/test_reference_parity.py`, ~6 s): random 3-layer models from
   `transformers` saved as real safetensors. Covers 7 variants (untied/tied, Llama biases, llama3 and linear RoPE, MHA,
   Qwen2), positions past the original context, chunked prefill, cache vs recompute, EOS handling.
2. **Real-checkpoint parity** (`tests/parity/`, `-m parity`): 100 prompts × 128 greedy tokens on two pinned models.
   Results land in `benchmarks/results/<date>-<label>-m1-greedy-parity-*` with `env.json`. Current results: 100/100
   identical for both models in fp32 `[CPU-LAPTOP]` and bf16 `[GPU:A16x1@vultr]` (see README *Verified results*).
   Long runs capture the code revision at start (`envinfo.current_revision`) because HEAD can move during a 70-minute run.
3. **Rejection paths:** every loader and config error has a test that triggers it.

Why exact token equality is meaningful in fp32: both implementations perform the same operations in the same
order, so per-step logit differences are ~1e-6, far below typical top-2 gaps. A token flip in fp32 would point to a
real bug or a pathological tie, and the report records the reference's top-2 gap at any divergence to tell the two apart.

## 6. Technology choices (M0–M1)

See `docs/design-decisions/`: ADR-0001 (Python/PyTorch engine + Go router, layering) and ADR-0002 (torch 2.14.1 pinned
for CPU and CUDA 13.0 builds in one lockfile, matching the A16 host driver).

## 7. Debugging procedures

| Symptom | Likely cause | How to check |
|---|---|---|
| Fluent but wrong text | RoPE layout or theta/scaling mismatch | `test_positions_beyond_original_context_match`; compare `inverse_frequencies` against the reference |
| Garbage from the first token | Wrong GQA grouping or tied/untied head | Logit parity on a tiny variant with the same head counts |
| Cached output ≠ recompute | Cache `length` advanced early, or wrong mask branch | `test_kv_cache_path_equals_recompute_path`; check `past` in `Attention.forward` |
| `CheckpointError: missing tensors` | Architecture flag not mapped (e.g. biases) | Compare `model.state_dict().keys()` with `safetensors` keys |
| GPU numbers look throttled | Unlicensed vGPU or another tenant's load | `env.json` → `gpu.details["License Status"]`; `nvidia-smi -q` |
| Paged output ≠ contiguous | Shared block written (missing COW), hit on uncomputed block, slot mapping off-by-one | `check_invariants()`; run the stateful test with `HYPOTHESIS_PROFILE=focused`; `scripts/dev/mutation_check_block_manager.sh` confirms the tests still catch these faults |
| `NoFreeBlocksError` with apparently free memory | Watermark reserve at admission, or idle blocks pinned by in-use children | `stats()`; `check_invariants()` asserts idle blocks are evictable (liveness) |
| Blocks leak over time | A sequence never freed (scheduler bug) | `stats().used_blocks` vs live sequences; stress soak pattern in `tests/stress/` |

## 8. Scaling, failure handling, limitations

Not applicable yet. M1 is a single-sequence, in-process path, so scaling (batching, paging, routing) and failure handling
(admission, cancellation, worker loss) begin at M2–M4 and M8. Current limitations: `docs/known-issues.md`.
