# Paged KV cache: block manager, prefix sharing, copy-on-write

Implementation: `src/dli/kv/block_manager.py`, `src/dli/kv/hashing.py`, `src/dli/kv/paged_store.py`.
Tests: `tests/unit/kv/`, `tests/unit/model/test_paged_generation.py`, `tests/stress/`, `tests/parity/test_paged_parity.py`.

## 1. Problem

Each running sequence needs its attention keys and values for every past token, in every layer. A contiguous
per-sequence buffer must be sized for the maximum length up front. Most sequences never reach it, so most of the
reserved memory is wasted, and that waste caps how many sequences fit in GPU memory, which caps batch size and
throughput. Identical prompt prefixes (system prompts, few-shot templates, earlier turns of a chat) are also recomputed
and stored once per request.

## 2. Why it matters

KV memory, not compute, is usually what limits concurrency in LLM serving. Example, Qwen2.5-0.5B in bf16: 12 KiB of KV
per token. Reserving a 4,096-token context costs 48 MiB per sequence whether it uses 100 tokens or 4,000. Prefix reuse
also skips prefill compute, which directly reduces time to first token for repeated prefixes.

## 3. Existing solutions

- **Contiguous caches** (HF `transformers` static/dynamic caches): simple, but they reserve memory or reallocate as they grow.
- **PagedAttention** (vLLM, Kwon et al., SOSP 2023): fixed-size blocks plus block tables; the inspiration for this design.
- **Automatic prefix caching** (vLLM hash-based; SGLang RadixAttention with a radix tree of token sequences).

## 4. Their limitations (relative to this project's goals)

They are production systems whose block managers are tightly coupled to their schedulers and kernels, which makes the
policies hard to study or swap. Eviction is typically LRU-only. The contribution here is a self-contained,
property-tested manager with pluggable eviction, used to *measure* policy effects (see `prefix-cache-eviction.md`).

## 5. Design

- **Blocks:** the KV pool is `num_blocks × block_size` token slots per layer. A sequence's *block table* maps logical
  position `p` to slot `table[p // B] * B + p % B`.
- **States:** FREE (free list), USED (`ref > 0`), CACHED-IDLE (`ref == 0` but registered in the prefix index; valid KV).
- **Identity:** a full block's hash is `SHA-256(len(ns) ‖ ns ‖ parent_hash ‖ int64 tokens)`, a chain, because a block's KV
  depends on the whole prefix (`hashing.py`).
- **Prefix index:** `hash → block` plus per-block `(parent, depth, cached-children count)`. A lookup walks the prompt's
  full-block hashes from the root and stops at the first miss.
- **Rules** (module docstring of `block_manager.py`): cache only full blocks; register only after KV is computed; always
  recompute the block holding the last prompt token; keep the index prefix-closed with leaf-first eviction; never write
  shared blocks (copy-on-write); fail before mutating; register a block only under the parent block its own sequence holds.
- **Storage:** `PagedKVStore` holds `[layers, blocks, B, kv_heads, head_dim]` for K and V. `PagedSequenceView` scatters new
  rows to slots and, in the M2 reference path, gathers the sequence's pages for attention.

## 6. Alternatives considered

| Alternative | Why not (for now) |
|---|---|
| Radix tree keyed by token sequences (SGLang) | Supports partial-block sharing, but has more complex bookkeeping. Hash chains give O(1) lookups per block and simple invariants; partial-block sharing is a possible extension |
| Python `hash()` for block identity | 64-bit and not designed against adversarial collisions; a collision is a cross-request KV leak |
| Locking inside the manager | The scheduler is the single owner (ADR-0001); locks on every append are pure overhead. Thread confinement is enforced instead |
| Re-pointing duplicate blocks to the cached copy | Saves memory when identical prompts arrive concurrently, but complicates in-flight metadata. Counted instead (`duplicate_registrations`) |
| Swapping preempted KV to host memory | A scheduler concern (M3); the manager exposes the primitives (free/allocate/CopyOp) it needs |

## 7. Implementation details

- `allocate`: looks up hits, pre-checks capacity (revived idle hits no longer count as evictable), revives hits *before*
  taking new blocks so eviction cannot choose a block being reused, then takes the rest (free list first, then eviction).
- `append_slots`: allocates at block boundaries. If the partial last block is shared (`ref > 1`, after a fork) it returns
  `CopyOp(src, dst)`, which the caller applies with `PagedKVStore.copy_blocks` before the next write.
- `mark_computed`: hashes newly completed blocks from the sequence's unhashed tail (only the chain tip and tail are
  kept) and registers them.
- `free`: releases deepest-first, so children become eviction candidates before parents.

## 8. Complexity and tradeoffs

| Operation | Time | Notes |
|---|---|---|
| `allocate` | O(prompt_blocks) hashing + O(new_blocks × eviction cost) | SHA-256 ≈ µs per block |
| `append_slots` | O(1) amortized | O(1) extra on COW |
| `mark_computed` | O(newly full blocks) | |
| `free` | O(table length) | |
| Eviction | LRU O(1); cost-aware O(log n) amortized | Lazy heap with compaction |
| Memory overhead | O(blocks + table entries) | Per-sequence tail ≤ B tokens + chain tip |

Block size trades internal fragmentation (≤ B−1 slots per sequence), prefix-sharing granularity, and per-block overhead
against kernel efficiency (larger pages read more efficiently on GPU). Default 16 (ADR-0003).

## 9. Correctness testing

- 27 behavioral unit tests (`tests/unit/kv/test_block_manager.py`), each ending with `check_invariants()`.
- Model-based Hypothesis state machine with a shadow KV store, for both policies (`test_block_manager_stateful.py`).
  Focused profile: 2,000 examples × 80 steps per machine, passed.
- **Fault injection** (`scripts/dev/mutation_check_block_manager.sh`): four injected bugs (no COW, computed at allocation,
  non-leaf eviction, last block served from cache) are all detected. The run initially exposed a test gap (rule 3), which
  was fixed.
- Stress (`-m stress`): 10⁶ random operations per policy with per-operation conservation checks and full invariant
  checks every 2,000; a 10,000-request lifecycle soak. Zero violations, zero leaked blocks `[CPU-LAPTOP]`.
- End to end: paged decoding equals contiguous decoding on tiny models (block sizes 1/3/16, shared live prefixes, forks,
  eviction pressure) and on SmolLM2-135M (40/40 runs; `benchmarks/results/2026-11-10-CPU-LAPTOP-m2-paged-parity/`).

## 10. Performance measurements

- Manager overhead (incidental, from the stress soak): 10⁶ mixed operations in 8.0 s (LRU) / 10.0 s (cost-aware), i.e.
  ~8–10 µs per operation including the test's own bookkeeping `[CPU-LAPTOP]`. Not a benchmark.
- Prefix-cache hit rates by policy: `prefix-cache-eviction.md` `[SIM]`.
- **Not yet measured:** end-to-end throughput, TTFT, or memory capacity gains from paging (requires batching, M3, and the
  benchmark harness, M5). The M2 gather path is slower than the contiguous cache by construction.

## 11. Remaining limitations

- The gather-based read copies the whole prefix per layer per step (removed by the M6 kernel).
- Prefix sharing is at full-block granularity. A shared prefix shorter than B, or the tail of a longer one, is not shared.
- Duplicate concurrent prompts keep private copies (counted, not merged).
- Not thread-safe by design. Correct only under the single-owner scheduler model.
- Hash namespace policy (per model / per tenant) is the caller's responsibility. Cross-tenant sharing leaks hit timing.
