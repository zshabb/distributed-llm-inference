# Prefix-cache eviction: LRU vs cost-aware (GDSF), and a partly negative result

Implementation: `src/dli/kv/eviction.py` (policies), `src/dli/kv/block_manager.py` (candidate management).
Study: `benchmarks/scripts/prefix_cache_sim.py` → `benchmarks/results/2026-11-10-SIM-m2-prefix-eviction/`.

## 1. Problem
When the KV pool is full, a new request needs blocks, and cached-idle blocks (valid KV that no request is using)
must be reclaimed. Which ones to evict decides how much future prefill work the cache saves.

## 2. Why it matters
Every prefix-cache hit skips prefill compute for those tokens, which reduces time to first token and GPU load. Under
memory pressure, the eviction policy decides the hit rate.

## 3. Existing solutions
LRU over cached blocks (vLLM's automatic prefix caching); LRU over radix-tree leaves (SGLang). Frequency- and
cost-aware web-cache policies such as GreedyDual-Size-Frequency (Cherkasova 1998) exist but are rarely applied to KV.

## 4. Limitations of LRU (hypothesis)
LRU forgets popularity: a burst of one-off requests can flush a system prompt shared by most traffic.

## 5. Design
Only **leaf** cached-idle blocks are candidates, so the index stays prefix-closed (`block_manager.py`). Policies are
pluggable behind `EvictionPolicy`:
- `LRUPolicy`: evict the least recently released candidate. O(1).
- `CostAwarePolicy`: GDSF priority `H = L + freq × (1 + 0.05·depth)`. `L` (inflation) is set to the priority of each
  victim, which ages stale entries. O(log n) with a lazily invalidated heap, compacted when stale entries dominate.

## 6. Alternatives considered
LFU (no aging, so stale entries pin memory permanently; rejected), ARC/2Q (adaptive recency/frequency balance; candidate
follow-up, see §11), learned policies (no traces of real traffic yet).

## 7. Implementation details
The policy receives `on_cached` (depth), `on_hit` (frequency), `add_candidate`/`remove_candidate` (when a block becomes
or stops being an unused leaf), `pop_victim`, `on_evicted`. The block manager guarantees candidates are exactly the idle
leaves (`check_invariants`).

## 8. Complexity and tradeoffs
LRU O(1); cost-aware O(log n) amortized and somewhat slower in the soak (10⁶ ops: 10.0 s vs 8.0 s, incidental
`[CPU-LAPTOP]`). Cost-aware also has a tunable (`depth_weight`) whose right value depends on workload.

## 9. Correctness testing
Policy unit tests (ordering, frequency retention, aging, depth preference, bounded heap), both policies in the stateful
property test and the 10⁶-operation soak, and paged-decoding correctness under eviction pressure with both policies.

## 10. Measurements `[SIM]`

**Method.** Real `BlockManager` replay of synthetic traces (`benchmarks/methodology.md`): 3,000 requests per trace, block
size 16, 5 seeds, 4 pool sizes. Both policies replay the **identical** trace per (workload, seed), and differences are
paired. Metric: fraction of prompt tokens served from cache. **No forward passes and no latency**: these are hit rates,
not TTFT. Code `26b2bc065c67`.

| Workload | Pool (blocks / tokens) | LRU hit | Cost-aware hit | Paired Δ mean [min, max] | Seeds cost > LRU |
|---|---|---|---|---|---|
| shared-system | 128 / 2048 | 0.346 ± 0.026 | 0.384 ± 0.039 | +0.0377 [+0.0151, +0.0510] | 5/5 |
| shared-system | 256 / 4096 | 0.550 ± 0.035 | 0.591 ± 0.036 | +0.0408 [+0.0339, +0.0501] | 5/5 |
| shared-system | 512 / 8192 | 0.750 ± 0.019 | 0.775 ± 0.016 | +0.0250 [+0.0170, +0.0284] | 5/5 |
| shared-system | 1024 / 16384 | 0.852 ± 0.011 | 0.857 ± 0.011 | +0.0047 [+0.0031, +0.0060] | 5/5 |
| multi-turn | 128 / 2048 | 0.202 ± 0.024 | 0.200 ± 0.024 | −0.0014 [−0.0026, −0.0006] | 0/5 |
| multi-turn | 256 / 4096 | 0.407 ± 0.047 | 0.391 ± 0.042 | −0.0161 [−0.0240, −0.0099] | 0/5 |
| multi-turn | 512 / 8192 | 0.622 ± 0.042 | 0.565 ± 0.040 | −0.0576 [−0.0662, −0.0530] | 0/5 |
| multi-turn | 1024 / 16384 | 0.826 ± 0.022 | 0.717 ± 0.029 | −0.1090 [−0.1199, −0.0990] | 0/5 |
| scan-polluted | 128 / 2048 | 0.117 ± 0.008 | 0.117 ± 0.008 | −0.0003 [−0.0006, −0.0000] | 0/5 |
| scan-polluted | 256 / 4096 | 0.214 ± 0.009 | 0.211 ± 0.009 | −0.0028 [−0.0042, −0.0013] | 0/5 |
| scan-polluted | 512 / 8192 | 0.338 ± 0.005 | 0.334 ± 0.007 | −0.0044 [−0.0073, +0.0004] | 1/5 |
| scan-polluted | 1024 / 16384 | 0.476 ± 0.005 | 0.461 ± 0.007 | −0.0152 [−0.0198, −0.0100] | 0/5 |

**Conclusions (for these synthetic traces only):**
1. **Supported:** with a few popular shared system prompts and short unique suffixes, cost-aware beats LRU in every
   seed, by +2.5 to +4.1 hit-rate points under memory pressure, with the gain shrinking as the pool grows.
2. **Refuted:** the spec's hypothesis that cost-aware eviction raises hit rates on **multi-turn** workloads. It is
   worse in every seed, increasingly so with larger pools (−10.9 points at 16K tokens).
3. Scan pollution: no benefit; slightly worse.
4. **Decision:** LRU remains the default (`BlockManager(eviction_policy="lru")`). Cost-aware is an opt-in for
   system-prompt-dominated deployments.

**Likely mechanism (hypothesis, not yet tested in isolation):** in a multi-turn session, the early blocks are hit on every
turn, so they accumulate high frequency. After the session ends, those dead blocks outrank the newest tail blocks of
live sessions, which are exactly what the next turn needs. Aging through `L` recovers too slowly at large pool sizes, where
evictions (and therefore increments of `L`) are rare.

## 11. Remaining limitations and follow-ups
- Synthetic traces only; real traffic distributions may differ in either direction. No latency evidence until M3/M5.
- Candidate follow-ups, to be judged by the same paired methodology: frequency decay on session end, or an adaptive
  recency/frequency split (ARC-style). Not implemented now: tuning a policy on the same synthetic traces used to judge
  it would overfit.
- `depth_weight` was not swept. The default (0.05) is a stated assumption, not a fitted value.
