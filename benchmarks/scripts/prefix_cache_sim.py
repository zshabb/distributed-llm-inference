"""Trace simulation: LRU vs cost-aware (GDSF) prefix-cache eviction in the real BlockManager.

Evidence label: [SIM]. The actual BlockManager runs (allocation, hashing, prefix lookup, eviction),
but no forward passes happen. The outputs are prefix-hit rate and prefill tokens avoided, NOT
latency. TTFT impact needs the engine (M3) and the benchmark harness (M5).

Workloads (synthetic, seeded; see benchmarks/methodology.md):
  * shared-system: Zipf-popular system prompts + short unique user turns
  * multi-turn: sessions whose prompt grows by (response + new user message) each turn
  * scan-polluted: multi-turn traffic mixed with one-off long documents that churn the cache

Both policies replay the IDENTICAL request trace for each (workload, seed), so differences come
from the policy only. Requests run one at a time, so the cache competes only with itself.

Usage: uv run --frozen python benchmarks/scripts/prefix_cache_sim.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dli import envinfo
from dli.kv import BlockManager

BLOCK_SIZE = 16
POOL_SIZES = [128, 256, 512, 1024]
SEEDS = [1, 2, 3, 4, 5]
N_REQUESTS = 3000


@dataclass
class Request:
    prompt: list[int]
    decode: list[int]


def _toks(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(32000) for _ in range(n)]


def _zipf_choice(rng: random.Random, n: int, s: float = 1.1) -> int:
    weights = [1 / (i + 1) ** s for i in range(n)]
    return rng.choices(range(n), weights=weights)[0]


def make_trace(workload: str, seed: int) -> list[Request]:
    rng = random.Random(seed)
    systems = [_toks(rng, rng.randrange(200, 800)) for _ in range(20)]
    trace: list[Request] = []
    if workload == "shared-system":
        for _ in range(N_REQUESTS):
            sys_p = systems[_zipf_choice(rng, len(systems))]
            trace.append(Request(sys_p + _toks(rng, rng.randrange(10, 120)), _toks(rng, 64)))
        return trace

    sessions: dict[int, list[int]] = {}
    next_session = 0
    while len(trace) < N_REQUESTS:
        if workload == "scan-polluted" and rng.random() < 0.25:
            doc = systems[_zipf_choice(rng, len(systems))] + _toks(rng, rng.randrange(800, 2400))
            trace.append(Request(doc, _toks(rng, 32)))
            continue
        if not sessions or (len(sessions) < 12 and rng.random() < 0.15):
            sessions[next_session] = list(systems[_zipf_choice(rng, len(systems))])
            next_session += 1
        sid = rng.choice(list(sessions))
        ctx = sessions[sid] + _toks(rng, rng.randrange(20, 80))  # new user message
        response = _toks(rng, rng.randrange(30, 200))
        trace.append(Request(ctx, response))
        sessions[sid] = ctx + response
        if len(sessions[sid]) > 3000 or rng.random() < 0.1:  # session ends
            del sessions[sid]
    return trace


def replay(trace: list[Request], policy: str, num_blocks: int) -> dict[str, float]:
    bm = BlockManager(num_blocks, BLOCK_SIZE, eviction_policy=policy, watermark=0.0,
                      debug_checks=False)  # fmt: skip
    computed = skipped = 0
    t0 = time.perf_counter()
    for i, req in enumerate(trace):
        need = -(-(len(req.prompt) + len(req.decode)) // BLOCK_SIZE)
        if need > num_blocks:  # cannot ever fit: counted as computed in full, not cached
            computed += len(req.prompt)
            continue
        alloc = bm.allocate(i, req.prompt, respect_watermark=False)
        bm.mark_computed(i, len(req.prompt))
        computed += len(req.prompt) - alloc.num_cached_tokens
        skipped += alloc.num_cached_tokens
        for tok in req.decode:
            bm.append_slots(i, [tok])
            bm.mark_computed(i, bm.num_tokens(i))
        bm.free(i)
    s = bm.stats()
    total = computed + skipped
    return {
        "hit_rate": skipped / total,
        "prefill_tokens_computed": computed,
        "evictions": s.evictions,
        "seconds": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    revision = envinfo.current_revision()
    out = (
        args.out
        or Path("benchmarks/results") / f"{datetime.now(UTC):%Y-%m-%d}-SIM-m2-prefix-eviction"
    )
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for workload in ("shared-system", "multi-turn", "scan-polluted"):
        for seed in SEEDS:
            trace = make_trace(workload, seed)
            for blocks in POOL_SIZES:
                for policy in ("lru", "cost"):
                    r = replay(trace, policy, blocks)
                    rows.append({"workload": workload, "seed": seed, "pool_blocks": blocks,
                                 "policy": policy, **r})  # fmt: skip
                    print(f"{workload:14s} seed={seed} blocks={blocks:5d} {policy:4s} "
                          f"hit={r['hit_rate']:.4f}", file=sys.stderr)  # fmt: skip

    summary = []
    for workload in ("shared-system", "multi-turn", "scan-polluted"):
        for blocks in POOL_SIZES:
            per_policy = {}
            for policy in ("lru", "cost"):
                hits = [r["hit_rate"] for r in rows if (r["workload"], r["pool_blocks"],
                        r["policy"]) == (workload, blocks, policy)]  # fmt: skip
                per_policy[policy] = (statistics.mean(hits), statistics.stdev(hits))
            deltas = [
                c["hit_rate"] - lr["hit_rate"]
                for c in rows
                for lr in rows
                if c["policy"] == "cost"
                and lr["policy"] == "lru"
                and (c["workload"], c["seed"], c["pool_blocks"])
                == (lr["workload"], lr["seed"], lr["pool_blocks"])
                and c["workload"] == workload
                and c["pool_blocks"] == blocks
            ]
            summary.append({
                "workload": workload, "pool_blocks": blocks,
                "pool_tokens": blocks * BLOCK_SIZE,
                "lru_hit_mean": round(per_policy["lru"][0], 4),
                "lru_hit_std": round(per_policy["lru"][1], 4),
                "cost_hit_mean": round(per_policy["cost"][0], 4),
                "cost_hit_std": round(per_policy["cost"][1], 4),
                "paired_delta_mean": round(statistics.mean(deltas), 4),
                "paired_delta_min": round(min(deltas), 4),
                "paired_delta_max": round(max(deltas), 4),
                "seeds_cost_better": sum(d > 0 for d in deltas),
                "seeds": len(deltas),
            })  # fmt: skip

    meta = {
        "evidence_label": "SIM",
        "code_revision": revision,
        "block_size": BLOCK_SIZE,
        "requests_per_trace": N_REQUESTS,
        "seeds": SEEDS,
        "cost_policy": "CostAwarePolicy(depth_weight=0.05)",
        "note": "Prefix-hit rate from real BlockManager replay; no forward passes; not latency.",
    }
    envinfo.main([str(out / "env.json")], revision=revision)
    (out / "results.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "runs": rows}, indent=1) + "\n"
    )
    lines = [
        "| workload | pool (blocks / tokens) | LRU hit | cost-aware hit "
        "| paired Δ mean [min, max] | seeds cost > LRU |",
        "|---|---|---|---|---|---|",
    ]
    for s in summary:
        delta = (
            f"{s['paired_delta_mean']:+.4f} "
            f"[{s['paired_delta_min']:+.4f}, {s['paired_delta_max']:+.4f}]"
        )
        lines.append(
            f"| {s['workload']} | {s['pool_blocks']} / {s['pool_tokens']} | "
            f"{s['lru_hit_mean']:.3f} ± {s['lru_hit_std']:.3f} | "
            f"{s['cost_hit_mean']:.3f} ± {s['cost_hit_std']:.3f} | "
            f"{delta} | {s['seeds_cost_better']}/{s['seeds']} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
