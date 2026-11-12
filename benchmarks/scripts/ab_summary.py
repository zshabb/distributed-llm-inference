"""Summarize an A/B of campaign runs laid out as ``<root>/<variant>/rep<i>/`` (a campaign per rep).

For every (experiment, concurrency): median and [min, max] output tok/s per variant, median TTFT /
TPOT / E2E percentiles, graph replays per engine step and eager decode fallbacks (from the server's
Prometheus counters), the speedup of the second variant over the first, and whether the
repetition ranges are disjoint. Writes ``<root>/ab_summary.json``.

Usage: python benchmarks/scripts/ab_summary.py <root> main branch
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _load(root: Path, variant: str) -> dict[tuple[str, int], list[dict[str, float]]]:
    out: dict[tuple[str, int], list[dict[str, float]]] = {}
    for f in sorted((root / variant).glob("rep*/runs/*/*/rep0/*.json")):
        d = json.loads(f.read_text())
        r = d["resources"]
        out.setdefault((d["experiment"], int(d["point"])), []).append(
            {
                "tok": d["output_tok_per_s"],
                "ttft50": d["ttft_s"]["p50"] * 1e3,
                "ttft99": d["ttft_s"]["p99"] * 1e3,
                "tpot50": d["tpot_s"]["p50"] * 1e3,
                "e2e99": d["e2e_s"]["p99"] * 1e3,
                "steps": r.get("dli_engine_steps_total_delta") or 0.0,
                "replays": r.get("dli_cuda_graph_replays_total_delta") or 0.0,
                "fallbacks": r.get("dli_cuda_graph_fallbacks_total_delta") or 0.0,
                "gpu_mem": r.get("gpu_mem_used_max", 0.0) / 2**30,
            }
        )
    return out


def _stats(xs: list[dict[str, float]]) -> dict[str, Any]:
    toks = sorted(x["tok"] for x in xs)
    med = statistics.median
    return {
        "reps": len(xs),
        "tok_per_s_median": round(med(toks), 1),
        "tok_per_s_range": [round(toks[0], 1), round(toks[-1], 1)],
        "ttft_p50_ms": round(med(x["ttft50"] for x in xs)),
        "ttft_p99_ms": round(med(x["ttft99"] for x in xs)),
        "tpot_p50_ms": round(med(x["tpot50"] for x in xs), 1),
        "e2e_p99_ms": round(med(x["e2e99"] for x in xs)),
        "graph_replays_per_step": round(
            sum(x["replays"] for x in xs) / max(1.0, sum(x["steps"] for x in xs)), 3
        ),
        "graph_fallbacks": sum(x["fallbacks"] for x in xs),
        "gpu_mem_max_gib": round(max(x["gpu_mem"] for x in xs), 2),
    }


def main() -> None:
    root, a, b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    ra, rb = _load(root, a), _load(root, b)
    rows = []
    for key in sorted(ra.keys() & rb.keys()):
        sa, sb = _stats(ra[key]), _stats(rb[key])
        row = {
            "experiment": key[0],
            "concurrency": key[1],
            a: sa,
            b: sb,
            "speedup": round(sb["tok_per_s_median"] / sa["tok_per_s_median"], 2),
            "ranges_disjoint": sb["tok_per_s_range"][0] > sa["tok_per_s_range"][1]
            or sb["tok_per_s_range"][1] < sa["tok_per_s_range"][0],
        }
        rows.append(row)
        print(
            f"{key[0]:13s} c={key[1]:3d}  {a} {sa['tok_per_s_median']:8.0f}  "
            f"{b} {sb['tok_per_s_median']:8.0f}  x{row['speedup']}  "
            f"disjoint={row['ranges_disjoint']}"
        )
    (root / "ab_summary.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
