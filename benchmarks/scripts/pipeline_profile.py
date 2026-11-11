"""Where does closed-loop serving time go? The engine driven in-process like the HTTP benchmark.

Replicates closed-fixed (input 256, output 128, ``concurrency`` requests in flight, ``4 x
concurrency`` requests, a new request added as soon as one finishes) without HTTP, and times every
engine step by kind: decode-only, mixed (prefill + decode rows) and prefill-only. Comparing the
total with the HTTP run's throughput isolates the HTTP/async overhead.

Usage: uv run --frozen python benchmarks/scripts/pipeline_profile.py --out <dir> [--graphs]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from dli import envinfo
from dli.engine import LLMEngine, SchedulerConfig
from dli.engine.cuda_graph import BENCHMARK_SHAPES
from dli.model import hub, load_model
from dli.sampling import SamplingParams


def run(model_id: str, concurrency: int, graphs: bool, num_blocks: int) -> dict[str, object]:
    model = load_model(hub.download(model_id), dtype=torch.bfloat16, device="cuda")
    eng = LLMEngine.for_model(
        model,
        num_blocks=num_blocks,
        block_size=16,
        config=SchedulerConfig(max_num_seqs=128, max_num_batched_tokens=2048),
        cuda_graphs=BENCHMARK_SHAPES if graphs else None,
    )
    vocab = model.config.vocab_size
    gen = torch.Generator().manual_seed(concurrency)
    total = 4 * concurrency

    def add(i: int) -> None:
        ids = torch.randint(1000, vocab, (256,), generator=gen).tolist()
        eng.add_request(f"r{i}", ids, SamplingParams(max_tokens=128, ignore_eos=True))

    # warm-up (allocator, cuBLAS, graph capture): a few short requests
    for i in range(8):
        eng.add_request(f"w{i}", [5 + i] * 64, SamplingParams(max_tokens=16, ignore_eos=True))
    while eng.has_unfinished():
        eng.step()
    eng.stats.steps.clear()

    added = 0
    for _ in range(concurrency):
        add(added)
        added += 1
    steps: list[dict[str, float]] = []
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    generated = 0
    while eng.has_unfinished():
        t0 = time.perf_counter()
        outs = eng.step()  # sampling copies tokens to the host: the step is complete on return
        dt = time.perf_counter() - t0
        st = eng.stats.steps[-1]
        kind = (
            "decode"
            if st.num_prefill_tokens == 0
            else "prefill"
            if st.num_decode_tokens == 0
            else "mixed"
        )
        steps.append(
            {
                "kind": kind,
                "s": dt,
                "seqs": st.num_seqs,
                "prefill_tokens": st.num_prefill_tokens,
                "decode_tokens": st.num_decode_tokens,
                "forward_s": st.forward_s,
            }
        )
        generated += sum(len(o.new_token_ids) for o in outs)
        for o in outs:
            if o.finished and added < total:
                add(added)
                added += 1
    wall = time.perf_counter() - t_start

    by_kind: dict[str, dict[str, float]] = {}
    for kind in ("decode", "mixed", "prefill"):
        ks = [s for s in steps if s["kind"] == kind]
        if not ks:
            continue
        by_kind[kind] = {
            "steps": len(ks),
            "time_s": round(sum(s["s"] for s in ks), 3),
            "share_of_wall": round(sum(s["s"] for s in ks) / wall, 3),
            "mean_ms": round(statistics.mean(s["s"] for s in ks) * 1e3, 2),
            "median_ms": round(statistics.median(s["s"] for s in ks) * 1e3, 2),
            "mean_seqs": round(statistics.mean(s["seqs"] for s in ks), 1),
            "mean_prefill_tokens": round(statistics.mean(s["prefill_tokens"] for s in ks), 1),
            "mean_decode_tokens": round(statistics.mean(s["decode_tokens"] for s in ks), 1),
        }
    graph = eng.runner.decode_graphs  # type: ignore[attr-defined]
    return {
        "concurrency": concurrency,
        "graphs": graphs,
        "requests": total,
        "wall_s": round(wall, 3),
        "generated_tokens": generated,
        "output_tokens_per_s": round(generated / wall, 1),
        "steps": len(steps),
        "by_kind": by_kind,
        "graph": graph.summary() if graph is not None else None,
        "mixed_steps_detail": [
            {k: (round(v * 1e3, 2) if k in ("s", "forward_s") else v) for k, v in s.items()}
            for s in steps
            if s["kind"] != "decode"
        ][:60],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--concurrency", default="16,64")
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--num-blocks", type=int, default=100416)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rev = envinfo.current_revision()
    envinfo.main([str(a.out / "env.json")], revision=rev)
    res = [run(a.model, int(c), a.graphs, a.num_blocks) for c in a.concurrency.split(",")]
    for r in res:
        print(
            f"c={r['concurrency']} graphs={r['graphs']} {r['output_tokens_per_s']} tok/s "
            f"wall {r['wall_s']} s"
        )
        for k, v in r["by_kind"].items():  # type: ignore[attr-defined]
            print(f"   {k:8s} {v}")
    name = "pipeline-graphs.json" if a.graphs else "pipeline-eager.json"
    (a.out / name).write_text(json.dumps({"code_revision": rev, "runs": res}, indent=1) + "\n")


if __name__ == "__main__":
    main()
