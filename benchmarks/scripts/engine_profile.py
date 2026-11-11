"""Where does an engine step go? torch.profiler over steady-state decode steps (M5 evidence).

Runs the continuous-batching engine directly (no HTTP) with ``batch`` concurrent requests (input
256, output 128, ignore_eos), warms up past the prefills, then profiles ``steps`` decode steps.

Reports, per batch size:
* step time and its split into schedule / forward / postprocess (the engine's own StepStats);
* operator totals (self CPU time and device time), top 20 by device time;
* CUDA kernel launches per step and calls per step of the attention-path operators;
* a memory-bandwidth floor: bytes of weights + KV read per decode step / peak bandwidth.

Usage: uv run --frozen python benchmarks/scripts/engine_profile.py --model <repo> \
           --out benchmarks/results/<dir> [--batches 16,64] [--steps 20] [--bandwidth-gbs 200]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from dli import envinfo
from dli.engine import LLMEngine, SchedulerConfig
from dli.kv.paged_store import PagedKVStore
from dli.model import hub, load_model
from dli.sampling import SamplingParams


def run(
    model_id: str,
    batch: int,
    steps: int,
    bandwidth_gbs: float,
    attention: str = "batched",
    *,
    cuda_graph: bool = False,
    input_len: int = 256,
    num_blocks: int | None = None,
) -> dict[str, object]:
    path = hub.download(model_id)
    model = load_model(path, dtype=torch.bfloat16, device="cuda")
    cfg = model.config
    eng = LLMEngine.for_model(
        model,
        num_blocks=num_blocks or batch * 30,
        block_size=16,
        config=SchedulerConfig(max_num_seqs=batch, max_num_batched_tokens=2048),
        batched_decode=attention == "batched",
        cuda_graph=(16, 16) if cuda_graph else None,  # the M6 proof-of-concept shape
    )
    g = torch.Generator().manual_seed(0)
    for i in range(batch):
        ids = torch.randint(1000, cfg.vocab_size, (input_len,), generator=g).tolist()
        eng.add_request(f"r{i}", ids, SamplingParams(max_tokens=128, ignore_eos=True))
    # warm-up: finish every prefill and a few decode steps (allocator, kernels, graph capture)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(40):
        eng.step()
    torch.cuda.synchronize()
    # Unprofiled timing first: the profiler itself adds per-launch overhead (~2x on this engine).
    t0 = time.perf_counter()
    for _ in range(steps):
        eng.step()
    torch.cuda.synchronize()
    unprofiled_ms = (time.perf_counter() - t0) / steps * 1e3
    first = len(eng.stats.steps)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            eng.step()
        torch.cuda.synchronize()
    st = eng.stats.steps[first:]
    events = prof.key_averages()
    rows = sorted(events, key=lambda e: e.device_time_total, reverse=True)[:20]
    kernels = sum(e.count for e in events if e.device_type == torch.autograd.DeviceType.CUDA)
    calls = {
        e.key: e.count / steps
        for e in events
        if e.key
        in (
            "aten::scaled_dot_product_attention",
            "aten::index_select",
            "aten::cat",
            "aten::copy_",
            "aten::linear",
            "aten::mm",
            "aten::index_put_",
        )
    }
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    ctx = (
        statistics.mean(r.num_computed_tokens for r in eng.scheduler.running)
        if eng.scheduler.running
        else 0
    )
    kv = batch * ctx * PagedKVStore.bytes_per_block(cfg, 16, torch.bfloat16) / 16
    floor_ms = (weights + kv) / (bandwidth_gbs * 1e9) * 1e3
    step_ms = statistics.mean(s.total_s for s in st) * 1e3
    graph = getattr(eng.runner, "decode_graph", None)
    graph_stats = None
    if graph is not None:
        gs = graph.stats
        graph_stats = {
            "status": gs.status,
            "batch": gs.batch,
            "width_blocks": gs.width,
            "capture_s": round(gs.capture_s, 3),
            "free_before_capture_mib": gs.free_before >> 20,
            "eager_peak_mib": round(gs.eager_peak / 2**20, 1),
            "graph_pool_mib": round(gs.graph_pool / 2**20, 1),
            "replays": gs.replays,
            "fallbacks": gs.fallbacks,
        }
    return {
        "cuda_graph": graph_stats,
        "input_len": input_len,
        "num_blocks": num_blocks or batch * 30,
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() >> 20,
        "memory_reserved_mib": torch.cuda.memory_reserved() >> 20,
        "model": model_id,
        "attention": attention,
        "batch": batch,
        "unprofiled_step_ms": unprofiled_ms,
        "unprofiled_tokens_per_s": batch / unprofiled_ms * 1e3,
        "profiled_steps": steps,
        "step_ms_mean": step_ms,
        "schedule_ms_mean": statistics.mean(s.schedule_s for s in st) * 1e3,
        "forward_ms_mean": statistics.mean(s.forward_s for s in st) * 1e3,
        "postprocess_ms_mean": statistics.mean(s.postprocess_s for s in st) * 1e3,
        "decode_tokens_per_step": statistics.mean(s.num_decode_tokens for s in st),
        "tokens_per_s": statistics.mean(s.num_batched_tokens for s in st)
        / statistics.mean(s.total_s for s in st),
        "cuda_kernels_per_step": kernels / steps,
        "op_calls_per_step": calls,
        "bandwidth_floor_ms": floor_ms,
        "floor_inputs": {
            "weight_bytes": weights,
            "kv_bytes_read": kv,
            "mean_context": ctx,
            "assumed_bandwidth_gbs": bandwidth_gbs,
        },
        "top_ops_by_device_time": [
            {
                "op": e.key,
                "calls_per_step": e.count / steps,
                "device_ms_per_step": e.device_time_total / 1e3 / steps,
                "self_cpu_ms_per_step": e.self_cpu_time_total / 1e3 / steps,
            }
            for e in rows
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batches", default="16,64")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument(
        "--bandwidth-gbs", type=float, default=200.0, help="GA107/A16 per-GPU peak: 200 GB/s"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--cuda-graph", action="store_true", help="enable the batch-16 x 16-block decode graph"
    )
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument(
        "--num-blocks", type=int, default=None, help="KV pool blocks (default: batch x 30)"
    )
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument(
        "--attention",
        choices=["batched", "reference"],
        default="batched",
        help="decode attention path: batched (M6) or the per-sequence reference",
    )
    a = ap.parse_args()
    revision = envinfo.current_revision()
    a.out.mkdir(parents=True, exist_ok=True)
    envinfo.main([str(a.out / "env.json")], revision=revision)
    res = [
        run(
            a.model,
            int(b),
            a.steps,
            a.bandwidth_gbs,
            a.attention,
            cuda_graph=a.cuda_graph,
            input_len=a.input_len,
            num_blocks=a.num_blocks,
        )
        for b in a.batches.split(",")
    ]
    (a.out / f"{a.model.split('/')[-1]}-{a.attention}{a.tag}.json").write_text(
        json.dumps({"code_revision": revision, "runs": res}, indent=1) + "\n"
    )
    for r in res:
        print(
            f"batch {r['batch']}: step {r['step_ms_mean']:.1f} ms "
            f"(forward {r['forward_ms_mean']:.1f}), {r['tokens_per_s']:.0f} tok/s, "
            f"{r['cuda_kernels_per_step']:.0f} kernels/step, "
            f"bandwidth floor {r['bandwidth_floor_ms']:.2f} ms"
        )
        for t in r["top_ops_by_device_time"][:8]:  # type: ignore[index]
            print(
                f"   {t['op'][:48]:48s} {t['calls_per_step']:8.1f} calls  "
                f"{t['device_ms_per_step']:7.2f} ms dev  {t['self_cpu_ms_per_step']:7.2f} ms cpu"
            )


if __name__ == "__main__":
    main()
