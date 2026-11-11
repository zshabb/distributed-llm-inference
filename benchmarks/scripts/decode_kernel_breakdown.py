"""GPU time of one steady-state decode step, by kernel category (M6 audit).

Drives the engine directly (closed-fixed shape: ``batch`` requests, ``input_len`` prompt tokens,
128 output), warms up past prefill (and graph capture), then profiles ``steps`` decode steps and
sums CUDA kernel time per category: attention, KV page gather, KV write, GEMMs, sampling,
elementwise/norm/RoPE and other. Kernels inside a CUDA graph replay are attributed too.

Usage: uv run --frozen python benchmarks/scripts/decode_kernel_breakdown.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from dli.engine import LLMEngine, SchedulerConfig
from dli.engine.cuda_graph import BENCHMARK_SHAPES
from dli.model import hub, load_model
from dli.sampling import SamplingParams

CATEGORIES = [
    ("attention", r"fmha|flash|attention|efficient_attention"),
    ("kv_gather", r"gather|index_select"),
    ("kv_write", r"index_copy|index_put|scatter"),
    ("gemm", r"gemm|cutlass|cublas|splitkreduce|sm80_xmma|ampere_|_tn|_nt"),
    ("sampling", r"softmax|argmax|topk|sort|multinomial|exponential|cumsum|max_kernel"),
    ("norm_rope_elementwise", r"elementwise|vectorized|reduce|mul|add|pow|rsqrt|silu|neg|cat"),
]


def categorize(name: str) -> str:
    low = name.lower()
    for cat, pat in CATEGORIES:
        if re.search(pat, low):
            return cat
    return "other"


def run(model_id: str, batch: int, input_len: int, graphs: bool, steps: int) -> dict[str, object]:
    model = load_model(hub.download(model_id), dtype=torch.bfloat16, device="cuda")
    blocks_per_seq = -(-(input_len + 128) // 16)
    eng = LLMEngine.for_model(
        model,
        num_blocks=batch * blocks_per_seq + 64,
        block_size=16,
        config=SchedulerConfig(max_num_seqs=batch, max_num_batched_tokens=2048),
        cuda_graphs=BENCHMARK_SHAPES if graphs else None,
    )
    g = torch.Generator().manual_seed(0)
    for i in range(batch):
        ids = torch.randint(1000, model.config.vocab_size, (input_len,), generator=g).tolist()
        eng.add_request(f"r{i}", ids, SamplingParams(max_tokens=128, ignore_eos=True))
    while any(r.num_computed_tokens < len(r.prompt_ids) for r in eng.scheduler.running) or (
        eng.scheduler.waiting
    ):
        eng.step()
    for _ in range(5):
        eng.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        eng.step()
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) / steps * 1e3
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            eng.step()
        torch.cuda.synchronize()
    per_cat: dict[str, float] = defaultdict(float)
    kernels: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            dt = e.device_time / 1e3  # ms
            per_cat[categorize(e.name)] += dt
            kernels[e.name][0] += dt
            kernels[e.name][1] += 1
    total = sum(per_cat.values())
    ctx = sum(r.num_computed_tokens for r in eng.scheduler.running) / max(
        1, len(eng.scheduler.running)
    )
    graphs_obj = eng.runner.decode_graphs  # type: ignore[attr-defined]
    return {
        "batch": batch,
        "input_len": input_len,
        "graphs": graphs,
        "mean_context": round(ctx, 1),
        "unprofiled_step_ms": round(step_ms, 3),
        "gpu_ms_per_step": round(total / steps, 3),
        "gpu_ms_by_category": {
            k: round(v / steps, 3) for k, v in sorted(per_cat.items(), key=lambda x: -x[1])
        },
        "top_kernels": [
            {
                "kernel": k[:110],
                "ms_per_step": round(v[0] / steps, 3),
                "calls_per_step": v[1] / steps,
                "category": categorize(k),
            }
            for k, v in sorted(kernels.items(), key=lambda x: -x[1][0])[:18]
        ],
        "graph": graphs_obj.summary() if graphs_obj is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = [
        run(a.model, 64, 256, True, a.steps),
        run(a.model, 16, 256, True, a.steps),
        run(a.model, 64, 1024, False, a.steps),
    ]
    for r in res:
        print(
            f"B={r['batch']} ctx~{r['mean_context']} graphs={r['graphs']}: "
            f"step {r['unprofiled_step_ms']} ms, GPU {r['gpu_ms_per_step']} ms -> "
            f"{r['gpu_ms_by_category']}"
        )
    (a.out / "decode_kernel_breakdown.json").write_text(json.dumps({"runs": res}, indent=1) + "\n")


if __name__ == "__main__":
    main()
