"""Engine/scheduler overhead and batched-vs-sequential observation (M3).

Part A (engine overhead only): a NullRunner that does no model work (marks KV computed and returns
zero logits) isolates the cost of scheduling + sampling + bookkeeping per step, swept over
concurrency. cProfile hotspots of the largest configuration are saved as profiling evidence.

Part B (real model, SmolLM2-135M on CPU): overhead share of each engine step, and output tokens/s
of the batched engine vs sequential unbatched greedy decoding on the same prompts. Runs are
interleaved A/B/A/B (laptop noise control, ENGINEERING_STANDARDS §7) and report median + range.
This is an M3 observation, NOT the M5 benchmark (no open-loop load, no latency percentiles).

Usage: uv run --frozen python benchmarks/scripts/scheduler_overhead.py [--skip-model]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from dli import envinfo
from dli.engine import LLMEngine, SchedulerConfig, SchedulerOutput
from dli.generate import greedy_generate
from dli.kv import BlockManager
from dli.model import hub, load_model
from dli.sampling import SamplingParams

VOCAB = 32000


class NullRunner:
    def __init__(self, bm: BlockManager) -> None:
        self.bm = bm

    def execute(self, sched: SchedulerOutput) -> torch.Tensor:
        for ch in sched.chunks:
            self.bm.mark_computed(ch.request.request_id, ch.start + ch.num_tokens)
        rows = sum(c.needs_logits for c in sched.chunks)
        return torch.zeros(rows, VOCAB)


def overhead_sweep() -> list[dict[str, float]]:
    results = []
    for concurrency in (8, 32, 128, 256):
        bm = BlockManager(16384, 16, watermark=0.0, debug_checks=False)
        eng = LLMEngine(
            NullRunner(bm),
            bm,
            SchedulerConfig(max_num_seqs=concurrency, max_num_batched_tokens=8192),
        )
        for i in range(concurrency * 4):
            eng.add_request(
                f"r{i}", list(range(i % 97, i % 97 + 64)), SamplingParams(max_tokens=64)
            )
        while eng.has_unfinished():
            eng.step()
        decode_steps = [s for s in eng.stats.steps if s.num_prefill_tokens == 0 and s.num_seqs > 0]
        sched_us = [s.schedule_s * 1e6 for s in decode_steps]
        post_us = [s.postprocess_s * 1e6 for s in decode_steps]
        per_seq = [(s.schedule_s + s.postprocess_s) * 1e6 / s.num_seqs for s in decode_steps]
        results.append(
            {
                "concurrency": concurrency,
                "decode_steps": len(decode_steps),
                "schedule_us_median": round(statistics.median(sched_us), 1),
                "postprocess_us_median": round(statistics.median(post_us), 1),
                "overhead_us_per_seq_median": round(statistics.median(per_seq), 2),
                "overhead_us_per_seq_p95": round(statistics.quantiles(per_seq, n=20)[18], 2),
            }
        )
    return results


def profile_largest(out: Path) -> str:
    bm = BlockManager(16384, 16, watermark=0.0, debug_checks=False)
    eng = LLMEngine(
        NullRunner(bm), bm, SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=8192)
    )
    for i in range(512):
        eng.add_request(f"r{i}", list(range(64)), SamplingParams(max_tokens=64))
    prof = cProfile.Profile()
    prof.enable()
    while eng.has_unfinished():
        eng.step()
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(25)
    (out / "profile_null_runner_256.txt").write_text(_redact_paths(buf.getvalue()))
    return buf.getvalue()


def _redact_paths(text: str) -> str:
    """Strip machine-specific absolute paths (home dirs, venvs) from profiler output.

    Results are published; absolute paths would expose the developer's username and layout.
    """
    # Site-packages first: the venv may live inside the repo, and its prefix must not survive.
    text = re.sub(r"/\S*/site-packages/", "<site-packages>/", text)
    repo = str(Path(__file__).resolve().parents[2])
    return text.replace(repo + "/", "<repo>/")


def model_observation(
    n_prompts: int = 24, new_tokens: int = 32, reps: int = 3
) -> dict[str, object]:
    path = hub.download("HuggingFaceTB/SmolLM2-135M-Instruct")
    model = load_model(path)
    prompts_file = Path(__file__).resolve().parents[2] / "tests" / "parity" / "prompts.jsonl"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    texts = [json.loads(line) for line in prompts_file.read_text().splitlines()]
    prompts = [
        tok(t["text"], add_special_tokens=False)["input_ids"]
        for t in texts
        if t["kind"] == "completion"
    ][:n_prompts]
    params = SamplingParams(max_tokens=new_tokens, ignore_eos=True)
    total_out = n_prompts * new_tokens

    def run_batched() -> tuple[float, float]:
        eng = LLMEngine.for_model(
            model,
            num_blocks=2048,
            block_size=16,
            config=SchedulerConfig(max_num_seqs=n_prompts, max_num_batched_tokens=1024),
        )
        t0 = time.perf_counter()
        outs = eng.generate(prompts, params)
        dt = time.perf_counter() - t0
        assert all(len(o) == new_tokens for o in outs)
        steps = eng.stats.steps
        overhead = sum(s.schedule_s + s.postprocess_s for s in steps) / sum(
            s.total_s for s in steps
        )
        return total_out / dt, overhead

    def run_sequential() -> float:
        t0 = time.perf_counter()
        for p in prompts:
            greedy_generate(model, p, new_tokens)
        return total_out / (time.perf_counter() - t0)

    run_batched()  # warm-up (allocator, thread pools)
    batched, overhead, sequential = [], [], []
    for _ in range(reps):  # interleaved A/B
        tps, ov = run_batched()
        batched.append(tps)
        overhead.append(ov)
        sequential.append(run_sequential())
    return {
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "prompts": n_prompts,
        "new_tokens_each": new_tokens,
        "repetitions": reps,
        "batched_output_tok_s": [round(x, 1) for x in batched],
        "sequential_output_tok_s": [round(x, 1) for x in sequential],
        "batched_median": round(statistics.median(batched), 1),
        "sequential_median": round(statistics.median(sequential), 1),
        "speedup_median": round(statistics.median(batched) / statistics.median(sequential), 2),
        "engine_overhead_share_median": round(statistics.median(overhead), 4),
        "torch_threads": torch.get_num_threads(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    revision = envinfo.current_revision()
    out = (
        Path("benchmarks/results")
        / f"{datetime.now(UTC):%Y-%m-%d}-CPU-LAPTOP-m3-scheduler-overhead"
    )
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {"code_revision": revision, "evidence_label": "CPU-LAPTOP"}
    result["null_runner_sweep"] = overhead_sweep()
    print(json.dumps(result["null_runner_sweep"], indent=1))
    profile_largest(out)
    if not args.skip_model:
        result["smollm2_observation"] = model_observation()
        print(json.dumps(result["smollm2_observation"], indent=1))
    envinfo.main([str(out / "env.json")], revision=revision)
    (out / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
