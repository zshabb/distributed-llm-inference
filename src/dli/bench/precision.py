"""bf16 numerical fidelity and output-quality study (``docs/evaluation-plan-m5.md``).

Four configurations, all fed the **same tokens** (teacher forcing), so every step compares like with
like and one rounding flip cannot cascade into a different continuation:

* **A** fp32, unbatched (contiguous KV, the M1 path): the reference.
* **B** bf16, unbatched: the effect of precision alone.
* **C** bf16, continuous-batching engine (real scheduler: chunked prefill, batching, preemption):
  precision plus batching, i.e. what the server does.
* **D** fp32, engine: control. Must agree with A (re-verifies M3 under teacher forcing).

Teacher forcing inside the engine uses :class:`TeacherForcingRunner`, a wrapper around the real
``ModelRunner``: it records each sampling row's logits, then replaces the row with a one-hot row of
the forced token, so the engine's own greedy sampler appends exactly the reference token. Nothing in
the engine changes, and preemption/recompute happens exactly as in serving.

Parts:
1. *fidelity*: the 100 M1 prompts x 128 tokens; per step top-1 agreement, KL(A||X), max |logit|
   error, agreement restricted to steps whose reference top-2 gap exceeds one bf16 ulp.
2. *perplexity*: WikiText-103 test, first ``ppl_windows`` windows of 512 tokens; the last 256 tokens
   of each window are scored given the first 256 (in C those 256 are decode steps).
3. *arc*: ARC-Easy test (all items), log-likelihood scoring of each answer choice; accuracy with
   exact McNemar tests and paired difference CIs.

Usage: ``python -m dli.bench.precision --model <repo> --parts fidelity,ppl,arc --out DIR``
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from dli import envinfo
from dli.engine import LLMEngine, SchedulerConfig
from dli.engine.scheduler import SchedulerOutput
from dli.model import LlamaForCausalLM, hub, load_model
from dli.model.kv_cache import ContiguousKVCache
from dli.sampling import SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[3]


# M3 configuration that produced the failed bf16 parity result: small token budget (chunked
# prefill), 48 sequences, KV pool at 25% of demand (preemption).
@dataclass(frozen=True)
class EngineSetup:
    max_num_seqs: int
    max_num_batched_tokens: int
    pool_fraction: float  # KV pool size as a fraction of the total demand of all sequences


M3_ENGINE = EngineSetup(max_num_seqs=48, max_num_batched_tokens=256, pool_fraction=0.25)
# Production-like configuration for the quality parts: ample pool, larger batches.
SERVE_ENGINE = EngineSetup(max_num_seqs=128, max_num_batched_tokens=2048, pool_fraction=1.5)


# ------------------------------------------------------------------------------- metrics


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 numbers at |x| (8 significant bits)."""
    mag = x.abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


def compare(ref: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-row agreement and divergence of logits ``x`` from reference logits ``ref`` ([R, V])."""
    r, y = ref.float(), x.float()
    lr, ly = F.log_softmax(r, -1), F.log_softmax(y, -1)
    kl = (lr.exp() * (lr - ly)).sum(-1).clamp_min(0.0)
    top2 = torch.topk(r, 2, dim=-1)
    gap = top2.values[:, 0] - top2.values[:, 1]
    return {
        "agree": (r.argmax(-1) == y.argmax(-1)),
        "kl": kl,
        "max_abs_err": (r - y).abs().amax(-1),
        "ref_gap": gap,
        "beyond_ulp": gap > bf16_ulp(top2.values[:, 0]),
    }


@dataclass
class StepLog:
    """Per-step comparison results for one configuration, keyed by (sequence index, step)."""

    rows: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    # Expected 0: after preemption the engine replays known tokens as a prefill without sampling,
    # so each step's row is produced once. Counted as a guard against double-scoring.
    duplicates: int = 0

    def add(self, seq: int, steps: Sequence[int], m: dict[str, torch.Tensor]) -> None:
        cpu = {k: v.detach().float().cpu().tolist() for k, v in m.items()}
        for i, k in enumerate(steps):
            if (seq, k) in self.rows:
                self.duplicates += 1
            self.rows[(seq, k)] = {name: vals[i] for name, vals in cpu.items()}

    def summary(self) -> dict[str, Any]:
        rows = list(self.rows.values())
        agree = np.array([bool(r["agree"]) for r in rows])
        beyond = np.array([bool(r["beyond_ulp"]) for r in rows])
        kl = np.array([r["kl"] for r in rows])
        err = np.array([r["max_abs_err"] for r in rows])
        dis = [r for r in rows if not r["agree"]]
        return {
            "steps": len(rows),
            "top1_agreement": float(agree.mean()),
            "disagreements": len(dis),
            "steps_beyond_1ulp": int(beyond.sum()),
            "top1_agreement_beyond_1ulp": float(agree[beyond].mean()) if beyond.any() else None,
            "disagreements_beyond_1ulp": int((~agree & beyond).sum()),
            "max_ref_gap_at_disagreement": max((r["ref_gap"] for r in dis), default=None),
            "kl_mean": float(kl.mean()),
            "kl_p50": float(np.percentile(kl, 50)),
            "kl_p99": float(np.percentile(kl, 99)),
            "kl_max": float(kl.max()),
            "max_abs_err_p50": float(np.percentile(err, 50)),
            "max_abs_err_p99": float(np.percentile(err, 99)),
            "duplicate_rows": self.duplicates,
        }

    def per_sequence_mean_kl(self, n: int) -> list[float]:
        by: dict[int, list[float]] = {}
        for (s, _), r in self.rows.items():
            by.setdefault(s, []).append(r["kl"])
        return [float(np.mean(by[i])) for i in range(n)]


# ------------------------------------------------------------------------------- execution paths


@torch.inference_mode()
def reference_greedy(
    model: LlamaForCausalLM, ids: Sequence[int], steps: int
) -> tuple[list[int], torch.Tensor]:
    """A: greedy decoding on the contiguous path; returns tokens and each step's logits (CPU)."""
    cfg, dev = model.config, model.inv_freq.device
    cache = ContiguousKVCache(
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        len(ids) + steps,
        model.model.embed_tokens.weight.dtype,
        dev,
    )
    logits = model(
        torch.tensor(ids, device=dev),
        torch.arange(len(ids), device=dev),
        cache,
        last_token_only=True,
    )[-1]
    toks, rows = [], []
    for _ in range(steps):
        rows.append(logits.float().cpu())
        nxt = int(torch.argmax(logits))
        toks.append(nxt)
        pos = torch.tensor([cache.length], device=dev)
        logits = model(torch.tensor([nxt], device=dev), pos, cache, last_token_only=True)[-1]
    return toks, torch.stack(rows)


@torch.inference_mode()
def forced_unbatched(
    model: LlamaForCausalLM, ids: Sequence[int], forced: Sequence[int]
) -> torch.Tensor:
    """B (or A replayed): decode step by step on the contiguous path, feeding ``forced`` tokens."""
    cfg, dev = model.config, model.inv_freq.device
    cache = ContiguousKVCache(
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        len(ids) + len(forced),
        model.model.embed_tokens.weight.dtype,
        dev,
    )
    logits = model(
        torch.tensor(ids, device=dev),
        torch.arange(len(ids), device=dev),
        cache,
        last_token_only=True,
    )[-1]
    rows = []
    for tok in forced:
        rows.append(logits)
        pos = torch.tensor([cache.length], device=dev)
        logits = model(torch.tensor([tok], device=dev), pos, cache, last_token_only=True)[-1]
    return torch.stack(rows)


@torch.inference_mode()
def full_sequence_logits(model: LlamaForCausalLM, ids: Sequence[int]) -> torch.Tensor:
    """One forward over the whole sequence (prefill shape): logits for every position."""
    dev = model.inv_freq.device
    return model(torch.tensor(ids, device=dev), torch.arange(len(ids), device=dev))  # type: ignore[no-any-return]


RowSink = Callable[[list[tuple[int, int]], torch.Tensor], None]


class TeacherForcingRunner:
    """Wraps a ``Runner``: records each sampling row, then forces the next token to ``forced``."""

    def __init__(
        self, inner: Any, forced: dict[str, list[int]], index: dict[str, int], sink: RowSink
    ) -> None:
        self.inner, self.forced, self.index, self.sink = inner, forced, index, sink

    def execute(self, sched: SchedulerOutput) -> torch.Tensor:
        logits: torch.Tensor = self.inner.execute(sched)
        rows = [c for c in sched.chunks if c.needs_logits]
        keys = [
            (self.index[c.request.request_id], c.start + c.num_tokens - len(c.request.prompt_ids))
            for c in rows
        ]
        if rows:
            self.sink(keys, logits)
        out = torch.full_like(logits, float("-inf"))
        for i, c in enumerate(rows):
            out[i, self.forced[c.request.request_id][keys[i][1]]] = 0.0
        return out


def engine_forced(
    model: LlamaForCausalLM,
    prompts: Sequence[Sequence[int]],
    forced: Sequence[Sequence[int]],
    sink: RowSink,
    setup: EngineSetup,
    *,
    block_size: int = 16,
) -> dict[str, Any]:
    """C/D: run all sequences through the continuous-batching engine with teacher forcing."""
    max_num_seqs, max_num_batched_tokens = setup.max_num_seqs, setup.max_num_batched_tokens
    pool_fraction = setup.pool_fraction
    need = [-(-(len(p) + len(f)) // block_size) for p, f in zip(prompts, forced, strict=True)]
    num_blocks = max(int(sum(need) * pool_fraction), max(need) + 1)
    eng = LLMEngine.for_model(
        model,
        num_blocks=num_blocks,
        block_size=block_size,
        config=SchedulerConfig(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
        ),
    )
    rids = [f"s{i}" for i in range(len(prompts))]
    eng.runner = TeacherForcingRunner(
        eng.runner,
        {r: list(f) for r, f in zip(rids, forced, strict=True)},
        {r: i for i, r in enumerate(rids)},
        sink,
    )
    reqs = [
        eng.add_request(r, list(p), SamplingParams(max_tokens=len(f), ignore_eos=True))
        for r, p, f in zip(rids, prompts, forced, strict=True)
    ]
    t0 = time.perf_counter()
    while eng.has_unfinished():
        eng.step()
    for req, f in zip(reqs, forced, strict=True):
        if req.output_ids != list(f):
            raise AssertionError(f"teacher forcing failed for {req.request_id}")
    return {
        "num_blocks": num_blocks,
        "block_size": block_size,
        "steps": len(eng.stats.steps),
        "preemptions": sum(r.num_preemptions for r in reqs),
        "seconds": time.perf_counter() - t0,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "pool_fraction": pool_fraction,
    }


# ------------------------------------------------------------------------------- part 1: fidelity


def fidelity(
    models: dict[str, LlamaForCausalLM], prompts: Sequence[Sequence[int]], steps: int
) -> dict[str, Any]:
    """Teacher-forced per-step fidelity of B, C, D against A on the given prompts."""
    a_toks, a_logits = [], []
    t0 = time.perf_counter()
    for p in prompts:
        t, lg = reference_greedy(models["fp32"], p, steps)
        a_toks.append(t)
        a_logits.append(lg)
    timings = {"A_s": time.perf_counter() - t0}
    dev = models["fp32"].inv_freq.device
    logs = {k: StepLog() for k in ("B", "C", "D")}

    t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        rows = forced_unbatched(models["bf16"], p, a_toks[i])
        logs["B"].add(i, range(steps), compare(a_logits[i].to(dev), rows))
    timings["B_s"] = time.perf_counter() - t0

    def sink_for(log: StepLog) -> RowSink:
        def sink(keys: list[tuple[int, int]], lg: torch.Tensor) -> None:
            ref = torch.stack([a_logits[s][k] for s, k in keys]).to(dev)
            m = compare(ref, lg)
            for j, (s, k) in enumerate(keys):
                log.add(s, [k], {n: v[j : j + 1] for n, v in m.items()})

        return sink

    engines = {}
    for name, key in (("C", "bf16"), ("D", "fp32")):
        engines[name] = engine_forced(models[key], prompts, a_toks, sink_for(logs[name]), M3_ENGINE)
        timings[f"{name}_s"] = engines[name]["seconds"]

    from scipy.stats import wilcoxon

    from dli.bench.stats import paired_bootstrap_ci

    kb = logs["B"].per_sequence_mean_kl(len(prompts))
    kc = logs["C"].per_sequence_mean_kl(len(prompts))
    diff, lo, hi = paired_bootstrap_ci(kb, kc)
    w = wilcoxon(kc, kb)
    return {
        "prompts": len(prompts),
        "steps_per_prompt": steps,
        "B_bf16_unbatched": logs["B"].summary(),
        "C_bf16_engine": logs["C"].summary(),
        "D_fp32_engine": logs["D"].summary(),
        "engine_config": engines,
        "batching_effect_mean_kl_C_minus_B": {
            "mean": diff,
            "ci95": [lo, hi],
            "wilcoxon_p": float(w.pvalue),
            "mean_kl_B": float(np.mean(kb)),
            "mean_kl_C": float(np.mean(kc)),
        },
        "per_prompt_mean_kl": {"B": kb, "C": kc},
        "timings_s": timings,
    }


# ------------------------------------------------------------------------------- part 2: perplexity


def wikitext_windows(tokenizer: Any, n: int, length: int = 512) -> list[list[int]]:
    from dli.bench.datasets import WIKITEXT103_TEST, read_rows

    text = "".join(str(r["text"]) for r in read_rows(WIKITEXT103_TEST))
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    windows = [ids[i : i + length] for i in range(0, len(ids) - length + 1, length)]
    return [[int(t) for t in w] for w in windows[:n]]


def _nll(logits: torch.Tensor, targets: Sequence[int]) -> float:
    lp = F.log_softmax(logits.float(), -1)
    t = torch.tensor(list(targets), device=logits.device)
    return float(-lp.gather(1, t[:, None]).sum())


def perplexity(
    models: dict[str, LlamaForCausalLM], windows: Sequence[Sequence[int]], context: int
) -> dict[str, Any]:
    """Per-window NLL of tokens ``context..end`` for A, B (prefill shape) and C (engine)."""
    nll: dict[str, list[float]] = {"A": [], "B": [], "C": []}
    for w in windows:
        targets = w[context:]
        for name, key in (("A", "fp32"), ("B", "bf16")):
            lg = full_sequence_logits(models[key], w)[context - 1 : len(w) - 1]
            nll[name].append(_nll(lg, targets))
    c_rows: dict[int, dict[int, float]] = {}

    def sink(keys: list[tuple[int, int]], lg: torch.Tensor) -> None:
        lp = F.log_softmax(lg.float(), -1)
        for j, (s, k) in enumerate(keys):
            c_rows.setdefault(s, {})[k] = float(-lp[j, windows[s][context + k]])

    cfg = engine_forced(
        models["bf16"],
        [w[:context] for w in windows],
        [w[context:] for w in windows],
        sink,
        SERVE_ENGINE,
    )
    nll["C"] = [sum(c_rows[i].values()) for i in range(len(windows))]
    tokens = sum(len(w) - context for w in windows)

    from dli.bench.stats import paired_bootstrap_ci

    out: dict[str, Any] = {
        "windows": len(windows),
        "context": context,
        "scored_tokens": tokens,
        "engine_config": cfg,
    }
    for name in nll:
        out[f"ppl_{name}"] = math.exp(sum(nll[name]) / tokens)
    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        per_tok_x = [v / (len(w) - context) for v, w in zip(nll[x], windows, strict=True)]
        per_tok_y = [v / (len(w) - context) for v, w in zip(nll[y], windows, strict=True)]
        d, lo, hi = paired_bootstrap_ci(per_tok_x, per_tok_y)
        out[f"ratio_{y}_over_{x}"] = {"ratio": math.exp(d), "ci95": [math.exp(lo), math.exp(hi)]}
    out["per_window_nll"] = nll
    return out


# ------------------------------------------------------------------------------- part 3: ARC-Easy


def arc_items(tokenizer: Any) -> list[tuple[list[int], list[list[int]], int]]:
    """(context ids, choice continuation ids, gold index) for every ARC-Easy test item."""
    from dli.bench.datasets import ARC_EASY_TEST, read_rows

    items = []
    for r in read_rows(ARC_EASY_TEST):
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        ctx = tokenizer(f"Question: {r['question']}\nAnswer:", add_special_tokens=False)[
            "input_ids"
        ]
        conts = [tokenizer(f" {t}", add_special_tokens=False)["input_ids"] for t in texts]
        items.append(
            (
                [int(t) for t in ctx],
                [[int(t) for t in c] for c in conts],
                labels.index(r["answerKey"]),
            )
        )
    return items


def arc(
    models: dict[str, LlamaForCausalLM], items: Sequence[tuple[list[int], list[list[int]], int]]
) -> dict[str, Any]:
    flat = [
        (i, j, ctx, cont) for i, (ctx, conts, _) in enumerate(items) for j, cont in enumerate(conts)
    ]
    score: dict[str, dict[tuple[int, int], float]] = {"A": {}, "B": {}, "C": {}}
    for name, key in (("A", "fp32"), ("B", "bf16")):
        for i, j, ctx, cont in flat:
            lg = full_sequence_logits(models[key], ctx + cont)[
                len(ctx) - 1 : len(ctx) + len(cont) - 1
            ]
            score[name][(i, j)] = -_nll(lg, cont)

    c_rows: dict[int, dict[int, float]] = {}

    def sink(keys: list[tuple[int, int]], lg: torch.Tensor) -> None:
        lp = F.log_softmax(lg.float(), -1)
        for r, (s, k) in enumerate(keys):
            c_rows.setdefault(s, {})[k] = float(lp[r, flat[s][3][k]])

    cfg = engine_forced(
        models["bf16"], [f[2] for f in flat], [f[3] for f in flat], sink, SERVE_ENGINE
    )
    for s, (i, j, _, _) in enumerate(flat):
        score["C"][(i, j)] = sum(c_rows[s].values())

    def pick(name: str, i: int, k: int) -> int:
        return max(range(k), key=lambda j: score[name][(i, j)])

    correct = {
        name: [int(pick(name, i, len(conts)) == gold) for i, (_, conts, gold) in enumerate(items)]
        for name in score
    }
    from dli.bench.stats import mcnemar_exact, paired_proportion_diff_ci

    n = len(items)
    out: dict[str, Any] = {
        "items": n,
        "choices_scored": len(flat),
        "engine_config": cfg,
        "accuracy": {k: sum(v) / n for k, v in correct.items()},
    }
    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        only_x = sum(a and not b for a, b in zip(correct[x], correct[y], strict=True))
        only_y = sum(b and not a for a, b in zip(correct[x], correct[y], strict=True))
        lo, hi = paired_proportion_diff_ci(only_x, only_y, n)
        out[f"{y}_vs_{x}"] = {
            "diff": (only_y - only_x) / n,
            "ci95": [lo, hi],
            "only_x": only_x,
            "only_y": only_y,
            "mcnemar_p": mcnemar_exact(only_x, only_y),
        }
    out["per_item_correct"] = correct
    return out


# ------------------------------------------------------------------------------- CLI


def m1_prompt_ids(tokenizer: Any) -> list[list[int]]:
    """The 100 M1 parity prompts, tokenized exactly as the parity suite does."""
    out = []
    for line in (REPO_ROOT / "tests" / "parity" / "prompts.jsonl").read_text().splitlines():
        p = json.loads(line)
        if p["kind"] == "chat":
            ids = tokenizer.apply_chat_template(
                p["messages"], add_generation_prompt=True, tokenize=True
            )
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
        else:
            ids = tokenizer(p["text"], add_special_tokens=False)["input_ids"]
        out.append([int(t) for t in ids])
    return out


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(prog="dli-precision")
    ap.add_argument("--model", required=True, help="pinned repo id or a local checkpoint directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--parts", default="fidelity,ppl,arc")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--ppl-windows", type=int, default=200)
    a = ap.parse_args(argv)
    revision = envinfo.current_revision()
    path = Path(a.model) if Path(a.model).is_dir() else hub.download(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    models = {
        "fp32": load_model(path, dtype=torch.float32, device=a.device),
        "bf16": load_model(path, dtype=torch.bfloat16, device=a.device),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    envinfo.main([str(a.out / "env.json")], revision=revision)
    result: dict[str, Any] = {
        "model": a.model,
        "revision": hub.PINNED_REVISIONS.get(a.model),
        "code_revision": revision,
        "device": a.device,
    }
    name = a.model.split("/")[-1]
    for part in a.parts.split(","):
        t0 = time.perf_counter()
        if part == "fidelity":
            result[part] = fidelity(models, m1_prompt_ids(tok)[: a.prompts], a.steps)
        elif part == "ppl":
            result[part] = perplexity(models, wikitext_windows(tok, a.ppl_windows), context=256)
        elif part == "arc":
            result[part] = arc(models, arc_items(tok))
        else:
            raise SystemExit(f"unknown part {part!r}")
        result[part]["wall_s"] = time.perf_counter() - t0
        (a.out / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(f"{name} {part} done in {result[part]['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
