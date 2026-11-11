"""Run a benchmark campaign from a JSON plan, and build its report.

    python -m dli.bench.campaign run benchmarks/plans/<plan>.json --out benchmarks/results/<dir>
    python -m dli.bench.campaign report benchmarks/results/<dir>

Execution order: for each repetition, for each model, the backends run in an order rotated by the
repetition number (ABC, BCA, CAB...), so slow drift over a session does not favour one backend.
Each backend server is started fresh, warmed up with a few unrecorded requests, then runs every
experiment point for that model, then is stopped. For one (repetition, point) every backend
receives the identical request list (same seed).

Open-loop sweeps stop after the first *saturated* point: completion rate < 99%, median TTFT above
``saturation_ttft_s``, or a growing queue (median TTFT of the last third of arrivals > 2x the first
third's + 0.5 s). Further points would only measure an ever-growing queue and cost GPU time.

The first version (used for the 2026-11-11 campaign) also called a point saturated when the
completed request rate was below 90% of the offered rate. That rate divides by a span that includes
the drain tail after the last arrival, so it under-states throughput and stopped sweeps early for
backends with long end-to-end tails; it was replaced by the queue-growth test.

The overload experiment offers ``factor x`` the highest unsaturated rate that backend reached in the
same repetition (spec AC5), or the explicit ``rates`` an overload experiment lists.

Raw per-request records are stored gzip-compressed (latencies already derived; per-chunk times are
reduced to inter-token statistics to keep the repository small).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dli import envinfo
from dli.bench import backends, loadgen
from dli.bench.datasets import OASST1, oasst1_conversations, read_rows
from dli.bench.monitor import Monitor
from dli.bench.workloads import (
    LengthDist,
    Workload,
    load_system_prompts,
    multi_turn,
    shared_prefix,
    synthetic,
)
from dli.model import hub

REPO_ROOT = Path(__file__).resolve().parents[3]


def _dist(spec: dict[str, Any]) -> LengthDist:
    if "fixed" in spec:
        return LengthDist.fixed(int(spec["fixed"]))
    return LengthDist.lognormal(
        int(spec["median"]), float(spec["sigma"]), int(spec["lo"]), int(spec["hi"])
    )


def build_workload(spec: dict[str, Any], *, n: int, vocab: int, seed: int, name: str) -> Workload:
    kind = spec["type"]
    if kind == "synthetic":
        return synthetic(
            name, n, _dist(spec["input"]), _dist(spec["output"]), vocab=vocab, seed=seed
        )
    if kind == "shared_prefix":
        return shared_prefix(
            name,
            n,
            num_prefixes=int(spec["num_prefixes"]),
            prefix_len=int(spec["prefix_len"]),
            suffix_len=_dist(spec["suffix"]),
            output_len=_dist(spec["output"]),
            zipf_s=float(spec["zipf_s"]),
            vocab=vocab,
            seed=seed,
        )
    if kind == "multiturn":
        convs = oasst1_conversations(read_rows(OASST1), min_turns=int(spec.get("min_turns", 2)))
        prompts = load_system_prompts(REPO_ROOT / spec["system_prompts"])
        return multi_turn(
            name,
            convs,
            num_sessions=n,
            system_prompts=prompts,
            max_turns=int(spec["max_turns"]),
            output_tokens=int(spec["output_tokens"]),
            seed=seed,
        )
    raise ValueError(f"unknown workload type {kind!r}")


def _point_n(exp: dict[str, Any], point: float) -> int:
    if exp["mode"] == "closed":
        return (
            int(exp.get("n_per_concurrency", 4) * point + exp.get("n_extra", 0))
            if "n" not in exp
            else int(exp["n"])
        )
    n = int(point * float(exp.get("duration_s", 60)))
    return max(int(exp.get("min_n", 30)), min(int(exp.get("max_n", 600)), n))


def saturated(m: dict[str, Any], offered: float | None, ttft_limit: float) -> bool:
    """Incomplete, too slow, or a queue that grows through the run (``offered`` kept for logs)."""
    if m["completion_rate"] < 0.99:
        return True
    ttft = m.get("ttft_s") or {}
    if ttft and ttft["p50"] > ttft_limit:
        return True
    first, last = m.get("ttft_first_third_p50_s"), m.get("ttft_last_third_p50_s")
    return first is not None and last is not None and last > 2 * first + 0.5


def slim(rec: loadgen.RequestRecord) -> dict[str, Any]:
    d = rec.as_dict()
    ct = d.pop("chunk_times")
    gaps = np.diff(ct) if len(ct) > 1 else np.array([])
    d["chunks"] = len(ct)
    d["itl_p50"] = float(np.median(gaps)) if len(gaps) else None
    d["itl_max"] = float(gaps.max()) if len(gaps) else None
    return d


@dataclass
class Ctx:
    plan: dict[str, Any]
    out: Path
    log: Any


def _run_point(
    ctx: Ctx,
    srv: backends.Running,
    exp_id: str,
    exp: dict[str, Any],
    point: float,
    *,
    vocab: int,
    rep: int,
    headers: dict[str, str],
    tag: str,
    stream: bool,
) -> dict[str, Any]:
    plan = ctx.plan
    seed = 1000 * rep + round(point * 100)
    n = _point_n(exp, point)
    wl = build_workload(exp["workload"], n=n, vocab=vocab, seed=seed, name=exp_id)
    target = loadgen.Target(
        srv.base_url,
        srv.model_name,
        stream=stream,
        headers=headers,
        request_timeout_s=float(plan.get("request_timeout_s", 600)),
    )
    mon = Monitor(
        interval_s=0.5,
        pid=srv.pid,
        metrics_url=srv.metrics_url,
        metric_names=srv.metric_names,
        gpu=plan.get("device", "cpu").startswith("cuda"),
    )
    t_start = time.time()
    with mon:
        if exp["mode"] == "closed":
            res = asyncio.run(loadgen.closed_loop(target, wl.requests, int(point)))
        elif exp["mode"] == "open":
            res = asyncio.run(
                loadgen.open_loop(
                    target, wl.requests, point, seed=seed, drain_s=float(exp.get("drain_s", 120))
                )
            )
        else:
            res = asyncio.run(
                loadgen.sessions_open_loop(
                    target,
                    wl.sessions,
                    point,
                    think_s=float(exp.get("think_s", 1.0)),
                    seed=seed,
                    drain_s=float(exp.get("drain_s", 180)),
                )
            )
    m = loadgen.run_metrics(res)
    m.update(
        {
            "experiment": exp_id,
            "point": point,
            "tag": tag,
            "rep": rep,
            "backend": srv.name,
            "model": srv.model_name,
            "workload": wl.params,
            "resources": mon.summary(),
            "started_unix": t_start,
            "headers": headers,
        }
    )
    rel = Path(srv.model_name.replace("/", "_")) / srv.name / f"rep{rep}"
    d = ctx.out / "runs" / rel
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{exp_id}@{point:g}{('-' + tag) if tag else ''}"
    (d / f"{stem}.json").write_text(json.dumps(m, indent=1) + "\n")
    with gzip.open(d / f"{stem}.records.jsonl.gz", "wt") as f:
        for r in res.records:
            f.write(json.dumps(slim(r)) + "\n")
    ttft = (m.get("ttft_s") or {}).get("p50")
    line = (
        f"{time.strftime('%H:%M:%S')} rep{rep} {srv.model_name} {srv.name} {stem}: "
        f"ok {m['ok']}/{m['requests']} "
        f"tok/s {m['output_tok_per_s']:.1f} ttft_p50 {ttft if ttft is None else round(ttft, 3)} "
        f"wall {res.wall_s:.0f}s"
    )
    print(line, flush=True)
    ctx.log.write(
        json.dumps({"event": "point", "file": str(rel / stem), "wall_s": res.wall_s}) + "\n"
    )
    ctx.log.flush()
    return m


def _warm_up(srv: backends.Running, vocab: int, stream: bool) -> None:
    wl = synthetic("warmup", 8, LengthDist.fixed(64), LengthDist.fixed(16), vocab=vocab, seed=99)
    target = loadgen.Target(srv.base_url, srv.model_name, stream=stream)
    res = asyncio.run(loadgen.closed_loop(target, wl.requests, 4))
    bad = [r for r in res.records if r.status != "ok"]
    if bad:
        raise RuntimeError(f"warm-up failed on {srv.name}: {bad[0].error}")


def run(
    plan_path: Path, out: Path, only_models: set[str] | None, only_backends: set[str] | None
) -> None:
    plan = json.loads(plan_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    revision = envinfo.current_revision()
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    envinfo.main([str(out / "env.json")], revision=revision)
    ttft_limit = float(plan.get("saturation_ttft_s", 5.0))
    with (out / "campaign_log.jsonl").open("a") as log:
        ctx = Ctx(plan, out, log)
        log.write(json.dumps({"event": "start", "unix": time.time(), "revision": revision}) + "\n")
        for rep in range(int(plan["reps"])):
            for model in plan["models"]:
                if only_models and model["name"] not in only_models:
                    continue
                model_dir = Path(model["path"]) if "path" in model else hub.download(model["repo"])
                vocab = int(json.loads((model_dir / "config.json").read_text())["vocab_size"])
                names = [b for b in model["backends"] if not only_backends or b in only_backends]
                k = rep % max(len(names), 1)
                for bname in names[k:] + names[:k]:
                    settings = {
                        **plan["backends"][bname],
                        **model.get("overrides", {}).get(bname, {}),
                    }
                    kind = str(settings.get("kind", bname))  # variants, e.g. dli-noprefix
                    stream = kind != "hf"
                    srv = backends.start(
                        kind,
                        model_dir=model_dir,
                        model_name=model["name"],
                        port=int(plan.get("port", 8300)),
                        dtype=plan["dtype"],
                        device=plan["device"],
                        max_num_seqs=int(plan["max_num_seqs"]),
                        max_num_batched_tokens=int(plan["max_num_batched_tokens"]),
                        log_dir=out / "server-logs",
                        settings=settings,
                    )
                    srv.name = bname  # results and reports use the plan's backend name
                    log.write(
                        json.dumps(
                            {
                                "event": "server",
                                "rep": rep,
                                "model": model["name"],
                                **srv.describe(),
                            }
                        )
                        + "\n"
                    )
                    print(
                        f"started {bname} for {model['name']} in {srv.startup_s:.0f}s", flush=True
                    )
                    try:
                        _warm_up(srv, vocab, stream)
                        sat: dict[str, float] = {}
                        for exp_id in model["experiments"]:
                            exp = plan["experiments"][exp_id]
                            if bname not in exp.get("backends", [bname]):
                                continue
                            if exp["mode"] == "overload":
                                bexp = {**plan["experiments"][exp["base"]], "mode": "open"}
                                bexp.update(exp.get("overrides", {}))
                                if "rates" in exp:  # explicit, e.g. 2x a measured saturation
                                    points = [(float(r), f"r{float(r):g}") for r in exp["rates"]]
                                else:
                                    base = sat.get(exp["base"])
                                    if base is None:
                                        print(
                                            f"skip {exp_id}: no unsaturated rate for {bname}",
                                            flush=True,
                                        )
                                        continue
                                    points = [
                                        (round(base * f, 3), f"x{f:g}") for f in exp["factors"]
                                    ]
                                for rate, label in points:
                                    for var in exp["variants"]:
                                        _run_point(
                                            ctx,
                                            srv,
                                            exp_id,
                                            bexp,
                                            rate,
                                            vocab=vocab,
                                            rep=rep,
                                            headers=var.get("headers", {}),
                                            tag=f"{label}-{var['name']}",
                                            stream=stream,
                                        )
                                continue
                            for point in exp["points"]:
                                m = _run_point(
                                    ctx,
                                    srv,
                                    exp_id,
                                    exp,
                                    float(point),
                                    vocab=vocab,
                                    rep=rep,
                                    headers=exp.get("headers", {}),
                                    tag="",
                                    stream=stream,
                                )
                                if exp["mode"] in ("open", "sessions"):
                                    offered = float(point) if exp["mode"] == "open" else None
                                    if saturated(m, offered, ttft_limit):
                                        break
                                    sat[exp_id] = float(point)
                    finally:
                        backends.stop(srv)
        log.write(json.dumps({"event": "end", "unix": time.time()}) + "\n")


# ----------------------------------------------------------------------------------- report


def _get(m: dict[str, Any], path: str) -> float | None:
    cur: Any = m
    for part in path.split("."):
        if not isinstance(cur, dict) or cur.get(part) is None:
            return None
        cur = cur[part]
    return float(cur)


COLUMNS = [
    ("output_tok_per_s", "out tok/s"),
    ("completed_req_per_s", "req/s"),
    ("completion_rate", "completed"),
    ("ttft_s.p50", "TTFT p50"),
    ("ttft_s.p99", "TTFT p99"),
    ("tpot_s.p50", "TPOT p50"),
    ("tpot_s.p99", "TPOT p99"),
    ("itl_s.p99", "ITL p99"),
    ("e2e_s.p99", "E2E p99"),
    ("rejection_latency_s.p99", "reject p99"),
    ("resources.gpu_mem_used_max", "GPU mem max"),
    ("client_lag_s.p99", "client lag p99"),
]


def aggregate(results_dir: Path) -> list[dict[str, Any]]:
    """Median and range across repetitions for every (model, backend, experiment, point, tag)."""
    groups: dict[tuple[str, str, str, float, str], list[dict[str, Any]]] = {}
    for f in sorted((results_dir / "runs").rglob("*.json")):
        m = json.loads(f.read_text())
        groups.setdefault(
            (m["model"], m["backend"], m["experiment"], float(m["point"]), m.get("tag", "")), []
        ).append(m)
    rows = []
    for (model, backend, exp, point, tag), ms in sorted(groups.items()):
        row: dict[str, Any] = {
            "model": model,
            "backend": backend,
            "experiment": exp,
            "point": point,
            "tag": tag,
            "reps": len(ms),
        }
        for key, _ in COLUMNS:
            vals = [v for v in (_get(m, key) for m in ms) if v is not None and not math.isnan(v)]
            if vals:
                row[key] = {"median": float(np.median(vals)), "min": min(vals), "max": max(vals)}
        rows.append(row)
    return rows


def _fmt(v: dict[str, float] | None, key: str) -> str:
    if not v:
        return "-"
    med = v["median"]
    if key == "completion_rate":
        return f"{100 * med:.1f}%"
    if key.endswith("gpu_mem_used_max"):
        return f"{med / 2**30:.2f} GiB"
    if "tok_per_s" in key or "req_per_s" in key:
        spread = f" [{v['min']:.1f}-{v['max']:.1f}]" if v["max"] > v["min"] else ""
        return f"{med:.1f}{spread}"
    return f"{med * 1000:.0f} ms" if med < 10 else f"{med:.1f} s"


def report(results_dir: Path) -> str:
    rows = aggregate(results_dir)
    (results_dir / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = [
        f"# Summary: {results_dir.name}",
        "",
        "Median across repetitions; throughput shows [min-max].",
        "Latencies run from the scheduled arrival (open loop) or the send time (closed loop).",
        "Generated by `python -m dli.bench.campaign report`; do not edit by hand.",
        "",
    ]
    for exp in sorted({r["experiment"] for r in rows}):
        lines += [
            f"## {exp}",
            "",
            "| model | backend | point | tag | reps | " + " | ".join(h for _, h in COLUMNS) + " |",
            "|" + "---|" * (5 + len(COLUMNS)),
        ]
        for r in (r for r in rows if r["experiment"] == exp):
            cells = [r["model"], r["backend"], f"{r['point']:g}", r["tag"] or "", str(r["reps"])]
            cells += [_fmt(r.get(k), k) for k, _ in COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines)
    (results_dir / "SUMMARY.md").write_text(text)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dli-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("plan", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--models", default=None, help="comma-separated model names from the plan")
    r.add_argument("--backends", default=None, help="comma-separated backend names")
    rp = sub.add_parser("report")
    rp.add_argument("results", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(
            a.plan,
            a.out,
            set(a.models.split(",")) if a.models else None,
            set(a.backends.split(",")) if a.backends else None,
        )
        print(report(a.out))
    else:
        print(report(a.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
