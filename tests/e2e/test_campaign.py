"""A whole campaign on the tiny model: real dli and HF servers, plan -> raw results -> report."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from dli.bench import campaign

from .conftest import _free_port


def test_campaign_runs_backends_and_reports(tiny_dir: Path, tmp_path: Path) -> None:
    plan = {
        "name": "tiny",
        "label": "TEST",
        "device": "cpu",
        "dtype": "float32",
        "reps": 1,
        "port": _free_port(),
        "max_num_seqs": 8,
        "max_num_batched_tokens": 64,
        "saturation_ttft_s": 5.0,
        "backends": {
            # a variant of the native backend: "kind" selects the launcher, the key names results
            "dli-np": {
                "kind": "dli",
                "kv_cache_gb": 0.01,
                "block_size": 4,
                "extra_args": ["--no-prefix-caching"],
            },
            "hf": {"max_batch": 4, "max_wait_ms": 5},
        },
        "models": [
            {
                "name": "tiny",
                "path": str(tiny_dir),
                "backends": ["dli-np", "hf"],
                "experiments": ["closed", "open", "overload"],
            }
        ],
        "experiments": {
            "closed": {
                "workload": {"type": "synthetic", "input": {"fixed": 8}, "output": {"fixed": 4}},
                "mode": "closed",
                "points": [2],
                "n": 6,
            },
            "open": {
                "workload": {"type": "synthetic", "input": {"fixed": 8}, "output": {"fixed": 4}},
                "mode": "open",
                "points": [10, 20],
                "duration_s": 0.5,
                "min_n": 5,
                "max_n": 10,
            },
            "overload": {
                "mode": "overload",
                "base": "open",
                "factors": [2],
                "backends": ["dli-np"],
                "variants": [{"name": "deadline", "headers": {"X-Deadline-Ms": "5000"}}],
            },
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    out = tmp_path / "results"
    campaign.run(plan_path, out, None, None)
    text = campaign.report(out)

    runs = out / "runs" / "tiny"
    assert (runs / "dli-np" / "rep0" / "closed@2.json").exists()
    assert (runs / "hf" / "rep0" / "closed@2.json").exists()
    m = json.loads((runs / "dli-np" / "rep0" / "closed@2.json").read_text())
    assert m["ok"] == 6
    assert m["output_tokens"] == 24
    with gzip.open(runs / "dli-np" / "rep0" / "closed@2.records.jsonl.gz", "rt") as f:
        recs = [json.loads(line) for line in f]
    assert len(recs) == 6
    assert all("chunk_times" not in r and r["chunks"] >= 1 for r in recs)
    hf = json.loads((runs / "hf" / "rep0" / "closed@2.json").read_text())
    assert hf["ok"] == 6  # the baseline served the same requests
    overload = list((runs / "dli-np" / "rep0").glob("overload@*-x2-deadline.json"))
    assert len(overload) == 1  # offered 2x the highest unsaturated open-loop rate
    assert not list((runs / "hf" / "rep0").glob("overload*"))  # dli-only experiment
    assert (out / "env.json").exists()
    assert (out / "plan.json").exists()
    assert "## closed" in text
    assert "| tiny | hf | 2 |" in text
    servers = [
        json.loads(line)
        for line in (out / "campaign_log.jsonl").read_text().splitlines()
        if '"server"' in line
    ]
    assert {s["name"] for s in servers} == {"dli-np", "hf"}
    native = next(s for s in servers if s["name"] == "dli-np")
    assert "--no-auth" in native["cmd"]
    assert "--no-prefix-caching" in native["cmd"]
