"""M2 real-checkpoint spot check: paged KV (with prefix reuse) vs the contiguous path.

The contiguous path is the one M1 proved token-identical to `transformers` on these prompts, so
equality here is transitive evidence for paged storage on real weights. Focused-tier test (~2 min
on CPU), marked ``parity`` because it needs the pinned checkpoint.

Pass 1 runs every prompt on an empty prefix cache. Pass 2 reruns them with a shared prefix
prepended, after a warm-up request that caches that prefix, so most prompt blocks are served from
cache. Both passes must equal the contiguous result.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from dli import envinfo
from dli.generate import greedy_generate, paged_greedy_generate
from dli.kv import BlockManager
from dli.kv.paged_store import PagedKVStore
from dli.model import hub, load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_REVISION = envinfo.current_revision()
REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_PROMPTS = int(os.environ.get("DLI_PAGED_PROMPTS", "20"))
NEW_TOKENS = int(os.environ.get("DLI_PAGED_TOKENS", "32"))
BLOCK_SIZE = 16
LABEL = os.environ.get("DLI_EVIDENCE_LABEL", "CPU-LAPTOP")


@pytest.mark.parity
def test_paged_kv_matches_contiguous_on_real_checkpoint() -> None:
    from transformers import AutoTokenizer

    path = hub.download(REPO_ID)
    tok = AutoTokenizer.from_pretrained(path)
    model = load_model(path)
    prompts = [
        json.loads(line)
        for line in Path(__file__).with_name("prompts.jsonl").read_text().splitlines()
    ]
    texts = [p["text"] for p in prompts if p["kind"] == "completion"][:N_PROMPTS]
    shared = tok(
        "You are a careful assistant. Answer precisely and cite your reasoning step by step.\n\n",
        add_special_tokens=False,
    )["input_ids"]

    bm = BlockManager(num_blocks=512, block_size=BLOCK_SIZE, eviction_policy="lru", watermark=0.0)
    store = PagedKVStore.for_model(model.config, 512, BLOCK_SIZE)
    records: list[dict[str, Any]] = []
    paged_greedy_generate(model, bm, store, "warmup", [*shared, 0], 1)  # caches `shared` blocks
    for i, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        for pass_name, prompt in (("cold", ids), ("shared-prefix", [*shared, *ids])):
            ref = greedy_generate(model, prompt, NEW_TOKENS)
            got = paged_greedy_generate(model, bm, store, f"{pass_name}-{i}", prompt, NEW_TOKENS)
            records.append(
                {
                    "prompt": i,
                    "pass": pass_name,
                    "prompt_tokens": len(prompt),
                    "cached_prompt_tokens": got.num_cached_prompt_tokens,
                    "identical": got.tokens == ref,
                }
            )
    bm.check_invariants()

    stats = bm.stats()
    summary = {
        "model": REPO_ID,
        "revision": hub.PINNED_REVISIONS[REPO_ID],
        "code_revision": RUN_REVISION,
        "evidence_label": LABEL,
        "block_size": BLOCK_SIZE,
        "new_tokens": NEW_TOKENS,
        "runs": len(records),
        "identical_runs": sum(r["identical"] for r in records),
        "shared_prefix_runs_with_cache_hits": sum(
            r["cached_prompt_tokens"] > 0 for r in records if r["pass"] == "shared-prefix"
        ),
        "prefix_hit_rate_tokens": round(stats.hit_rate, 4),
    }
    out = (
        REPO_ROOT
        / "benchmarks"
        / "results"
        / f"{datetime.now(UTC):%Y-%m-%d}-{LABEL}-m2-paged-parity"
    )
    out.mkdir(parents=True, exist_ok=True)
    envinfo.main([str(out / "env.json")], revision=RUN_REVISION)
    (out / "SmolLM2-135M-Instruct.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=2))
    assert summary["identical_runs"] == len(records)
    assert summary["shared_prefix_runs_with_cache_hits"] == len(texts)
