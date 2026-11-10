"""M3 milestone check: the continuous-batching engine vs the unbatched path on real checkpoints.

All 100 M1 prompts are submitted at once to an engine configured to force chunked prefill (small
token budget) and preemption (KV pool smaller than total demand). Every request's greedy tokens are
compared with the unbatched contiguous-cache path, which M1 tied to `transformers`.

Acceptance (fixed before running):
* fp32: identical tokens, EXCEPT that a divergence is accepted when the reference's top-2 logit gap
  at that step is < 1e-3. Batched matmuls sum in a different order than unbatched ones (~1e-6
  relative), which can legitimately flip a near-tie. Any divergence at a larger gap is a bug.
* bf16 (GPU): M1's rule: >= 98% identical on the first 64 tokens, divergences only within bf16
  rounding (gap <= 1.0 at magnitude 32).

Run: ``uv run --frozen pytest -m parity -s tests/parity/test_engine_parity.py``
GPU: ``DLI_PARITY_DEVICE=cuda DLI_PARITY_DTYPE=bfloat16 DLI_EVIDENCE_LABEL=GPU-A16x1-vultr ...``
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch

from dli import envinfo
from dli.engine import LLMEngine, SchedulerConfig
from dli.model import LlamaForCausalLM, hub, load_model
from dli.model.kv_cache import ContiguousKVCache
from dli.sampling import SamplingParams

from .test_hf_greedy_parity import _prompt_ids

MODELS = ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
NEW_TOKENS = int(os.environ.get("DLI_PARITY_TOKENS", "128"))
NUM_PROMPTS = int(os.environ.get("DLI_PARITY_PROMPTS", "100"))
DEVICE = os.environ.get("DLI_PARITY_DEVICE", "cpu")
DTYPE = getattr(torch, os.environ.get("DLI_PARITY_DTYPE", "float32"))
LABEL = os.environ.get("DLI_EVIDENCE_LABEL", "CPU-LAPTOP")
FP32_NEAR_TIE = 1e-3
BF16_GAP = 1.0
BLOCK = 16
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_REVISION = envinfo.current_revision()


@torch.inference_mode()
def _unbatched_with_gaps(
    model: LlamaForCausalLM, ids: list[int], steps: int
) -> tuple[list[int], list[float]]:
    """The M1 contiguous path, also recording the top-2 logit gap at every step."""
    cfg = model.config
    cache = ContiguousKVCache(
        cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, len(ids) + steps,
        model.model.embed_tokens.weight.dtype, model.inv_freq.device,
    )  # fmt: skip
    dev = model.inv_freq.device
    logits = model(torch.tensor(ids, device=dev), torch.arange(len(ids), device=dev), cache,
                   last_token_only=True)[-1].float()  # fmt: skip
    toks, gaps = [], []
    for _ in range(steps):
        top2 = torch.topk(logits, 2).values
        gaps.append(float(top2[0] - top2[1]))
        nxt = int(torch.argmax(logits))
        toks.append(nxt)
        pos = torch.tensor([cache.length], device=dev)
        logits = model(torch.tensor([nxt], device=dev), pos, cache, last_token_only=True)[
            -1
        ].float()
    return toks, gaps


@pytest.mark.parity
@pytest.mark.parametrize("repo_id", MODELS)
def test_engine_matches_unbatched(repo_id: str) -> None:
    from transformers import AutoTokenizer

    path = hub.download(repo_id)
    tok = AutoTokenizer.from_pretrained(path)
    model = load_model(path, dtype=DTYPE, device=DEVICE)
    prompts = [
        _prompt_ids(tok, json.loads(line))
        for line in Path(__file__).with_name("prompts.jsonl").read_text().splitlines()
    ][:NUM_PROMPTS]

    t0 = time.perf_counter()
    refs = [_unbatched_with_gaps(model, p, NEW_TOKENS) for p in prompts]
    t_ref = time.perf_counter() - t0

    # Pool = 25% of total demand (but >= the largest single request, so nothing is rejected):
    # with up to 48 concurrent sequences this forces preemption. A 256-token budget forces
    # chunked prefill.
    need = [-(-(len(p) + NEW_TOKENS) // BLOCK) for p in prompts]
    num_blocks = max(int(sum(need) * 0.25), max(need))
    eng = LLMEngine.for_model(
        model, num_blocks=num_blocks, block_size=BLOCK,
        config=SchedulerConfig(max_num_seqs=48, max_num_batched_tokens=256),
    )  # fmt: skip
    params = SamplingParams(max_tokens=NEW_TOKENS, ignore_eos=True)
    reqs = [eng.add_request(f"p{i}", p, params) for i, p in enumerate(prompts)]
    t1 = time.perf_counter()
    while eng.has_unfinished():
        eng.step()
    t_eng = time.perf_counter() - t1

    records: list[dict[str, Any]] = []
    for i, (r, (ref, gaps)) in enumerate(zip(reqs, refs, strict=True)):
        div = next(
            (j for j, (a, b) in enumerate(zip(ref, r.output_ids, strict=True)) if a != b), None
        )
        records.append(
            {
                "id": i,
                "prompt_tokens": len(prompts[i]),
                "identical": div is None,
                "first_divergence": div,
                "ref_top2_gap_at_divergence": None if div is None else gaps[div],
                "preemptions": r.num_preemptions,
            }
        )
    steps = eng.stats.steps
    summary = {
        "model": repo_id,
        "revision": hub.PINNED_REVISIONS[repo_id],
        "code_revision": RUN_REVISION,
        "evidence_label": LABEL,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "prompts": len(records),
        "new_tokens": NEW_TOKENS,
        "identical_prompts": sum(r["identical"] for r in records),
        "num_blocks": num_blocks,
        "block_size": BLOCK,
        "total_preemptions": sum(r["preemptions"] for r in records),
        "engine_steps": len(steps),
        "max_batched_tokens_seen": max(s.num_batched_tokens for s in steps),
        "wall_seconds_unbatched_reference": round(t_ref, 1),
        "wall_seconds_engine": round(t_eng, 1),
        "note": "Wall times are incidental (reference also records logit gaps); not a benchmark.",
    }
    out = (
        REPO_ROOT
        / "benchmarks"
        / "results"
        / (f"{datetime.now(UTC):%Y-%m-%d}-{LABEL}-m3-engine-parity-{str(DTYPE).split('.')[-1]}")
    )
    out.mkdir(parents=True, exist_ok=True)
    envinfo.main([str(out / "env.json")], revision=RUN_REVISION)
    (out / f"{repo_id.split('/')[-1]}.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=2))

    assert summary["total_preemptions"] > 0, "configuration did not exercise preemption"
    divergent = [r for r in records if not r["identical"]]
    if DTYPE is torch.float32:  # dtypes are singletons
        for r in divergent:
            assert r["ref_top2_gap_at_divergence"] < FP32_NEAR_TIE, r
    else:
        prefix_ok = sum(
            r["first_divergence"] is None or r["first_divergence"] >= 64 for r in records
        )
        assert prefix_ok >= 0.98 * len(records)
        for r in divergent:
            assert r["ref_top2_gap_at_divergence"] <= BF16_GAP, r
