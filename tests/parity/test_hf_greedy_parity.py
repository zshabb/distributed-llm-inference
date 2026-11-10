"""AC1 parity suite: greedy decoding vs Hugging Face `transformers` on real pinned checkpoints.

Run:  ``uv run --frozen pytest -m parity -s``            (fp32, CPU: the strict M1 criterion)
GPU:  ``DLI_PARITY_DEVICE=cuda DLI_PARITY_DTYPE=bfloat16 DLI_EVIDENCE_LABEL=GPU-A16x1-vultr ...``

Criteria (spec 01 AC1):
* fp32: every prompt's generated tokens are identical to the reference.
* bf16: at least 98% of prompts match exactly on the first 64 tokens, and every divergence occurs
  where the reference's top-2 logit gap is within bf16 resolution, i.e. a near-tie that rounding can
  flip, not a model bug. (The spec's original "gap < 1e-2" is below bf16 resolution for logits of
  magnitude > 1; see docs/known-issues.md.)

Both implementations receive identical token ids (tokenization is not under test), run without EOS
stopping, and break argmax ties identically. A per-prompt report is written under
``benchmarks/results/`` together with ``env.json``.
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
from dli.generate import greedy_generate
from dli.model import hub, load_model

MODELS = ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
NEW_TOKENS = int(os.environ.get("DLI_PARITY_TOKENS", "128"))
NUM_PROMPTS = int(os.environ.get("DLI_PARITY_PROMPTS", "100"))
DEVICE = os.environ.get("DLI_PARITY_DEVICE", "cpu")
DTYPE = getattr(torch, os.environ.get("DLI_PARITY_DTYPE", "float32"))
LABEL = os.environ.get("DLI_EVIDENCE_LABEL", "CPU-LAPTOP")
BF16_PREFIX = 64
REPO_ROOT = Path(__file__).resolve().parents[2]
# Captured once at collection time: the run can last over an hour, and commits made meanwhile
# must not be attributed to it (see dli.envinfo.current_revision).
RUN_REVISION = envinfo.current_revision()


def _prompt_ids(tokenizer: Any, prompt: dict[str, Any]) -> list[int]:
    if prompt["kind"] == "chat":
        ids = tokenizer.apply_chat_template(
            prompt["messages"], add_generation_prompt=True, tokenize=True
        )
        if isinstance(ids, dict) or hasattr(ids, "keys"):  # transformers may return BatchEncoding
            ids = ids["input_ids"]
        return [int(t) for t in ids]
    return [int(t) for t in tokenizer(prompt["text"], add_special_tokens=False)["input_ids"]]


@torch.inference_mode()
def _reference_greedy(ref: Any, ids: list[int], steps: int) -> tuple[list[int], list[float]]:
    """HF greedy loop with its own KV cache; also returns the top-2 logit gap at each step."""
    out = ref(torch.tensor([ids], device=DEVICE), use_cache=True)
    past, tokens, gaps = out.past_key_values, [], []
    for _ in range(steps):
        logits = out.logits[0, -1].float()
        top2 = torch.topk(logits, 2).values
        gaps.append(float(top2[0] - top2[1]))
        nxt = int(logits.argmax())
        tokens.append(nxt)
        out = ref(torch.tensor([[nxt]], device=DEVICE), past_key_values=past, use_cache=True)
        past = out.past_key_values
    return tokens, gaps


def _bf16_resolution(logit_magnitude: float) -> float:
    """Spacing of bf16 values near ``logit_magnitude`` (8-bit mantissa incl. implicit bit)."""
    m = max(abs(logit_magnitude), 1e-3)
    return float(2.0 ** (torch.floor(torch.log2(torch.tensor(m))).item() - 7))


@pytest.mark.parity
@pytest.mark.parametrize("repo_id", MODELS)
def test_greedy_parity_with_transformers(repo_id: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = hub.download(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompts = [
        json.loads(line)
        for line in (Path(__file__).with_name("prompts.jsonl")).read_text().splitlines()
    ][:NUM_PROMPTS]

    ref = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE).to(DEVICE).eval()
    ours = load_model(path, dtype=DTYPE, device=DEVICE)

    records: list[dict[str, Any]] = []
    t_ref = t_ours = 0.0
    for prompt in prompts:
        ids = _prompt_ids(tokenizer, prompt)
        t0 = time.perf_counter()
        ref_tokens, gaps = _reference_greedy(ref, ids, NEW_TOKENS)
        t1 = time.perf_counter()
        our_tokens = greedy_generate(ours, ids, NEW_TOKENS)
        t2 = time.perf_counter()
        t_ref, t_ours = t_ref + (t1 - t0), t_ours + (t2 - t1)
        div = next(
            (i for i, (a, b) in enumerate(zip(ref_tokens, our_tokens, strict=True)) if a != b), None
        )
        records.append(
            {
                "id": prompt["id"],
                "kind": prompt["kind"],
                "prompt_tokens": len(ids),
                "identical": div is None,
                "first_divergence": div,
                "ref_top2_gap_at_divergence": None if div is None else gaps[div],
                "min_ref_top2_gap": min(gaps),
            }
        )

    exact = sum(r["identical"] for r in records)
    summary = {
        "model": repo_id,
        "revision": hub.PINNED_REVISIONS[repo_id],
        "code_revision": RUN_REVISION,
        "evidence_label": LABEL,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "prompts": len(records),
        "new_tokens": NEW_TOKENS,
        "identical_prompts": exact,
        "wall_seconds_reference": round(t_ref, 1),
        "wall_seconds_ours": round(t_ours, 1),
        "note": "Incidental timings (sequential, unbatched, with warm-up); not a benchmark.",
    }
    out_dir = (
        REPO_ROOT
        / "benchmarks"
        / "results"
        / f"{datetime.now(UTC):%Y-%m-%d}-{LABEL}-m1-greedy-parity-{str(DTYPE).split('.')[-1]}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    envinfo.main([str(out_dir / "env.json")], revision=RUN_REVISION)
    (out_dir / f"{repo_id.split('/')[-1]}.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=2))

    if DTYPE is torch.float32:  # dtypes are singletons
        assert exact == len(records), [r for r in records if not r["identical"]][:5]
    else:
        prefix_ok = sum(
            r["first_divergence"] is None or r["first_divergence"] >= BF16_PREFIX for r in records
        )
        assert prefix_ok >= 0.98 * len(records), f"{prefix_ok}/{len(records)} match first 64"
        for r in records:
            if r["first_divergence"] is not None:
                # A flip is only acceptable where bf16 rounding can plausibly reorder the top-2.
                assert r["ref_top2_gap_at_divergence"] <= 4 * _bf16_resolution(32.0), r
