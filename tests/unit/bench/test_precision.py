"""The precision study's machinery, checked where the right answer is known exactly.

With the *same* fp32 model in every configuration, the engine-based paths (teacher-forced decode,
perplexity, ARC scoring) must reproduce the unbatched reference: any indexing off-by-one between a
logits row and the token it predicts would show up as a disagreement or an NLL mismatch here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dli.bench.precision import (
    M3_ENGINE,
    StepLog,
    arc,
    bf16_ulp,
    compare,
    engine_forced,
    fidelity,
    perplexity,
    reference_greedy,
)
from dli.model import LlamaForCausalLM, load_model
from tests.helpers.tiny import build_tiny_model


@pytest.fixture(scope="module")
def fp32(tmp_path_factory: pytest.TempPathFactory) -> LlamaForCausalLM:
    d: Path = tmp_path_factory.mktemp("prec-tiny")
    build_tiny_model("llama", d)
    return load_model(d, dtype=torch.float32)


PROMPTS = [[5, 9, 13, 2, 40], [7, 7, 7], list(range(3, 30)), [50, 60]]


def test_bf16_ulp_and_compare_basics() -> None:
    assert bf16_ulp(torch.tensor([32.0, 1.0, -3.0])).tolist() == [0.25, 2**-7, 2**-6]
    x = torch.randn(4, 50)
    m = compare(x, x.clone())
    assert bool(m["agree"].all())
    assert float(m["kl"].max()) == 0.0
    assert float(m["max_abs_err"].max()) == 0.0


def test_engine_teacher_forcing_appends_exactly_the_forced_tokens(fp32: LlamaForCausalLM) -> None:
    # long continuations: running sequences outgrow the 25% pool, which forces preemption
    forced = [[11 + i % 50 for i in range(30)], [20] * 40, [30 + i for i in range(25)], [40] * 35]
    seen: list[tuple[int, int]] = []
    info = engine_forced(
        fp32, PROMPTS, forced, lambda keys, lg: seen.extend(keys), M3_ENGINE, block_size=4
    )
    # engine_forced itself asserts that the outputs equal `forced`
    assert info["preemptions"] > 0  # 25% pool: the M3 configuration really preempts
    # Each (sequence, step) row exactly once, even with preemption: recompute replays known tokens
    # as a prefill and does not sample them again.
    assert sorted(seen) == sorted((s, k) for s, f in enumerate(forced) for k in range(len(f)))


def test_fp32_engine_equals_reference_under_teacher_forcing(fp32: LlamaForCausalLM) -> None:
    out = fidelity({"fp32": fp32, "bf16": fp32}, PROMPTS, steps=40)
    for cfg in ("B_bf16_unbatched", "C_bf16_engine", "D_fp32_engine"):
        s = out[cfg]
        assert s["steps"] == len(PROMPTS) * 40
        assert s["top1_agreement"] == 1.0, cfg
        assert s["kl_max"] < 1e-6, cfg
    assert out["engine_config"]["D"]["preemptions"] > 0  # measured through preemption
    assert out["D_fp32_engine"]["duplicate_rows"] == 0


def test_perplexity_paths_agree_in_fp32(fp32: LlamaForCausalLM) -> None:
    g = torch.Generator().manual_seed(0)
    windows = [[int(t) for t in torch.randint(3, 97, (32,), generator=g)] for _ in range(5)]
    out = perplexity({"fp32": fp32, "bf16": fp32}, windows, context=16)
    assert out["scored_tokens"] == 5 * 16
    for a, c in zip(out["per_window_nll"]["A"], out["per_window_nll"]["C"], strict=True):
        assert c == pytest.approx(a, rel=1e-4)  # engine decode rows score the same targets
    assert out["ratio_C_over_A"]["ratio"] == pytest.approx(1.0, abs=1e-4)


def test_arc_scoring_paths_agree_in_fp32(fp32: LlamaForCausalLM) -> None:
    items = [
        ([5, 6, 7], [[8], [9, 10], [11, 12, 13]], 1),
        ([20, 21], [[22, 23], [24]], 0),
        ([30], [[31], [32], [33], [34]], 3),
    ]
    out = arc({"fp32": fp32, "bf16": fp32}, items)
    assert (
        out["per_item_correct"]["A"] == out["per_item_correct"]["B"] == out["per_item_correct"]["C"]
    )
    assert out["C_vs_A"]["only_x"] == out["C_vs_A"]["only_y"] == 0


def test_reference_greedy_logits_match_tokens(fp32: LlamaForCausalLM) -> None:
    toks, logits = reference_greedy(fp32, PROMPTS[0], 6)
    assert toks == logits.argmax(-1).tolist()


def test_steplog_counts_duplicates_and_conditions_on_margin() -> None:
    log = StepLog()
    m = compare(
        torch.tensor([[2.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        torch.tensor([[2.0, 1.0, 0.0], [0.9, 1.0, 0.0]]),
    )
    log.add(0, [0, 1], m)
    log.add(0, [1], {k: v[1:] for k, v in m.items()})
    s = log.summary()
    assert s["steps"] == 2
    assert s["duplicate_rows"] == 1
    assert s["disagreements"] == 1
    assert s["disagreements_beyond_1ulp"] == 0  # the flip happened at an exact tie
    assert s["top1_agreement_beyond_1ulp"] == 1.0
