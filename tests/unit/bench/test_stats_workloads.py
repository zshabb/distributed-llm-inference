"""Benchmark statistics and workload generators: exact, deterministic, honest on edge cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dli.bench.datasets import Conversation, oasst1_conversations
from dli.bench.stats import (
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap_ci,
    paired_proportion_diff_ci,
    summarize,
)
from dli.bench.workloads import TOKEN_LO, LengthDist, multi_turn, shared_prefix, synthetic


def test_summarize_matches_numpy_and_never_fabricates_zero() -> None:
    v = [0.1 * i for i in range(1, 101)]
    s = summarize(v)
    assert s is not None
    assert s.n == 100
    assert s.p50 == pytest.approx(np.percentile(v, 50))
    assert s.p99 == pytest.approx(np.percentile(v, 99))
    assert s.max == pytest.approx(10.0)
    assert summarize([]) is None


def test_bootstrap_is_deterministic_and_covers_the_median() -> None:
    rng = np.random.default_rng(1)
    v = list(rng.normal(5.0, 1.0, 400))
    lo, hi = bootstrap_ci(v)
    assert (lo, hi) == bootstrap_ci(v)  # fixed seed: same data, same report
    assert lo < float(np.median(v)) < hi
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_ci([1.0])


def test_paired_bootstrap_detects_a_shift_and_not_noise() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 300)
    mean, lo, _ = paired_bootstrap_ci(a, a + 0.5)
    assert mean == pytest.approx(0.5)
    assert lo > 0.4
    _, lo0, hi0 = paired_bootstrap_ci(a, a + rng.normal(0, 0.01, 300))
    assert lo0 < 0 < hi0


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 2**10)
    assert mcnemar_exact(5, 5) == 1.0
    # 2 vs 8 discordant: 2 * P(X <= 2 | n=10, p=1/2) = 2 * 56/1024
    assert mcnemar_exact(2, 8) == pytest.approx(2 * 56 / 1024)


def test_paired_proportion_ci_sign_and_width() -> None:
    lo, hi = paired_proportion_diff_ci(only_a=10, only_b=30, n=1000)
    assert lo > 0  # B better by 2 points on 1000 items with 40 discordant pairs
    assert hi < 0.04
    lo, hi = paired_proportion_diff_ci(0, 0, 1000)
    assert lo < 0 < hi  # no discordance: interval still has width (1/n), never a point


def test_synthetic_workload_exact_lengths_and_safe_tokens() -> None:
    wl = synthetic("w", 50, LengthDist.fixed(64), LengthDist.fixed(16), vocab=5000, seed=3)
    assert len(wl.requests) == 50
    for r in wl.requests:
        assert r.prompt is not None
        assert len(r.prompt) == 64
        assert r.max_tokens == 16
        assert min(r.prompt) >= TOKEN_LO
        assert max(r.prompt) < 5000
    again = synthetic("w", 50, LengthDist.fixed(64), LengthDist.fixed(16), vocab=5000, seed=3)
    assert again.requests == wl.requests  # every backend gets the identical list


def test_lognormal_lengths_are_clipped_and_centred() -> None:
    d = LengthDist.lognormal(256, 0.8, 16, 1024)
    x = d.sample(np.random.default_rng(0), 5000)
    assert min(x) >= 16
    assert max(x) <= 1024
    assert 230 < float(np.median(x)) < 280


def test_shared_prefix_workload_shares_prefixes_with_zipf_skew() -> None:
    wl = shared_prefix(
        "p",
        400,
        num_prefixes=8,
        prefix_len=128,
        suffix_len=LengthDist.fixed(32),
        output_len=LengthDist.fixed(8),
        zipf_s=1.1,
        vocab=32000,
        seed=0,
    )
    heads = [r.prompt[:128] for r in wl.requests if r.prompt is not None]
    counts = sorted((heads.count(h) for h in set(heads)), reverse=True)
    assert len(counts) <= 8
    assert counts[0] > 3 * counts[-1]  # popular prefixes dominate
    tails = {r.prompt[128:] for r in wl.requests if r.prompt is not None}
    assert len(tails) == 400  # suffixes are unique


def _row(
    mid: str, parent: str | None, role: str, text: str, rank: int | None = None
) -> dict[str, object]:
    return {
        "message_id": mid,
        "parent_id": parent,
        "role": role,
        "text": text,
        "rank": rank,
        "lang": "en",
        "deleted": False,
    }


def test_oasst1_follows_the_best_ranked_branch() -> None:
    rows = [
        _row("a", None, "prompter", "hi"),
        _row("b1", "a", "assistant", "x", rank=1),
        _row("b0", "a", "assistant", "y", rank=0),  # best ranked
        _row("c", "b0", "prompter", "follow-up"),
        _row("d", "b1", "prompter", "other branch"),
        _row("z", None, "prompter", "single turn only"),
    ]
    convs = oasst1_conversations(rows, min_turns=2)
    assert convs == [Conversation("a", ("hi", "follow-up"))]


def test_multi_turn_sessions_are_deterministic() -> None:
    convs = [Conversation(str(i), tuple(f"u{i}-{t}" for t in range(5))) for i in range(10)]
    wl = multi_turn(
        "m",
        convs,
        num_sessions=4,
        system_prompts=["S1", "S2"],
        max_turns=3,
        output_tokens=32,
        seed=7,
    )
    assert len(wl.sessions) == 4
    assert all(len(s.user_turns) == 3 and s.system in ("S1", "S2") for s in wl.sessions)
    assert (
        wl.sessions
        == multi_turn(
            "m",
            convs,
            num_sessions=4,
            system_prompts=["S1", "S2"],
            max_turns=3,
            output_tokens=32,
            seed=7,
        ).sessions
    )
    with pytest.raises(ValueError, match="conversations available"):
        multi_turn("m", convs, num_sessions=11, system_prompts=["S"], max_turns=2, output_tokens=8)
    assert not math.isnan(wl.params["turns_total"])
