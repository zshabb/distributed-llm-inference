from __future__ import annotations

from collections import Counter

import pytest
import torch

from dli.sampling import SamplingParams, make_generator, sample

LOGITS = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))


def test_greedy_is_argmax_and_needs_no_generator() -> None:
    logits = torch.tensor([[0.1, 2.0, 2.0, -1.0]])
    assert sample(logits, [SamplingParams()], [None]) == [1]  # ties -> lowest id


def test_top_k_one_and_tiny_top_p_reduce_to_greedy() -> None:
    p_k = SamplingParams(temperature=1.0, top_k=1)
    p_p = SamplingParams(temperature=1.0, top_p=1e-6)
    assert sample(LOGITS, [p_k], [None]) == [0]
    assert sample(LOGITS, [p_p], [make_generator(0)]) == [0]


def _freqs(params: SamplingParams, n: int = 10_000) -> dict[int, float]:
    # 10k draws: binomial std <= 0.005, so a 0.02 tolerance is a ~4 sigma bound (non-flaky).
    g = make_generator(123)
    counts = Counter(sample(LOGITS, [params], [g])[0] for _ in range(n))
    return {t: c / n for t, c in counts.items()}


def test_temperature_one_matches_softmax_distribution() -> None:
    f = _freqs(SamplingParams(temperature=1.0))
    for tok, p in enumerate([0.5, 0.3, 0.15, 0.05]):
        assert abs(f.get(tok, 0.0) - p) < 0.02


def test_top_k_and_top_p_restrict_and_renormalize() -> None:
    fk = _freqs(SamplingParams(temperature=1.0, top_k=2))
    assert set(fk) == {0, 1}
    assert abs(fk[0] - 0.5 / 0.8) < 0.02
    # top_p=0.6: tokens 0 and 1 survive (preceding mass 0 and 0.5 < 0.6); token 2 does not.
    fp = _freqs(SamplingParams(temperature=1.0, top_p=0.6))
    assert set(fp) == {0, 1}


def test_seeded_sampling_is_reproducible_and_batch_invariant() -> None:
    torch.manual_seed(0)
    rows = torch.randn(5, 50)
    params = [SamplingParams(temperature=0.9, top_p=0.95, top_k=20, seed=7)] * 5

    def run(batch: torch.Tensor, row_ids: list[int]) -> list[int]:
        gens = {i: make_generator(100 + i) for i in row_ids}
        out = []
        for _ in range(10):  # ten steps, one draw per request per step
            out.append(sample(batch, params[: len(row_ids)], [gens[i] for i in row_ids]))
        return [step[row_ids.index(2)] for step in out]

    alone = run(rows[2:3], [2])
    batched = run(rows, [0, 1, 2, 3, 4])
    assert alone == batched, "a request's draws must not depend on its batch neighbours"
    assert run(rows[2:3], [2]) == alone


def test_mixed_greedy_and_stochastic_rows() -> None:
    logits = torch.tensor([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])
    out = sample(
        logits,
        [SamplingParams(), SamplingParams(temperature=1.0, top_k=1)],
        [None, None],
    )
    assert out == [1, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"top_k": 0},
        {"top_p": 0.0},
        {"top_p": 1.5},
    ],
)
def test_invalid_params_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must"):
        SamplingParams(**kwargs)  # type: ignore[arg-type]


def test_stochastic_row_without_generator_is_an_error() -> None:
    with pytest.raises(ValueError, match="generator"):
        sample(LOGITS, [SamplingParams(temperature=1.0)], [None])
