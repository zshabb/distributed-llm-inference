"""Sampling parameters and a batched sampler with per-request random generators.

Batch invariance: each request owns a ``torch.Generator``, and exactly one draw is taken from it per
sampled token. The random numbers a request sees therefore depend only on its own seed and
position, never on which other requests share the batch. Given identical logits, a seeded request
produces the same tokens whether it runs alone, batched, or after preemption and recompute
(the generator is not rewound on recompute because no new draws happen during recompute).

Masking (top-k, top-p) is vectorized across rows with one sort. Only the final draw loops over rows,
because ``torch.multinomial`` accepts a single generator per call.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Per-request generation settings (OpenAI-compatible subset, see contract C1).

    Attributes:
        max_tokens: Maximum generated tokens.
        temperature: 0 selects greedy decoding (argmax, lowest id on ties).
        top_k: Keep the k most likely tokens; -1 disables.
        top_p: Nucleus sampling: smallest set of tokens with cumulative probability >= top_p.
        seed: Seed for this request's generator; None draws one from the engine's RNG.
        stop_token_ids: Generation stops after emitting any of these (token included in output).
        ignore_eos: Do not stop at the model's EOS tokens (benchmarks use this for fixed lengths).
    """

    max_tokens: int = 16
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0.0 <= self.temperature <= 2.0 or math.isnan(self.temperature):
            raise ValueError("temperature must be in [0, 2]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0 or self.top_k == 1


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def sample(
    logits: torch.Tensor,
    params: Sequence[SamplingParams],
    generators: Sequence[torch.Generator | None],
) -> list[int]:
    """Choose one token per row.

    Args:
        logits: ``[rows, vocab]`` next-token logits (any float dtype; computed in float32).
        params: Sampling settings per row.
        generators: Generator per row (required for non-greedy rows).

    Returns:
        One token id per row.
    """
    rows = logits.shape[0]
    if not len(params) == len(generators) == rows:
        raise ValueError("logits, params and generators must have one entry per row")
    logits = logits.float()
    out = torch.argmax(logits, dim=-1)  # greedy rows keep this
    stochastic = [i for i, p in enumerate(params) if not p.greedy]
    if not stochastic:
        return out.tolist()

    idx = torch.tensor(stochastic, device=logits.device)
    sub = logits.index_select(0, idx)
    temps = torch.tensor([params[i].temperature for i in stochastic], device=logits.device)
    sub = sub / temps[:, None]

    # One descending sort serves both top-k (by rank) and top-p (by cumulative mass).
    sorted_logits, order = torch.sort(sub, dim=-1, descending=True)
    vocab = sub.shape[-1]
    ks = torch.tensor(
        [params[i].top_k if params[i].top_k > 0 else vocab for i in stochastic],
        device=logits.device,
    )
    ranks = torch.arange(vocab, device=logits.device)[None, :]
    sorted_logits = sorted_logits.masked_fill(ranks >= ks[:, None], float("-inf"))
    ps = torch.tensor([params[i].top_p for i in stochastic], device=logits.device)
    probs = torch.softmax(sorted_logits, dim=-1)
    # Drop a token if the mass of strictly more likely tokens already reaches top_p. The most
    # likely token always survives (its preceding mass is 0).
    preceding = torch.cumsum(probs, dim=-1) - probs
    sorted_logits = sorted_logits.masked_fill(preceding >= ps[:, None], float("-inf"))
    probs = torch.softmax(sorted_logits, dim=-1)

    for row, i in enumerate(stochastic):
        gen = generators[i]
        if gen is None:
            raise ValueError(f"row {i} samples stochastically but has no generator")
        choice = torch.multinomial(probs[row], 1, generator=gen)
        out[i] = order[row, choice]
    return out.tolist()
