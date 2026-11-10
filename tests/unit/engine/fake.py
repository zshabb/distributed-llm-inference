"""Deterministic fake model for scheduler/engine tests.

next_token(history) depends on EVERY token of the history, so any scheduling bug that drops,
duplicates, or reorders tokens, or loses a request's progress across preemption, changes the
output. The reference is the same rule applied to one request alone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Sequence

import torch

from dli.engine.scheduler import SchedulerOutput
from dli.kv.block_manager import BlockManager

VOCAB = 50


def next_token(history: Sequence[int]) -> int:
    digest = hashlib.blake2b(bytes(str(list(history)), "ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % VOCAB


def reference(prompt: Sequence[int], max_tokens: int, stop: Collection[int] = ()) -> list[int]:
    toks, out = list(prompt), []
    for _ in range(max_tokens):
        t = next_token(toks)
        out.append(t)
        toks.append(t)
        if t in stop:
            break
    return out


class FakeRunner:
    """Implements the Runner contract: marks chunks computed and returns one-hot logits."""

    def __init__(self, bm: BlockManager) -> None:
        self.bm = bm
        self.calls: list[SchedulerOutput] = []

    def execute(self, sched: SchedulerOutput) -> torch.Tensor:
        self.calls.append(sched)
        rows = []
        for ch in sched.chunks:
            rid = ch.request.request_id
            # The contract: KV for [start, start+n) is now written.
            assert self.bm.num_computed_tokens(rid) == ch.start, "chunk does not continue the KV"
            self.bm.mark_computed(rid, ch.start + ch.num_tokens)
            if ch.needs_logits:
                row = torch.full((VOCAB,), -1e9)
                row[next_token(ch.request.all_token_ids[: ch.start + ch.num_tokens])] = 0.0
                rows.append(row)
        return torch.stack(rows) if rows else torch.empty(0, VOCAB)
