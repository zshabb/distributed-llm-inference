"""Turns a scheduler decision into ONE batched forward pass over paged KV."""

from __future__ import annotations

from typing import Protocol

import torch

from dli.engine.scheduler import SchedulerOutput
from dli.kv.block_manager import BlockManager
from dli.kv.paged_store import PagedKVStore, PagedSequenceView
from dli.model.kv_cache import BatchedKV, Segment
from dli.model.llama import LlamaForCausalLM


class Runner(Protocol):
    """Executes a schedule. After ``execute`` returns, every scheduled chunk's KV is written and
    marked computed in the block manager.

    Returns logits ``[k, vocab]``, one row per chunk with ``needs_logits``, in schedule order.
    """

    def execute(self, sched: SchedulerOutput) -> torch.Tensor: ...


class ModelRunner:
    """Flattens scheduled chunks into one input, runs the model, and returns sampling rows."""

    def __init__(
        self, model: LlamaForCausalLM, block_manager: BlockManager, store: PagedKVStore
    ) -> None:
        self.model = model
        self.bm = block_manager
        self.store = store
        self.device = model.inv_freq.device

    @torch.inference_mode()
    def execute(self, sched: SchedulerOutput) -> torch.Tensor:
        # Copy-on-write first: writes in this pass may target blocks that were just un-shared.
        self.store.copy_blocks(sched.copy_ops)
        ids: list[int] = []
        positions: list[int] = []
        segments: list[Segment] = []
        logits_rows: list[int] = []
        for ch in sched.chunks:
            offset = len(ids)
            ids.extend(ch.request.all_token_ids[ch.start : ch.start + ch.num_tokens])
            positions.extend(range(ch.start, ch.start + ch.num_tokens))
            view = PagedSequenceView(self.store, self.bm, ch.request.request_id)
            segments.append(Segment(offset, ch.num_tokens, view))
            if ch.needs_logits:
                logits_rows.append(offset + ch.num_tokens - 1)
        logits: torch.Tensor = self.model(
            torch.tensor(ids, dtype=torch.long, device=self.device),
            torch.tensor(positions, dtype=torch.long, device=self.device),
            BatchedKV(tuple(segments)),
            logits_indices=torch.tensor(logits_rows, dtype=torch.long, device=self.device),
        )
        return logits
