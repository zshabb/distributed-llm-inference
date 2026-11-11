"""Turns a scheduler decision into ONE batched forward pass over paged KV."""

from __future__ import annotations

from typing import Protocol

import torch

from dli.engine.cuda_graph import DecodeGraph
from dli.engine.scheduler import SchedulerOutput
from dli.kv.block_manager import BlockManager
from dli.kv.paged_store import PagedKVStore, PagedSequenceView
from dli.model.kv_cache import BatchedDecodeKV, BatchedKV, Segment
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
        self,
        model: LlamaForCausalLM,
        block_manager: BlockManager,
        store: PagedKVStore,
        *,
        batched_decode: bool = True,
        decode_graph: DecodeGraph | None = None,
    ) -> None:
        self.model = model
        self.bm = block_manager
        self.store = store
        self.device = model.inv_freq.device
        # Steady-state decode steps (every chunk is one new token) use batched attention; prefill
        # and mixed steps, and batched_decode=False, use the per-sequence reference path.
        self.batched_decode = batched_decode
        # Optional CUDA graph of the batched decode forward (M6 proof of concept, off by default).
        self.decode_graph = decode_graph

    @torch.inference_mode()
    def execute(self, sched: SchedulerOutput) -> torch.Tensor:
        # Copy-on-write first: writes in this pass may target blocks that were just un-shared.
        self.store.copy_blocks(sched.copy_ops)
        if (
            self.batched_decode
            and sched.chunks
            and all(ch.num_tokens == 1 and ch.needs_logits for ch in sched.chunks)
        ):
            return self._execute_decode(sched)
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

    def _execute_decode(self, sched: SchedulerOutput) -> torch.Tensor:
        """All chunks are single decode tokens: build batched metadata once, attend in one call.

        If a decode CUDA graph is configured and the step fits its shape, the graph is replayed;
        otherwise (or if it could not be captured) the eager batched path runs.
        """
        bs = self.store.block_size
        ids, positions, slots, tables, lens = [], [], [], [], []
        for ch in sched.chunks:
            rid = ch.request.request_id
            ids.append(ch.request.all_token_ids[ch.start])
            positions.append(ch.start)
            slots.append(self.bm.slot_mapping(rid, ch.start, ch.start + 1)[0])
            tables.append(self.bm.block_table(rid))
            lens.append(ch.start + 1)
        width = max(len(t) for t in tables)
        requests = [ch.request.request_id for ch in sched.chunks]

        def advance() -> None:
            for rid, n in zip(requests, lens, strict=True):
                self.bm.mark_computed(rid, n)

        graph = self.decode_graph
        if graph is not None:
            if graph.stats.status == "not captured":
                graph.capture()  # first eligible step, on the thread that drives the engine
            if graph.fits(len(ids), width):
                logits = graph.run(ids, positions, slots, tables, lens)
                advance()
                return logits
            graph.note_fallback(
                graph.stats.status
                if not graph.ready
                else "batch"
                if len(ids) > graph.batch
                else "width"
            )

        padded = [list(t) + [t[0]] * (width - len(t)) for t in tables]  # padding is masked out
        dev = self.device
        seq_lens = torch.tensor(lens, dtype=torch.long, device=dev)
        dec = BatchedDecodeKV(
            k_pool=self.store.k,
            v_pool=self.store.v,
            slots=torch.tensor(slots, dtype=torch.long, device=dev),
            block_tables=torch.tensor(padded, dtype=torch.long, device=dev),
            key_mask=BatchedDecodeKV.key_mask_for(seq_lens, width * bs),
            on_advance=advance,
        )
        out: torch.Tensor = self.model(
            torch.tensor(ids, dtype=torch.long, device=dev),
            torch.tensor(positions, dtype=torch.long, device=dev),
            dec,
        )
        return out
