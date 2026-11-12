"""Turns a scheduler decision into ONE batched forward pass over paged KV."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import torch

from dli.engine.cuda_graph import DecodeGraphs
from dli.engine.scheduler import ScheduledChunk, SchedulerOutput
from dli.kv.block_manager import BlockManager
from dli.kv.paged_store import PagedKVStore, PagedSequenceView
from dli.model.kv_cache import BatchedDecodeKV, BatchedKV, MixedKV, Segment
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
        decode_graphs: DecodeGraphs | None = None,
        decode_attention: str = "sdpa",
    ) -> None:
        self.model = model
        self.bm = block_manager
        self.store = store
        self.device = model.inv_freq.device
        # Steady-state decode steps (every chunk is one new token) use batched attention; prefill
        # and mixed steps, and batched_decode=False, use the per-sequence reference path.
        self.batched_decode = batched_decode
        # Optional CUDA graphs of the batched decode forward, one per bucket shape (off by default).
        self.decode_graphs = decode_graphs
        # Batched decode attention kernel: "sdpa" (reference, any device) or "triton" (CUDA).
        self.decode_attention = decode_attention

    @torch.inference_mode()
    def execute(self, sched: SchedulerOutput) -> torch.Tensor:
        # Copy-on-write first: writes in this pass may target blocks that were just un-shared.
        self.store.copy_blocks(sched.copy_ops)
        if self.batched_decode and sched.chunks:
            if all(ch.num_tokens == 1 and ch.needs_logits for ch in sched.chunks):
                return self._execute_decode(sched)
            if any(ch.num_tokens == 1 for ch in sched.chunks):
                return self._execute_mixed(sched)
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

    def _decode_rows(
        self, chunks: Sequence[ScheduledChunk]
    ) -> tuple[list[int], list[int], list[int], list[list[int]], list[int], Callable[[], None]]:
        """Metadata of single-token chunks: ids, positions, slots, block tables, lengths, and the
        callback that marks the tokens computed once every layer has written them."""
        ids, positions, slots, tables, lens = [], [], [], [], []
        for ch in chunks:
            rid = ch.request.request_id
            ids.append(ch.request.all_token_ids[ch.start])
            positions.append(ch.start)
            slots.append(self.bm.slot_mapping(rid, ch.start, ch.start + 1)[0])
            tables.append(list(self.bm.block_table(rid)))
            lens.append(ch.start + 1)
        requests = [ch.request.request_id for ch in chunks]

        def advance() -> None:
            for rid, n in zip(requests, lens, strict=True):
                self.bm.mark_computed(rid, n)

        return ids, positions, slots, tables, lens, advance

    def _decode_kv(
        self,
        slots: list[int],
        tables: list[list[int]],
        lens: list[int],
        advance: Callable[[], None],
    ) -> BatchedDecodeKV:
        """Eager batched-decode attention metadata (tables padded to the widest; padding masked)."""
        width = max(len(t) for t in tables)
        padded = [t + [t[0]] * (width - len(t)) for t in tables]
        dev = self.device
        seq_lens = torch.tensor(lens, dtype=torch.long, device=dev)
        triton = self.decode_attention == "triton"
        return BatchedDecodeKV(
            k_pool=self.store.k,
            v_pool=self.store.v,
            slots=torch.tensor(slots, dtype=torch.long, device=dev),
            block_tables=torch.tensor(padded, dtype=torch.long, device=dev),
            key_mask=None
            if triton
            else BatchedDecodeKV.key_mask_for(seq_lens, width * self.store.block_size),
            on_advance=advance,
            seq_lens=seq_lens,
            kernel=self.decode_attention,
        )

    def _execute_decode(self, sched: SchedulerOutput) -> torch.Tensor:
        """All chunks are single decode tokens: build batched metadata once, attend in one call.

        If decode CUDA graphs are configured, the smallest captured bucket the step fits into is
        replayed; otherwise (no bucket fits, or none could be captured) the eager batched path runs.
        """
        ids, positions, slots, tables, lens, advance = self._decode_rows(sched.chunks)
        graphs = self.decode_graphs
        if graphs is not None:
            if graphs.stats.status == "not captured":
                graphs.capture()  # first decode step, on the thread that drives the engine
            graph = graphs.select(len(ids), max(len(t) for t in tables))  # counts fallbacks
            if graph is not None:
                logits = graph.run(ids, positions, slots, tables, lens)
                advance()
                return logits
        dev = self.device
        out: torch.Tensor = self.model(
            torch.tensor(ids, dtype=torch.long, device=dev),
            torch.tensor(positions, dtype=torch.long, device=dev),
            self._decode_kv(slots, tables, lens, advance),
        )
        return out

    def _execute_mixed(self, sched: SchedulerOutput) -> torch.Tensor:
        """Single-token rows batched (as in decode) plus multi-token prefill chunks per sequence.

        The flattened batch puts every single-token chunk first, then each prefill chunk; logits
        come back in schedule order. Runs eagerly: decode graphs cover pure decode steps only.
        """
        singles = [ch for ch in sched.chunks if ch.num_tokens == 1]
        ids, positions, slots, tables, lens, advance = self._decode_rows(singles)
        row = {id(ch): i for i, ch in enumerate(singles)}
        segments: list[Segment] = []
        for ch in sched.chunks:
            if ch.num_tokens == 1:
                continue
            offset = len(ids)
            ids.extend(ch.request.all_token_ids[ch.start : ch.start + ch.num_tokens])
            positions.extend(range(ch.start, ch.start + ch.num_tokens))
            view = PagedSequenceView(self.store, self.bm, ch.request.request_id)
            segments.append(Segment(offset, ch.num_tokens, view))
            row[id(ch)] = offset + ch.num_tokens - 1
        dev = self.device
        logits: torch.Tensor = self.model(
            torch.tensor(ids, dtype=torch.long, device=dev),
            torch.tensor(positions, dtype=torch.long, device=dev),
            MixedKV(self._decode_kv(slots, tables, lens, advance), tuple(segments)),
            logits_indices=torch.tensor(
                [row[id(ch)] for ch in sched.chunks if ch.needs_logits],
                dtype=torch.long,
                device=dev,
            ),
        )
        return logits
