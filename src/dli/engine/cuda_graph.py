"""CUDA graph for the batched decode forward pass (M6 proof of concept: one fixed shape).

After M6 step 1 a steady-state decode step is ~1,200 small kernels whose launch cost (~22 ms on the
A5000) dwarfs their GPU time (~8 ms). A CUDA graph records those launches once and replays them
with one call. This module captures exactly one shape, ``batch x width`` (sequences x block-table
blocks); any step that does not fit runs the eager batched path.

What is in the graph: the model forward for :class:`~dli.model.kv_cache.BatchedDecodeKV`
(embedding, RoPE, every layer including the batched KV write, page gather and attention, the LM
head) and the key mask, computed from ``seq_lens`` inside the graph. What is not: metadata
construction, copy-on-write copies, sampling, scheduler and block-manager bookkeeping.

Graph rules this follows:
* **Static buffers.** Inputs live in tensors allocated once; each step copies new values into them
  before ``replay()``. The logits come back in a static output buffer (copied out by the caller).
* **Stable addresses.** The KV pools are allocated once (``PagedKVStore``) and only modified in
  place, so the captured kernels keep reading and writing the right memory.
* **Padding is harmless.** Rows beyond the real batch write their K/V to a *scratch block* that the
  block manager never allocates (the store has one block more than the manager) and attend to that
  one written slot, so they can neither corrupt real KV nor produce NaN from an all-masked row.
* **Memory is checked first.** Capture is attempted only if free device memory exceeds the measured
  activation peak of one eager forward at this shape (times a safety factor); otherwise, or if
  capture fails, the graph is disabled and every step runs eagerly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from dli.kv.paged_store import PagedKVStore
from dli.model.kv_cache import BatchedDecodeKV
from dli.model.llama import LlamaForCausalLM

log = logging.getLogger(__name__)


@dataclass
class GraphStats:
    status: str = "not captured"  # not captured | captured | disabled: <reason>
    batch: int = 0
    width: int = 0
    capture_s: float = 0.0
    free_before: int = 0  # device bytes free before capture
    eager_peak: int = 0  # extra bytes one eager forward at this shape needed
    graph_pool: int = 0  # reserved-memory growth caused by capture (graph private pool)
    replays: int = 0
    fallbacks: dict[str, int] = field(default_factory=dict)


class DecodeGraph:
    """One captured batched-decode forward of shape ``batch x width``."""

    def __init__(
        self,
        model: LlamaForCausalLM,
        store: PagedKVStore,
        *,
        batch: int,
        width: int,
        scratch_block: int,
        memory_factor: float = 2.0,
        memory_margin: int = 256 << 20,
    ) -> None:
        if scratch_block >= store.k.shape[1]:
            raise ValueError("scratch block must exist in the store")
        self.model, self.store = model, store
        self.batch, self.width, self.scratch_block = batch, width, scratch_block
        self.memory_factor, self.memory_margin = memory_factor, memory_margin
        self.stats = GraphStats(batch=batch, width=width)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._buffers: dict[str, torch.Tensor] = {}
        self._logits: torch.Tensor | None = None

    # ---------------------------------------------------------------- shape gate

    def fits(self, num_seqs: int, width: int) -> bool:
        return self.ready and num_seqs <= self.batch and width <= self.width

    @property
    def ready(self) -> bool:
        return self._graph is not None

    def note_fallback(self, reason: str) -> None:
        self.stats.fallbacks[reason] = self.stats.fallbacks.get(reason, 0) + 1

    # ---------------------------------------------------------------- capture

    def _forward(self) -> torch.Tensor:
        b = self._buffers
        dec = BatchedDecodeKV(
            k_pool=self.store.k,
            v_pool=self.store.v,
            slots=b["slots"],
            block_tables=b["block_tables"],
            key_mask=BatchedDecodeKV.key_mask_for(
                b["seq_lens"], self.width * self.store.block_size
            ),
            on_advance=lambda: None,  # bookkeeping happens outside the graph
        )
        logits: torch.Tensor = self.model(b["input_ids"], b["positions"], dec)
        return logits

    def _reset_padding(self) -> None:
        """Every row points at the scratch block (the state in which capture runs)."""
        b, bs = self._buffers, self.store.block_size
        b["input_ids"].zero_()
        b["positions"].zero_()
        b["slots"].fill_(self.scratch_block * bs)
        b["block_tables"].fill_(self.scratch_block)
        b["seq_lens"].fill_(1)

    def capture(self) -> bool:
        """Capture the graph; on insufficient memory or any capture error, disable it."""
        dev = self.store.k.device
        if dev.type != "cuda":
            self.stats.status = "disabled: not a CUDA device"
            return False
        b, w = self.batch, self.width
        self._buffers = {
            "input_ids": torch.zeros(b, dtype=torch.long, device=dev),
            "positions": torch.zeros(b, dtype=torch.long, device=dev),
            "slots": torch.zeros(b, dtype=torch.long, device=dev),
            "block_tables": torch.zeros(b, w, dtype=torch.long, device=dev),
            "seq_lens": torch.ones(b, dtype=torch.long, device=dev),
        }
        self._reset_padding()
        torch.cuda.synchronize(dev)
        # Measure what one eager forward at this shape needs, then require headroom for it.
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
        self._forward()
        torch.cuda.synchronize(dev)
        self.stats.eager_peak = torch.cuda.max_memory_allocated(dev) - base
        free, _ = torch.cuda.mem_get_info(dev)
        self.stats.free_before = free
        needed = int(self.stats.eager_peak * self.memory_factor) + self.memory_margin
        if free < needed:
            self.stats.status = f"disabled: {free >> 20} MiB free < {needed >> 20} MiB needed"
            log.warning("CUDA graph %s", self.stats.status)
            return False
        reserved_before = torch.cuda.memory_reserved(dev)
        t0 = time.perf_counter()
        try:
            side = torch.cuda.Stream(dev)  # type: ignore[no-untyped-call]
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):  # warm-up outside capture (allocator, cuBLAS handles)
                for _ in range(2):
                    self._forward()
            torch.cuda.current_stream(dev).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._logits = self._forward()
            torch.cuda.synchronize(dev)
        except RuntimeError as exc:  # e.g. OOM or an op that cannot be captured
            self._logits = None
            self.stats.status = f"disabled: capture failed ({type(exc).__name__}: {str(exc)[:120]})"
            log.warning("CUDA graph %s", self.stats.status)
            torch.cuda.empty_cache()
            return False
        self._graph = graph
        self.stats.capture_s = time.perf_counter() - t0
        self.stats.graph_pool = torch.cuda.memory_reserved(dev) - reserved_before
        self.stats.status = "captured"
        log.info(
            "CUDA graph captured: batch %d x width %d in %.2fs, pool +%d MiB",
            self.batch,
            self.width,
            self.stats.capture_s,
            self.stats.graph_pool >> 20,
        )
        return True

    # ---------------------------------------------------------------- replay

    def run(
        self,
        input_ids: Sequence[int],
        positions: Sequence[int],
        slots: Sequence[int],
        tables: Sequence[Sequence[int]],
        seq_lens: Sequence[int],
    ) -> torch.Tensor:
        """Copy one step's metadata into the static buffers, replay; return ``[n, vocab]``."""
        if self._graph is None or self._logits is None:
            raise RuntimeError("graph not captured")
        n = len(input_ids)
        if n > self.batch or max(len(t) for t in tables) > self.width:
            raise ValueError("step does not fit the captured shape")
        bs, pad = self.store.block_size, self.batch - n
        scratch_slot = self.scratch_block * bs
        # Build the full padded step on the host, then one copy per buffer (no stale rows).
        host = {
            "input_ids": list(input_ids) + [0] * pad,
            "positions": list(positions) + [0] * pad,
            "slots": list(slots) + [scratch_slot] * pad,
            "block_tables": [list(t) + [self.scratch_block] * (self.width - len(t)) for t in tables]
            + [[self.scratch_block] * self.width] * pad,
            "seq_lens": list(seq_lens) + [1] * pad,
        }
        for name, values in host.items():
            self._buffers[name].copy_(torch.tensor(values, dtype=torch.long), non_blocking=False)
        self._graph.replay()
        self.stats.replays += 1
        return self._logits[:n].clone()  # the buffer is overwritten by the next replay
