"""CUDA graphs for the batched decode forward pass (M6): a small set of fixed shapes.

After M6 step 1 a steady-state decode step is ~1,200 small kernels whose launch cost (~22 ms on the
A5000) dwarfs their GPU time. A CUDA graph records those launches once and replays them with one
call. A graph has one fixed shape, ``batch x width`` (sequences x block-table blocks), so
:class:`DecodeGraphs` keeps a few *buckets* and replays the smallest one a step fits into, padding
the rest; any step that fits none runs the eager batched path.

What is in a graph: the model forward for :class:`~dli.model.kv_cache.BatchedDecodeKV` (embedding,
RoPE, every layer including the batched KV write, page gather and attention, the LM head) and the
key mask, computed from ``seq_lens`` inside the graph. What is not: metadata construction,
copy-on-write copies, sampling, scheduler and block-manager bookkeeping.

Graph rules this follows:
* **Static buffers.** Inputs live in tensors allocated once per graph; each step copies new values
  into them before ``replay()``. The logits come back in a static output buffer (copied out).
* **Stable addresses.** The KV pools are allocated once (``PagedKVStore``) and only modified in
  place, so the captured kernels keep reading and writing the right memory.
* **Padding is harmless.** Rows beyond the real batch write their K/V to a *scratch block* that the
  block manager never allocates (the store has one block more than the manager) and attend to that
  one written slot, so they can neither corrupt real KV nor produce NaN from an all-masked row.
* **One shared memory pool.** Graphs are captured largest first into one private pool, so the
  smaller ones reuse the largest one's activation memory. That is safe because graphs replay one at
  a time on one stream and every output is copied out before the next replay.
* **Memory is checked first.** Each capture is attempted only if free device memory exceeds the
  measured activation peak of one eager forward at its shape (times a safety factor); otherwise, or
  if capture fails, that graph is skipped and its steps go to a larger bucket or run eagerly.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from dli.kv.paged_store import PagedKVStore
from dli.model.kv_cache import BatchedDecodeKV
from dli.model.llama import LlamaForCausalLM

log = logging.getLogger(__name__)

Shape = tuple[int, int]  # (batch rows, block-table width in blocks)

# Covers the closed-fixed benchmark workload at block size 16: prompts of exactly 256 tokens plus
# 128 output tokens keep every decode context under 384 tokens = 24 blocks, at concurrency 1/16/64.
# Two batch buckets keep padding at most 4x for small batches; one width bucket because padding
# extra key blocks only adds masked gather work, which is cheap next to the launch cost saved.
BENCHMARK_SHAPES: tuple[Shape, ...] = ((16, 24), (64, 24))


def choose_shape(shapes: Sequence[Shape], num_seqs: int, width: int) -> Shape | None:
    """The smallest shape (fewest padded rows, then narrowest) that holds the step, if any."""
    fitting = [s for s in shapes if s[0] >= num_seqs and s[1] >= width]
    return min(fitting, default=None)


@dataclass
class GraphStats:
    status: str = "not captured"  # not captured | captured | disabled: <reason>
    batch: int = 0
    width: int = 0
    capture_s: float = 0.0
    free_before: int = 0  # device bytes free before this capture
    eager_peak: int = 0  # extra bytes one eager forward at this shape needed
    replays: int = 0


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
        pool: Any = None,  # torch.cuda.graph_pool_handle() (a private torch type)
    ) -> None:
        if scratch_block >= store.k.shape[1]:
            raise ValueError("scratch block must exist in the store")
        self.model, self.store = model, store
        self.batch, self.width, self.scratch_block = batch, width, scratch_block
        self.memory_factor, self.memory_margin = memory_factor, memory_margin
        self.pool = pool  # private memory pool shared with other graphs (None: its own)
        self.stats = GraphStats(batch=batch, width=width)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._buffers: dict[str, torch.Tensor] = {}
        self._logits: torch.Tensor | None = None

    @property
    def ready(self) -> bool:
        return self._graph is not None

    def fits(self, num_seqs: int, width: int) -> bool:
        return self.ready and num_seqs <= self.batch and width <= self.width

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
            log.warning("CUDA graph %dx%d %s", b, w, self.stats.status)
            self._buffers = {}
            return False
        t0 = time.perf_counter()
        try:
            side = torch.cuda.Stream(dev)  # type: ignore[no-untyped-call]
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):  # warm-up outside capture (allocator, cuBLAS handles)
                for _ in range(2):
                    self._forward()
            torch.cuda.current_stream(dev).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._logits = self._forward()
            torch.cuda.synchronize(dev)
        except RuntimeError as exc:  # e.g. OOM or an op that cannot be captured
            self._logits = None
            self._buffers = {}
            self.stats.status = f"disabled: capture failed ({type(exc).__name__}: {str(exc)[:120]})"
            log.warning("CUDA graph %dx%d %s", b, w, self.stats.status)
            torch.cuda.empty_cache()
            return False
        self._graph = graph
        self.stats.capture_s = time.perf_counter() - t0
        self.stats.status = "captured"
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


@dataclass
class GraphSetStats:
    status: str = "not captured"  # not captured | captured | disabled: <reason>
    capture_s: float = 0.0  # all captures together
    memory_overhead: int = 0  # reserved bytes added by capture: static buffers + shared pool
    device_free_drop: int = 0  # the same seen by the driver (includes non-PyTorch allocations)
    fallbacks: dict[str, int] = field(default_factory=dict)  # eager steps, by reason


class DecodeGraphs:
    """Decode CUDA graphs for a few bucket shapes, captured together, replayed by best fit."""

    def __init__(
        self,
        model: LlamaForCausalLM,
        store: PagedKVStore,
        *,
        shapes: Sequence[Shape],
        scratch_block: int,
        memory_factor: float = 2.0,
        memory_margin: int = 256 << 20,
    ) -> None:
        uniq = sorted(set(shapes))
        if not uniq or any(b < 1 or w < 1 for b, w in uniq):
            raise ValueError(f"cuda graph shapes must be positive (batch, width) pairs: {shapes}")
        self.store = store
        self.scratch_block = scratch_block
        pool = torch.cuda.graph_pool_handle() if store.k.device.type == "cuda" else None
        self.graphs = [
            DecodeGraph(
                model,
                store,
                batch=b,
                width=w,
                scratch_block=scratch_block,
                memory_factor=memory_factor,
                memory_margin=memory_margin,
                pool=pool,
            )
            for b, w in uniq
        ]
        self.stats = GraphSetStats()

    @property
    def shapes(self) -> list[Shape]:
        return [(g.batch, g.width) for g in self.graphs]

    @property
    def ready(self) -> bool:
        return any(g.ready for g in self.graphs)

    def capture(self) -> bool:
        """Capture every bucket, largest first (the others reuse its pool memory)."""
        dev = self.store.k.device
        if dev.type != "cuda":
            self.stats.status = "disabled: not a CUDA device"
            for g in self.graphs:
                g.stats.status = self.stats.status
            return False
        # torch.cuda.graph() empties the allocator cache on entry; do it first so the reserved
        # delta measures only what the graphs keep (static buffers + the shared private pool).
        torch.cuda.synchronize(dev)
        gc.collect()
        torch.cuda.empty_cache()
        reserved0 = torch.cuda.memory_reserved(dev)
        free0, _ = torch.cuda.mem_get_info(dev)
        t0 = time.perf_counter()
        for g in sorted(self.graphs, key=lambda g: (g.batch * g.width, g.batch), reverse=True):
            g.capture()
        torch.cuda.synchronize(dev)
        self.stats.capture_s = time.perf_counter() - t0
        torch.cuda.empty_cache()  # drop eager scratch freed during capture; graph pools stay
        self.stats.memory_overhead = torch.cuda.memory_reserved(dev) - reserved0
        self.stats.device_free_drop = free0 - torch.cuda.mem_get_info(dev)[0]
        captured = [g for g in self.graphs if g.ready]
        if captured:
            self.stats.status = "captured"
        else:
            reasons = {g.stats.status for g in self.graphs}
            self.stats.status = (
                reasons.pop() if len(reasons) == 1 else "disabled: no shape captured"
            )
        log.info(
            "CUDA graphs: %s of %s captured in %.2fs, +%d MiB reserved",
            [(g.batch, g.width) for g in captured],
            self.shapes,
            self.stats.capture_s,
            self.stats.memory_overhead >> 20,
        )
        return bool(captured)

    def select(self, num_seqs: int, width: int) -> DecodeGraph | None:
        """The smallest captured graph that holds the step, or None (and the reason is counted)."""
        ready = [g for g in self.graphs if g.ready]
        shape = choose_shape([(g.batch, g.width) for g in ready], num_seqs, width)
        if shape is not None:
            return next(g for g in ready if (g.batch, g.width) == shape)
        if not ready:
            reason = self.stats.status
        elif num_seqs > max(g.batch for g in ready):
            reason = "batch"
        else:
            reason = "width"
        self.stats.fallbacks[reason] = self.stats.fallbacks.get(reason, 0) + 1
        return None

    def summary(self) -> dict[str, object]:
        """JSON-ready usage and memory figures (benchmarks, logs)."""
        s = self.stats
        return {
            "status": s.status,
            "capture_s": round(s.capture_s, 3),
            "memory_overhead_mib": round(s.memory_overhead / 2**20, 1),
            "device_free_drop_mib": round(s.device_free_drop / 2**20, 1),
            "replays": sum(g.stats.replays for g in self.graphs),
            "fallbacks": dict(s.fallbacks),
            "graphs": [
                {
                    "batch": g.batch,
                    "width_blocks": g.width,
                    "status": g.stats.status,
                    "free_before_capture_mib": g.stats.free_before >> 20,
                    "eager_peak_mib": round(g.stats.eager_peak / 2**20, 1),
                    "capture_s": round(g.stats.capture_s, 3),
                    "replays": g.stats.replays,
                }
                for g in self.graphs
            ],
        }
