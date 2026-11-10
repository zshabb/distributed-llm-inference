"""Continuous-batching scheduler with chunked prefill.

Every engine step, the scheduler builds one batch under a **token budget**
(``max_num_batched_tokens``) and a **sequence cap** (``max_num_seqs``):

1. **Running requests first**, in admission order. Each needs either its next decode token or the
   next chunk of an unfinished prompt. Serving running requests first keeps inter-token latency
   bounded: a burst of new prompts cannot stall sequences that are already generating.
2. **Then admit waiting requests**, in queue order, while the budget, the sequence cap, and KV
   memory allow. A prompt longer than the remaining budget is **chunked**: its first part runs this
   step and the rest continues in later steps as a running request. This bounds the cost of any
   single step, so one long prompt cannot freeze every decode in the batch for a whole prefill.

KV memory is reserved through the block manager: a prompt's blocks at admission (reusing cached
prefix blocks), and one slot per new token just before it is computed. When a running request
needs a block and none is available, the newest running request not yet in this step's batch is
**preempted by recompute** (see :meth:`Scheduler._preempt`).

Thread confinement: like the block manager, the scheduler belongs to the single engine loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from dli.engine.request import FinishReason, Request, RequestStatus
from dli.kv.block_manager import BlockManager, CopyOp, NoFreeBlocksError


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Batch-shaping limits.

    Attributes:
        max_num_seqs: Maximum requests holding KV and being scheduled concurrently.
        max_num_batched_tokens: Token budget per step (prefill chunks + decode tokens).
        namespace: Prefix-cache namespace (model/weights identity; see ADR-0003).
    """

    max_num_seqs: int = 64
    max_num_batched_tokens: int = 512
    namespace: bytes = b""

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1:
            raise ValueError("max_num_seqs and max_num_batched_tokens must be >= 1")


@dataclass(frozen=True, slots=True)
class ScheduledChunk:
    """Compute ``num_tokens`` tokens of ``request`` starting at absolute position ``start``.

    ``needs_logits`` is true when the chunk reaches the end of the request's known tokens, so a
    new token must be sampled from the chunk's last position.
    """

    request: Request
    start: int
    num_tokens: int
    needs_logits: bool


@dataclass(slots=True)
class SchedulerOutput:
    chunks: list[ScheduledChunk] = field(default_factory=list)
    copy_ops: list[CopyOp] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)

    @property
    def num_batched_tokens(self) -> int:
        return sum(c.num_tokens for c in self.chunks)


class Scheduler:
    """Decides which tokens of which requests run in the next forward pass."""

    def __init__(self, config: SchedulerConfig, block_manager: BlockManager) -> None:
        self.config = config
        self.bm = block_manager
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self._by_id: dict[str, Request] = {}

    # ------------------------------------------------------------------ lifecycle

    def add(self, req: Request) -> bool:
        """Queue a request. Returns False (and finishes it with CAPACITY) if it can never fit.

        A request whose prompt plus ``max_tokens`` needs more blocks than the whole pool would be
        preempted forever even when running alone. Rejecting it up front is what guarantees every
        admitted request can eventually complete (no livelock).
        """
        if req.request_id in self._by_id:
            raise ValueError(f"duplicate request id {req.request_id!r}")
        worst_case = len(req.prompt_ids) + req.params.max_tokens
        if -(-worst_case // self.bm.block_size) > self.bm.num_blocks:
            req.status = RequestStatus.FINISHED
            req.finish_reason = FinishReason.CAPACITY
            return False
        self._by_id[req.request_id] = req
        self.waiting.append(req)
        return True

    def finish(self, req: Request, reason: FinishReason) -> None:
        """Release a request's KV and remove it from scheduling."""
        if req.status is RequestStatus.RUNNING:
            self.bm.free(req.request_id)
            self.running.remove(req)
        elif req.status is RequestStatus.WAITING:
            self.waiting.remove(req)
        req.status = RequestStatus.FINISHED
        req.finish_reason = reason
        self._by_id.pop(req.request_id, None)

    def abort(self, request_id: str) -> Request | None:
        """Cancel a queued or running request; its blocks are freed immediately.

        Safe between steps (the engine loop is single-threaded). Returns None for unknown or
        already finished ids, which makes repeated cancellation idempotent.
        """
        req = self._by_id.get(request_id)
        if req is None:
            return None
        self.finish(req, FinishReason.ABORTED)
        return req

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------ scheduling

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens

        scheduled: set[str] = set()

        for req in list(self.running):
            if budget == 0:
                break
            if req.status is not RequestStatus.RUNNING:
                continue  # preempted earlier in this loop as a victim
            n = min(req.num_tokens - req.num_computed_tokens, budget)
            if n <= 0:
                continue
            ops = self._reserve(req)
            while ops is None:
                # Out of KV memory: preempt the newest running request not yet in this batch.
                victim = self._pick_victim(exclude=scheduled | {req.request_id})
                if victim is None:
                    break
                self._preempt(victim)
                out.preempted.append(victim)
                ops = self._reserve(req)
            if ops is None:
                # Nobody else can yield: this request is itself the newest unscheduled one.
                # (It is never the oldest running request: capacity rejection guarantees the
                # oldest fits alone, so the loop above always finds a victim for it.)
                self._preempt(req)
                out.preempted.append(req)
                continue
            out.copy_ops.extend(ops)
            out.chunks.append(self._chunk(req, n))
            scheduled.add(req.request_id)
            budget -= n

        # Never admit in a step that had to preempt: readmitting what was just evicted (or
        # something equally large) would only thrash. Freed memory first serves running requests.
        while (
            not out.preempted
            and self.waiting
            and budget > 0
            and len(self.running) < self.config.max_num_seqs
        ):
            req = self.waiting[0]
            tokens = req.all_token_ids
            if not self.bm.can_allocate(tokens, self.config.namespace):
                break  # head of line waits for memory; admission order is preserved
            alloc = self.bm.allocate(req.request_id, tokens, namespace=self.config.namespace)
            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            req.num_computed_tokens = alloc.num_cached_tokens
            self.running.append(req)
            n = min(len(tokens) - alloc.num_cached_tokens, budget)
            out.chunks.append(self._chunk(req, n))
            budget -= n
        return out

    def _pick_victim(self, exclude: set[str]) -> Request | None:
        """Newest-admitted running request outside ``exclude`` (LIFO preemption).

        LIFO protects the requests that have waited and progressed longest; combined with
        capacity rejection, the oldest running request can always complete (liveness).
        """
        for cand in reversed(self.running):
            if cand.request_id not in exclude:
                return cand
        return None

    def _preempt(self, req: Request) -> None:
        """Preempt by recompute: free KV now, keep outputs, recompute prompt+outputs later.

        Full blocks the request computed stay in the prefix cache as cached-idle, so unless memory
        pressure evicts them, resuming recomputes only the uncached tail. The request goes to the
        FRONT of the waiting queue so it resumes before newer arrivals.
        """
        self.bm.free(req.request_id)
        self.running.remove(req)
        req.status = RequestStatus.WAITING
        req.num_computed_tokens = 0
        req.num_preemptions += 1
        self.waiting.appendleft(req)

    def _chunk(self, req: Request, n: int) -> ScheduledChunk:
        start = req.num_computed_tokens
        return ScheduledChunk(req, start, n, needs_logits=start + n == req.num_tokens)

    def _reserve(self, req: Request) -> list[CopyOp] | None:
        """Reserve KV slots for tokens the block manager has not seen yet (newly sampled ones)."""
        missing = req.num_tokens - self.bm.num_tokens(req.request_id)
        if missing <= 0:
            return []
        try:
            return self.bm.append_slots(req.request_id, req.all_token_ids[-missing:])
        except NoFreeBlocksError:
            return None
