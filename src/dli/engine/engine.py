"""LLMEngine: the single-threaded step loop tying scheduler, runner, and sampler together.

One ``step()``:  schedule → forward (one batched pass) → sample → update requests → retire finished.

Each phase is timed separately. ``StepStats`` exposes how much of a step is engine overhead
(scheduling, batch preparation, sampling, bookkeeping) versus model compute, which is the
"measurable scheduling overhead" M3 must report.
"""

from __future__ import annotations

import itertools
import random
import time
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field

import torch

from dli.engine.request import FinishReason, Priority, Request
from dli.engine.runner import ModelRunner, Runner
from dli.engine.scheduler import Scheduler, SchedulerConfig
from dli.kv.block_manager import BlockManager
from dli.kv.paged_store import PagedKVStore
from dli.model.llama import LlamaForCausalLM
from dli.sampling import SamplingParams, make_generator, sample


@dataclass(frozen=True, slots=True)
class RequestOutput:
    request_id: str
    new_token_ids: list[int]
    finished: bool
    finish_reason: FinishReason | None


@dataclass(frozen=True, slots=True)
class StepStats:
    """Timing and shape of one step (seconds)."""

    num_seqs: int
    num_batched_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    num_preempted: int
    schedule_s: float
    forward_s: float
    postprocess_s: float

    @property
    def total_s(self) -> float:
        return self.schedule_s + self.forward_s + self.postprocess_s


@dataclass
class _Counters:
    steps: list[StepStats] = field(default_factory=list)


class LLMEngine:
    """Continuous-batching engine over paged KV.

    Build with :meth:`for_model` for real inference. Tests may inject any :class:`Runner` (e.g. a
    deterministic fake model) together with its block manager.

    Thread-safety: confined to the thread that drives :meth:`step`. The async API (M4) will hand
    requests and cancellations to this loop through a queue.
    """

    def __init__(
        self,
        runner: Runner,
        block_manager: BlockManager,
        config: SchedulerConfig | None = None,
        *,
        eos_token_ids: Collection[int] = (),
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.runner = runner
        self.bm = block_manager
        self.scheduler = Scheduler(config or SchedulerConfig(), block_manager)
        self.eos = frozenset(eos_token_ids)
        self._rng = random.Random(seed)
        self._device = device
        self._arrivals = itertools.count()
        self._step_no = 0
        self.stats = _Counters()

    @classmethod
    def for_model(
        cls,
        model: LlamaForCausalLM,
        *,
        num_blocks: int,
        block_size: int = 16,
        config: SchedulerConfig | None = None,
        eviction_policy: str = "lru",
        eos_token_ids: Collection[int] = (),
        seed: int = 0,
    ) -> LLMEngine:
        device = model.inv_freq.device
        dtype = model.model.embed_tokens.weight.dtype
        bm = BlockManager(num_blocks, block_size, eviction_policy=eviction_policy)
        store = PagedKVStore.for_model(
            model.config, num_blocks, block_size, dtype=dtype, device=device
        )
        return cls(
            ModelRunner(model, bm, store), bm, config, eos_token_ids=eos_token_ids, seed=seed,
            device=device,
        )  # fmt: skip

    # ------------------------------------------------------------------ API

    def add_request(
        self,
        request_id: str,
        prompt_ids: Sequence[int],
        params: SamplingParams | None = None,
        *,
        priority: Priority = Priority.INTERACTIVE,
        tenant: str = "default",
    ) -> Request:
        """Queue a request. A request that can never fit finishes immediately with CAPACITY."""
        if not prompt_ids:
            raise ValueError("prompt must be non-empty")
        params = params or SamplingParams()
        seed = params.seed if params.seed is not None else self._rng.getrandbits(63)
        req = Request(
            request_id=request_id,
            prompt_ids=list(prompt_ids),
            params=params,
            arrival_seq=next(self._arrivals),
            priority=priority,
            tenant=tenant,
            generator=None if params.greedy else make_generator(seed, self._device),
            arrival_step=self._step_no,
        )
        self.scheduler.add(req)
        return req

    def abort(self, request_id: str) -> bool:
        """Cancel a request; returns False if it is unknown or already finished."""
        return self.scheduler.abort(request_id) is not None

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished

    def step(self) -> list[RequestOutput]:
        t0 = time.perf_counter()
        sched = self.scheduler.schedule()
        t1 = time.perf_counter()
        outputs: list[RequestOutput] = []
        if sched.chunks:
            logits = self.runner.execute(sched)
            t2 = time.perf_counter()
            sampling_chunks = [c for c in sched.chunks if c.needs_logits]
            for ch in sched.chunks:
                ch.request.num_computed_tokens += ch.num_tokens
            tokens = sample(
                logits,
                [c.request.params for c in sampling_chunks],
                [c.request.generator for c in sampling_chunks],
            )
            for ch, tok in zip(sampling_chunks, tokens, strict=True):
                outputs.append(self._append_token(ch.request, tok))
        else:
            t2 = t1
        t3 = time.perf_counter()
        prefill = sum(c.num_tokens for c in sched.chunks if c.num_tokens > 1 or not c.needs_logits)
        self.stats.steps.append(
            StepStats(
                num_seqs=len(sched.chunks),
                num_batched_tokens=sched.num_batched_tokens,
                num_prefill_tokens=prefill,
                num_decode_tokens=sched.num_batched_tokens - prefill,
                num_preempted=len(sched.preempted),
                schedule_s=t1 - t0,
                forward_s=t2 - t1,
                postprocess_s=t3 - t2,
            )
        )
        self._step_no += 1
        return outputs

    def generate(
        self,
        prompts: Iterable[Sequence[int]],
        params: SamplingParams | Sequence[SamplingParams] | None = None,
    ) -> list[list[int]]:
        """Run a batch of prompts to completion; returns generated ids in input order."""
        prompts = list(prompts)
        per = (
            params if isinstance(params, Sequence) else [params or SamplingParams()] * len(prompts)
        )
        reqs = [self.add_request(f"gen-{i}", p, per[i]) for i, p in enumerate(prompts)]
        while self.has_unfinished():
            self.step()
        return [r.output_ids for r in reqs]

    # ------------------------------------------------------------------ internals

    def _append_token(self, req: Request, tok: int) -> RequestOutput:
        req.output_ids.append(tok)
        reason: FinishReason | None = None
        if tok in req.params.stop_token_ids or (not req.params.ignore_eos and tok in self.eos):
            reason = FinishReason.STOP
        elif len(req.output_ids) >= req.params.max_tokens:
            reason = FinishReason.LENGTH
        if reason is not None:
            self.scheduler.finish(req, reason)
        return RequestOutput(req.request_id, [tok], reason is not None, reason)
