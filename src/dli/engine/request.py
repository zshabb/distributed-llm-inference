"""Request lifecycle state for the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, StrEnum

import torch

from dli.sampling import SamplingParams


class RequestStatus(Enum):
    WAITING = "waiting"  # queued, or preempted and waiting to be recomputed
    RUNNING = "running"  # holds KV blocks and is scheduled every step
    FINISHED = "finished"


class FinishReason(StrEnum):
    STOP = "stop"  # emitted EOS or a stop token
    LENGTH = "length"  # reached max_tokens
    ABORTED = "aborted"  # cancelled by the client
    CAPACITY = "capacity"  # can never fit in the KV pool (rejected, never runs)


class Priority(IntEnum):
    """Scheduling class (contract C1 ``X-Request-Priority``). Lower value = served first."""

    INTERACTIVE = 0
    BATCH = 1


@dataclass(eq=False)
class Request:
    """One generation request and its progress.

    ``num_computed_tokens`` counts tokens whose KV is written. It equals the block manager's
    ``num_computed_tokens`` for the request while it holds blocks (the engine asserts this). After a
    preemption it drops to 0 and the request recomputes ``prompt + outputs`` on readmission.
    """

    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    arrival_seq: int
    priority: Priority = Priority.INTERACTIVE
    tenant: str = "default"
    generator: torch.Generator | None = None
    output_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: FinishReason | None = None
    num_computed_tokens: int = 0
    num_preemptions: int = 0
    arrival_step: int = 0

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def is_finished(self) -> bool:
        return self.status is RequestStatus.FINISHED
