"""Request validation models for the OpenAI-compatible API (contract C1).

Validation is strict on purpose. Unsupported OpenAI parameters (``n > 1``, penalties, ``logprobs``,
``best_of``, ``echo``) are rejected with a 400 instead of being silently ignored: a client asking
for a presence penalty must not receive unpenalized output that looks like it honored the request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_STOP_STRINGS = 4
MAX_STOP_LEN = 128


class _SamplingFields(BaseModel):
    model_config = ConfigDict(extra="allow")  # unknown OpenAI fields are inspected, not trusted

    model: str
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict[str, bool] | None = None
    n: int = Field(default=1, ge=1)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logprobs: bool | int | None = None
    user: str | None = Field(default=None, max_length=256)

    @field_validator("top_k")
    @classmethod
    def _top_k(cls, v: int) -> int:
        if v == 0:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        return v

    @field_validator("stop")
    @classmethod
    def _stop(cls, v: str | list[str] | None) -> list[str] | None:
        if v is None:
            return None
        stops = [v] if isinstance(v, str) else v
        if len(stops) > MAX_STOP_STRINGS:
            raise ValueError(f"at most {MAX_STOP_STRINGS} stop sequences are allowed")
        if any(not s or len(s) > MAX_STOP_LEN for s in stops):
            raise ValueError(f"stop sequences must be 1-{MAX_STOP_LEN} characters")
        return stops

    def unsupported(self) -> str | None:
        """Name of a requested-but-unsupported feature (rejected with 400, never ignored)."""
        if self.n != 1:
            return "n > 1"
        if self.presence_penalty or self.frequency_penalty:
            return "presence_penalty / frequency_penalty"
        if self.logprobs not in (None, False, 0):
            return "logprobs"
        extra = self.model_extra or {}
        if extra.get("best_of", 1) != 1:
            return "best_of"
        for name in ("echo", "logit_bias", "suffix", "tools", "tool_choice", "functions",
                     "response_format"):  # fmt: skip
            if extra.get(name):  # falsy values (False, {}, None) request nothing
                return name
        return None

    @property
    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)

    @property
    def include_usage(self) -> bool:
        return bool((self.stream_options or {}).get("include_usage"))


class CompletionRequest(_SamplingFields):
    prompt: str | list[int]

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, v: str | list[int]) -> str | list[int]:
        if isinstance(v, list) and (not v or any(t < 0 for t in v)):
            raise ValueError("a token-id prompt must be a non-empty list of non-negative ints")
        if isinstance(v, str) and not v:
            raise ValueError("prompt must be non-empty")
        return v


class ChatMessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=1_000_000)
    name: str | None = Field(default=None, max_length=64)


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessageIn] = Field(min_length=1)
