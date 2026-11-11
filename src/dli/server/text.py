"""Text handling for the API: tokenizer, chat templates, incremental detokenization, stop strings.

Three correctness hazards live here:

1. **Chat templates are untrusted code.** A model repository ships its chat template as a Jinja
   program. It is rendered in Jinja's ``ImmutableSandboxedEnvironment``, which blocks attribute
   access to Python internals and mutation of passed-in objects, so a malicious template cannot
   reach the server process.
2. **Streaming must not emit broken characters.** Byte-level BPE can split one UTF-8 character
   across tokens, so decoding the newest token alone yields U+FFFD. :class:`IncrementalDetokenizer`
   re-decodes a short window and emits text only once it no longer ends in an incomplete character.
3. **Stop strings can span tokens and must never leak.** :class:`StopStringMatcher` scans the
   accumulated text and, while streaming, holds back any suffix that could still become a stop
   string, so a client never sees a partial stop sequence that the final response would remove.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


class ChatTemplateError(ValueError):
    """The model has no chat template, or the conversation is rejected by it."""


def _raise_exception(message: str) -> None:
    # HF chat templates call raise_exception() to reject invalid conversations (e.g. bad role
    # order).
    raise ChatTemplateError(message)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


class TextCodec:
    """Tokenizer + chat template for one model directory (``tokenizer.json`` and configs)."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        cfg = _read_json(model_dir / "tokenizer_config.json")
        gen = _read_json(model_dir / "generation_config.json")
        self.chat_template: str | None = cfg.get("chat_template")
        self.bos_token = _token_str(cfg.get("bos_token"))
        self.eos_token = _token_str(cfg.get("eos_token"))
        eos_ids: set[int] = set()
        gen_eos = gen.get("eos_token_id")
        if isinstance(gen_eos, int):
            eos_ids.add(gen_eos)
        elif isinstance(gen_eos, list):
            eos_ids.update(int(t) for t in gen_eos)
        if self.eos_token is not None:
            tid = self.tokenizer.token_to_id(self.eos_token)
            if tid is not None:
                eos_ids.add(tid)
        self.eos_token_ids: frozenset[int] = frozenset(eos_ids)
        self._env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self._env.globals["raise_exception"] = _raise_exception

    def encode(self, text: str) -> list[int]:
        """Token ids for raw text (no special tokens added: the API takes text as given)."""
        ids: list[int] = self.tokenizer.encode(text, add_special_tokens=False).ids
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        text: str = self.tokenizer.decode(list(ids), skip_special_tokens=True)
        return text

    def render_chat(self, messages: Sequence[ChatMessage]) -> list[int]:
        """Render a conversation with the model's template (sandboxed) and tokenize it."""
        if not self.chat_template:
            raise ChatTemplateError("this model has no chat template; use /v1/completions")
        try:
            text = self._env.from_string(self.chat_template).render(
                messages=[{"role": m.role, "content": m.content} for m in messages],
                add_generation_prompt=True,
                bos_token=self.bos_token or "",
                eos_token=self.eos_token or "",
            )
        except ChatTemplateError:
            raise
        except TemplateError as exc:  # includes SecurityError from the sandbox
            raise ChatTemplateError(f"chat template failed: {exc}") from exc
        # The template already places special tokens such as BOS; do not add them again.
        ids: list[int] = self.tokenizer.encode(text, add_special_tokens=False).ids
        return ids


class IncrementalDetokenizer:
    """Turns a growing token list into text deltas without splitting UTF-8 characters.

    Keeps two offsets into the token list: ``prefix`` (start of a small context window that makes
    subword joins decode correctly) and ``read`` (tokens already emitted). Each call decodes
    ``ids[prefix:]`` and emits the part beyond what ``ids[prefix:read]`` produced, unless the result
    ends in U+FFFD (an incomplete multi-byte character), in which case it waits for more tokens.
    """

    def __init__(self, codec: TextCodec, prompt_ids: Sequence[int]) -> None:
        self._codec = codec
        self._ids: list[int] = list(prompt_ids)
        # Start the window a few tokens back so the first generated token joins its predecessor.
        self._prefix = max(0, len(self._ids) - 5)
        self._read = len(self._ids)

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        prev = self._codec.decode(self._ids[self._prefix : self._read])
        full = self._codec.decode(self._ids[self._prefix :])
        if full.endswith("�") or len(full) <= len(prev):
            return ""
        delta = full[len(prev) :]
        self._prefix = self._read
        self._read = len(self._ids)
        return delta


class StopStringMatcher:
    """Detects stop strings in streamed text and decides what is safe to emit.

    Feed every text delta to :meth:`feed`, which returns ``(safe_text, stopped)``. ``safe_text``
    excludes the stop string and everything after it, and also holds back a trailing fragment that
    is a prefix of some stop string. Call :meth:`flush` at the end of generation to release
    held-back text that never became a stop string.
    """

    def __init__(self, stops: Sequence[str]) -> None:
        self._stops = [s for s in stops if s]
        self._buffer = ""  # text not yet emitted
        self.stopped = False

    def feed(self, delta: str) -> tuple[str, bool]:
        if self.stopped:
            return "", True
        self._buffer += delta
        cut = min((i for i in (self._buffer.find(s) for s in self._stops) if i >= 0), default=-1)
        if cut >= 0:
            self.stopped = True
            out, self._buffer = self._buffer[:cut], ""
            return out, True
        hold = self._longest_partial_suffix()
        out = self._buffer[: len(self._buffer) - hold]
        self._buffer = self._buffer[len(self._buffer) - hold :]
        return out, False

    def flush(self) -> str:
        out, self._buffer = ("" if self.stopped else self._buffer), ""
        return out

    def _longest_partial_suffix(self) -> int:
        best = 0
        for s in self._stops:
            for k in range(min(len(s) - 1, len(self._buffer)), 0, -1):
                if self._buffer.endswith(s[:k]):
                    best = max(best, k)
                    break
        return best


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _token_str(tok: object) -> str | None:
    if isinstance(tok, str):
        return tok
    if isinstance(tok, dict) and isinstance(tok.get("content"), str):
        return str(tok["content"])  # AddedToken serialized form
    return None
