"""Greedy decoding for a single sequence: contiguous KV (M1) and paged KV (M2).

Purpose: give the model layer an end-to-end, deterministic workload whose output can be compared
token-for-token with a reference implementation. Batching, sampling, stop strings, and streaming
belong to the scheduler/runner (M3) and API (M4) layers and are deliberately absent here.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import torch

from dli.kv.block_manager import BlockManager, SeqId, SequenceError
from dli.kv.paged_store import PagedKVStore, PagedSequenceView
from dli.model.kv_cache import ContiguousKVCache
from dli.model.llama import LlamaForCausalLM


@torch.inference_mode()
def greedy_generate(
    model: LlamaForCausalLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    eos_token_ids: Collection[int] = (),
    use_cache: bool = True,
) -> list[int]:
    """Generate up to ``max_new_tokens`` tokens by repeatedly taking the argmax.

    Args:
        model: Loaded model (eval mode).
        prompt_ids: Non-empty prompt token ids.
        max_new_tokens: Upper bound on generated tokens.
        eos_token_ids: Stop after emitting any of these (the EOS token is included in the output).
            Empty means "always generate exactly ``max_new_tokens``", which parity tests use.
        use_cache: If False, recompute the whole sequence every step. This is O(n^2) and serves
            only as an independent reference for testing the cached path.

    Returns:
        The generated token ids (prompt excluded).

    Note:
        Ties in argmax resolve to the lowest token id (``torch.argmax`` semantics), the same rule
        the reference uses, so exact-match comparisons are meaningful.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if max_new_tokens <= 0:
        return []
    device = model.inv_freq.device
    cfg = model.config
    eos = set(eos_token_ids)
    tokens = list(prompt_ids)
    out: list[int] = []

    cache = (
        ContiguousKVCache(
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
            max_len=len(prompt_ids) + max_new_tokens,
            dtype=model.model.embed_tokens.weight.dtype,
            device=device,
        )
        if use_cache
        else None
    )

    # First step processes the whole prompt (prefill); later steps feed one token (decode).
    step_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    step_pos = torch.arange(len(tokens), device=device)
    for _ in range(max_new_tokens):
        logits = model(step_ids, step_pos, cache, last_token_only=True)
        nxt = int(torch.argmax(logits[-1]).item())
        out.append(nxt)
        tokens.append(nxt)
        if nxt in eos:
            break
        if cache is not None:
            step_ids = torch.tensor([nxt], dtype=torch.long, device=device)
            step_pos = torch.tensor([len(tokens) - 1], device=device)
        else:
            step_ids = torch.tensor(tokens, dtype=torch.long, device=device)
            step_pos = torch.arange(len(tokens), device=device)
    return out


@dataclass(frozen=True, slots=True)
class PagedGeneration:
    """Result of :func:`paged_greedy_generate`."""

    tokens: list[int]
    num_cached_prompt_tokens: int


class PagedSequence:
    """Step-wise execution of ONE sequence over paged KV (M2).

    Bridges the block manager, the paged store, and the model for a single sequence: reserve slots,
    apply copy-on-write, run the forward pass through a :class:`PagedSequenceView`. It exists to
    validate paged KV end to end before the batched runner (M3) replaces it.
    """

    def __init__(
        self, model: LlamaForCausalLM, manager: BlockManager, store: PagedKVStore, seq_id: SeqId
    ) -> None:
        if not manager.has_sequence(seq_id):
            raise SequenceError(f"unknown sequence {seq_id!r}; allocate or fork it first")
        self.model, self.manager, self.store, self.seq_id = model, manager, store, seq_id

    @torch.inference_mode()
    def prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        """Compute the prompt tokens not served by the prefix cache. Returns last-token logits."""
        start = self.manager.num_computed_tokens(self.seq_id)
        if self.manager.num_tokens(self.seq_id) != len(prompt_ids):
            raise ValueError("prompt length does not match the allocation")
        return self._forward(prompt_ids[start:], start)

    @torch.inference_mode()
    def feed(self, token: int) -> torch.Tensor:
        """Append one token (sampled or forced) and return the logits that follow it."""
        start = self.manager.num_tokens(self.seq_id)
        self.store.copy_blocks(self.manager.append_slots(self.seq_id, [token]))
        return self._forward([token], start)

    def _forward(self, ids: list[int], start: int) -> torch.Tensor:
        device = self.model.inv_freq.device
        view = PagedSequenceView(self.store, self.manager, self.seq_id)
        logits = self.model(
            torch.tensor(ids, dtype=torch.long, device=device),
            torch.arange(start, start + len(ids), device=device),
            view,
            last_token_only=True,
        )
        last: torch.Tensor = logits[-1]
        return last


def paged_greedy_generate(
    model: LlamaForCausalLM,
    manager: BlockManager,
    store: PagedKVStore,
    seq_id: SeqId,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    eos_token_ids: Collection[int] = (),
    namespace: bytes = b"",
    release: bool = True,
) -> PagedGeneration:
    """Greedy decoding over paged KV, reusing any cached prompt prefix.

    Same token semantics as :func:`greedy_generate` (lowest-id argmax tie-break, EOS included).
    With ``release=True`` the sequence is freed at the end: its full blocks stay in the prefix
    cache as cached-idle blocks for later requests.
    """
    alloc = manager.allocate(seq_id, prompt_ids, namespace=namespace)
    seq = PagedSequence(model, manager, store, seq_id)
    eos = set(eos_token_ids)
    out: list[int] = []
    try:
        logits = seq.prefill(prompt_ids)
        for i in range(max_new_tokens):
            nxt = int(torch.argmax(logits).item())
            out.append(nxt)
            if nxt in eos or i == max_new_tokens - 1:
                break
            logits = seq.feed(nxt)
    finally:
        if release:
            manager.free(seq_id)
    return PagedGeneration(out, alloc.num_cached_tokens)
