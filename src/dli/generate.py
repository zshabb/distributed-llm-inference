"""Greedy decoding for a single sequence (M1).

Purpose: give the model layer an end-to-end, deterministic workload whose output can be compared
token-for-token with a reference implementation. Batching, sampling, stop strings, and streaming
belong to the scheduler/runner (M3) and API (M4) layers and are deliberately absent here.
"""

from __future__ import annotations

from collections.abc import Collection

import torch

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
