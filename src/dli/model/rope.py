"""Rotary position embeddings (RoPE), including Llama 3 frequency scaling.

Convention: the "rotate-half" (GPT-NeoX) layout used by Hugging Face Llama/Qwen checkpoints, where
dimension ``i`` is paired with ``i + head_dim/2`` (not adjacent pairs). Using the other layout with
these weights produces fluent-looking garbage, so the parity tests pin this down.

Numerics: frequencies and angles are computed in float32 regardless of model dtype, then the
cos/sin tables are cast to the activation dtype. That matches the reference implementation and
avoids the bf16 angle error that grows linearly with position.
"""

from __future__ import annotations

import math

import torch

from dli.model.config import RopeConfig


def inverse_frequencies(cfg: RopeConfig, head_dim: int) -> torch.Tensor:
    """Per-pair inverse frequencies, shape ``[head_dim // 2]``, float32.

    ``llama3`` keeps high frequencies unchanged, divides low frequencies by ``factor``, and
    interpolates smoothly between them. Wavelengths are compared against the original training
    context length.
    """
    exponents = torch.arange(0, head_dim, 2, dtype=torch.int64).to(torch.float32) / head_dim
    inv_freq = 1.0 / (cfg.theta**exponents)
    if cfg.rope_type == "default":
        return inv_freq
    if cfg.rope_type == "linear":
        # Position interpolation: dividing positions by `factor` == dividing frequencies by it.
        return inv_freq / cfg.factor
    if cfg.rope_type == "llama3":
        old_ctx = cfg.original_max_position_embeddings
        low_freq_wavelen = old_ctx / cfg.low_freq_factor
        high_freq_wavelen = old_ctx / cfg.high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / cfg.factor, inv_freq)
        smooth = (old_ctx / wavelen - cfg.low_freq_factor) / (
            cfg.high_freq_factor - cfg.low_freq_factor
        )
        smoothed = (1 - smooth) * scaled / cfg.factor + smooth * scaled
        is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
        return torch.where(is_medium, smoothed, scaled)
    raise ValueError(f"unsupported rope_type {cfg.rope_type!r}")  # guarded by config parsing


def rope_cos_sin(
    inv_freq: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``cos``/``sin`` tables of shape ``[num_tokens, head_dim]`` for absolute positions.

    Positions are explicit (not ``arange``) so the same code serves prefill, decode, and later
    batched/paged execution where tokens of different sequences sit at different positions.
    """
    angles = positions.to(torch.float32)[:, None] * inv_freq.to(positions.device)[None, :]
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``[heads, num_tokens, head_dim]`` by per-token angles."""
    return x * cos + _rotate_half(x) * sin
