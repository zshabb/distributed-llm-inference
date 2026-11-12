"""Llama-family decoder-only transformer (Llama 2/3, SmolLM2, Qwen2).

Written from the architecture definition rather than wrapping ``transformers`` so that every
operation the scheduler and KV-cache layers depend on is ours to change (paged attention in M2/M6,
tensor-parallel linears in M7). Module and parameter names mirror the Hugging Face checkpoint
layout, so a checkpoint's state dict maps 1:1 and the loader can be strict about names and shapes.

Execution model (M1): one sequence, flattened tokens ``[num_tokens]`` with explicit ``positions``.
The flattened-token interface is deliberate: continuous batching (M3) packs tokens of many
sequences into one such vector, so the per-layer math will not change, only the attention metadata.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from dli.model.config import ModelConfig
from dli.model.kv_cache import BatchedDecodeKV, BatchedKV, KVCacheView, MixedKV, Segment
from dli.model.paged_attention import paged_decode_attention
from dli.model.rope import apply_rope, inverse_frequencies, rope_cos_sin


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, computed in float32 and cast back.

    The float32 upcast is not optional for bf16 models: the mean of squares over thousands of
    channels loses too much precision in bf16. Matching the reference's cast placement is also
    required for bit-level parity in fp32 tests.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dtype)


class Attention(nn.Module):
    """Grouped-query self-attention with RoPE and an optional KV cache."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=cfg.o_bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segments: Sequence[Segment] | BatchedDecodeKV | MixedKV,
    ) -> torch.Tensor:
        n = x.shape[0]
        # Projections and RoPE run once over the whole flattened batch (one large matmul instead of
        # one per sequence: this is where batching pays off). Only attention is per sequence.
        # [n, heads*dim] -> [heads, n, dim]: heads lead so SDPA treats them as the batch dimension.
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(n, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(n, self.num_kv_heads, self.head_dim).transpose(0, 1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if isinstance(segments, BatchedDecodeKV):
            out = self._attend_decode_batch(q, k, v, segments)
        elif isinstance(segments, MixedKV):
            # Single-token rows first (one batched call), then the prefill chunks per sequence.
            d = segments.decode.batch_size
            out = torch.cat(
                [
                    self._attend_decode_batch(q[:, :d], k[:, :d], v[:, :d], segments.decode),
                    *self._attend_segments(q, k, v, segments.segments),
                ],
                dim=1,
            )
        elif len(segments) == 1:
            out = self._attend(q, k, v, segments[0].cache)
        else:
            out = torch.cat(self._attend_segments(q, k, v, segments), dim=1)
        merged = out.transpose(0, 1).reshape(n, self.num_heads * self.head_dim)
        projected: torch.Tensor = self.o_proj(merged)
        return projected

    def _attend_segments(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, segments: Sequence[Segment]
    ) -> list[torch.Tensor]:
        """Per-segment loop: the correct reference for variable-length attention."""
        return [
            self._attend(
                q[:, s.offset : s.offset + s.length],
                k[:, s.offset : s.offset + s.length],
                v[:, s.offset : s.offset + s.length],
                s.cache,
            )
            for s in segments
        ]

    def _attend_decode_batch(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dec: BatchedDecodeKV
    ) -> torch.Tensor:
        """One new token for each of B sequences, attended in one call against paged KV.

        ``q`` is ``[heads, B, d]``; ``k``/``v`` are ``[kv_heads, B, d]``. Returns ``[heads, B, d]``.
        Launches a fixed number of kernels per layer whatever B is; the per-sequence path issued
        about eight per sequence (M5 profile: write, gather and GQA expansion of K and V, and
        attention).
        """
        bsz = dec.batch_size
        k_pool, v_pool = dec.k_pool[self.layer_idx], dec.v_pool[self.layer_idx]
        kv_heads, dim = k_pool.shape[2], k_pool.shape[3]
        # Write all new K/V rows at once (keys post-RoPE, as in the reference path).
        k_pool.view(-1, kv_heads, dim).index_copy_(0, dec.slots, k.transpose(0, 1))
        v_pool.view(-1, kv_heads, dim).index_copy_(0, dec.slots, v.transpose(0, 1))
        if dec.kernel == "triton":
            # Read each sequence's live keys in place, a whole query group per KV head (M6).
            assert dec.seq_lens is not None
            out_t = paged_decode_attention(
                q.transpose(0, 1), k_pool, v_pool, dec.block_tables, dec.seq_lens
            )
            return out_t.transpose(0, 1)
        # Gather every sequence's blocks, padded: [B, max_blocks * block, kv_heads, d].
        keys = k_pool[dec.block_tables].view(bsz, -1, kv_heads, dim).transpose(1, 2)
        values = v_pool[dec.block_tables].view(bsz, -1, kv_heads, dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(2),  # [B, heads, 1, d]
            keys,
            values,
            attn_mask=dec.key_mask,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return out.squeeze(2).transpose(0, 1)

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_cache: KVCacheView | None
    ) -> torch.Tensor:
        """Causal attention for ONE sequence's new tokens against its cached prefix."""
        n = q.shape[1]
        if kv_cache is not None:
            # Keys are cached AFTER rotation: positions are absolute, so cached entries
            # never change.
            k, v = kv_cache.write(self.layer_idx, kv_cache.length, k, v)
        past = k.shape[1] - n

        # GQA: query head h reads KV head h // fan_out (contiguous groups, as in the checkpoints).
        # Materializing the expansion costs memory bandwidth; the paged kernel (M6) indexes instead.
        fan_out = self.num_heads // self.num_kv_heads
        if fan_out > 1:
            k = k.repeat_interleave(fan_out, dim=0)
            v = v.repeat_interleave(fan_out, dim=0)

        if n == 1:
            # Single new token attends to everything before it: no mask needed.
            out = F.scaled_dot_product_attention(q, k, v)
        elif past == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Chunk appended to a non-empty cache: query i (absolute position past+i)
            # may see keys at positions <= past+i.
            # SDPA's is_causal aligns the mask to the top-left corner, which is wrong here.
            q_pos = torch.arange(past, past + n, device=q.device)[:, None]
            k_pos = torch.arange(past + n, device=q.device)[None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=k_pos <= q_pos)
        return out


class MLP(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class DecoderLayer(nn.Module):
    """Pre-norm transformer block: ``x + attn(norm(x))`` then ``x + mlp(norm(x))``."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segments: Sequence[Segment] | BatchedDecodeKV,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, segments)
        out: torch.Tensor = x + self.mlp(self.post_attention_layernorm(x))
        return out


class Backbone(nn.Module):
    """Embedding + decoder stack + final norm (``model.*`` in checkpoints)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class LlamaForCausalLM(nn.Module):
    """Decoder-only language model producing next-token logits.

    Construct on the ``meta`` device and load weights with :func:`dli.model.loader.load_model`;
    direct construction allocates and randomly initializes every parameter.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.model = Backbone(cfg)
        # Tied checkpoints have no lm_head tensor; logits reuse the embedding matrix.
        self.lm_head = (
            None if cfg.tie_word_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, False)
        )
        self.register_buffer(
            "inv_freq", inverse_frequencies(cfg.rope, cfg.head_dim), persistent=False
        )

    def reset_rope(self, device: torch.device) -> None:
        """Recompute the non-persistent RoPE buffer on ``device`` (needed after meta init)."""
        self.inv_freq = inverse_frequencies(self.config.rope, self.config.head_dim).to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCacheView | BatchedKV | BatchedDecodeKV | MixedKV | None = None,
        *,
        last_token_only: bool = False,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute logits for a chunk of one sequence, or for a batch of sequence chunks.

        Args:
            input_ids: ``[num_tokens]`` token ids (flattened across sequences for a batch).
            positions: ``[num_tokens]`` absolute positions (each continues its sequence's cache).
            kv_cache: One sequence's cache, a :class:`BatchedKV` describing several sequences'
                segments, a :class:`BatchedDecodeKV` (one new token per sequence, batched
                attention), or a :class:`MixedKV` (batched single-token rows followed by prefill
                segments). Caches are written for all layers and advanced before returning.
            last_token_only: Return logits only for the final token. The LM head is the single
                largest matmul for small models, so skipping it during prefill matters.
            logits_indices: Flattened-batch indices whose logits are needed (e.g. the last token of
                each sequence that will sample). Takes precedence over ``last_token_only``.

        Returns:
            Float logits ``[k, vocab]`` for the selected tokens (all tokens by default).
        """
        if input_ids.dim() != 1 or input_ids.shape != positions.shape:
            raise ValueError("input_ids and positions must be 1-D tensors of equal length")
        n = input_ids.shape[0]
        segments: Sequence[Segment] | BatchedDecodeKV | MixedKV
        if isinstance(kv_cache, MixedKV):
            kv_cache.validate(n)
            segments = kv_cache
        elif isinstance(kv_cache, BatchedDecodeKV):
            if n != kv_cache.batch_size:
                raise ValueError("batched decode takes exactly one token per sequence")
            segments = kv_cache
        elif isinstance(kv_cache, BatchedKV):
            kv_cache.validate(n)
            segments = kv_cache.segments
        else:
            segments = (Segment(0, n, kv_cache),)
        h = self.model.embed_tokens(input_ids)
        cos, sin = rope_cos_sin(self.inv_freq, positions, h.dtype)
        for layer in self.model.layers:
            h = layer(h, cos, sin, segments)
        if isinstance(kv_cache, BatchedKV | BatchedDecodeKV | MixedKV):
            kv_cache.advance()
        elif kv_cache is not None:
            kv_cache.advance(n)
        h = self.model.norm(h)
        if logits_indices is not None:
            h = h.index_select(0, logits_indices)
        elif last_token_only:
            h = h[-1:]
        weight = self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(h, weight)
