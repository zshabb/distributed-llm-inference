"""Immutable model configuration parsed from a Hugging Face ``config.json``.

The parser is deliberately strict: an inference engine that silently ignores an unsupported
feature (sliding-window attention, an unknown RoPE variant) produces plausible-looking but wrong
text, which is far worse than refusing to load. Every rejected feature raises
:class:`UnsupportedModelError` with the offending key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_MODEL_TYPES = frozenset({"llama", "qwen2"})
SUPPORTED_ROPE_TYPES = frozenset({"default", "linear", "llama3"})


class UnsupportedModelError(ValueError):
    """The checkpoint uses an architecture feature this engine does not implement."""


@dataclass(frozen=True, slots=True)
class RopeConfig:
    """Rotary position embedding parameters.

    Attributes:
        theta: Base of the geometric frequency progression.
        rope_type: ``default``, ``linear`` (position interpolation), or ``llama3`` (frequency-band
            dependent scaling used by Llama 3.1+).
        factor: Scaling factor for ``linear``/``llama3``; 1.0 otherwise.
        low_freq_factor, high_freq_factor, original_max_position_embeddings: ``llama3`` only.
    """

    theta: float
    rope_type: str = "default"
    factor: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture hyper-parameters for a Llama-family decoder-only transformer.

    Qwen2 is represented as the Llama graph with biases on the q/k/v projections, which is the
    only architectural difference relevant to inference for the supported checkpoints.
    """

    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope: RopeConfig
    tie_word_embeddings: bool
    qkv_bias: bool
    o_bias: bool
    mlp_bias: bool

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise UnsupportedModelError(
                f"num_attention_heads={self.num_heads} is not a multiple of "
                f"num_key_value_heads={self.num_kv_heads}"
            )
        if self.head_dim % 2 != 0:
            raise UnsupportedModelError(f"head_dim={self.head_dim} must be even for RoPE")

    @property
    def num_queries_per_kv(self) -> int:
        """Grouped-query attention fan-out (query heads sharing one KV head)."""
        return self.num_heads // self.num_kv_heads

    @classmethod
    def from_hf_dict(cls, cfg: dict[str, Any]) -> ModelConfig:
        """Build a config from a parsed ``config.json``.

        Accepts both the Hub format (``rope_theta`` + ``rope_scaling``) and the transformers v5
        format (``rope_parameters``).

        Raises:
            UnsupportedModelError: Unknown architecture or an unsupported feature is enabled.
            KeyError: A required field is missing.
        """
        model_type = cfg.get("model_type")
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise UnsupportedModelError(f"model_type={model_type!r} is not supported")

        act = cfg.get("hidden_act", "silu")
        if act != "silu":
            raise UnsupportedModelError(f"hidden_act={act!r} (only 'silu' / SwiGLU is supported)")
        if cfg.get("use_sliding_window") or any(
            t != "full_attention" for t in cfg.get("layer_types") or []
        ):
            raise UnsupportedModelError("sliding-window attention is not supported")
        if cfg.get("pretraining_tp", 1) not in (1, None):
            raise UnsupportedModelError("pretraining_tp > 1 changes numerics and is not supported")

        num_heads = int(cfg["num_attention_heads"])
        hidden = int(cfg["hidden_size"])
        attention_bias = bool(cfg.get("attention_bias", False))
        is_qwen2 = model_type == "qwen2"
        return cls(
            model_type=model_type,
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(cfg["intermediate_size"]),
            num_layers=int(cfg["num_hidden_layers"]),
            num_heads=num_heads,
            num_kv_heads=int(cfg.get("num_key_value_heads") or num_heads),
            head_dim=int(cfg.get("head_dim") or hidden // num_heads),
            rms_norm_eps=float(cfg["rms_norm_eps"]),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 4096)),
            rope=_parse_rope(cfg),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            # Qwen2 always has q/k/v biases and never an o_proj bias;
            # Llama uses one flag for all four.
            qkv_bias=True if is_qwen2 else attention_bias,
            o_bias=False if is_qwen2 else attention_bias,
            mlp_bias=bool(cfg.get("mlp_bias", False)),
        )


def _parse_rope(cfg: dict[str, Any]) -> RopeConfig:
    params: dict[str, Any] = dict(cfg.get("rope_parameters") or {})
    scaling: dict[str, Any] = dict(cfg.get("rope_scaling") or {})
    merged = {**scaling, **params}  # rope_parameters (v5) wins over legacy rope_scaling
    theta = merged.get("rope_theta", cfg.get("rope_theta", 10000.0))
    # Legacy checkpoints spell the key "type" instead of "rope_type".
    rope_type = merged.get("rope_type", merged.get("type", "default"))
    if rope_type not in SUPPORTED_ROPE_TYPES:
        raise UnsupportedModelError(f"rope_type={rope_type!r} is not supported")
    if rope_type == "default":
        return RopeConfig(theta=float(theta))
    return RopeConfig(
        theta=float(theta),
        rope_type=rope_type,
        factor=float(merged["factor"]),
        low_freq_factor=float(merged.get("low_freq_factor", 1.0)),
        high_freq_factor=float(merged.get("high_freq_factor", 4.0)),
        original_max_position_embeddings=int(
            merged.get("original_max_position_embeddings", cfg.get("max_position_embeddings", 8192))
        ),
    )
