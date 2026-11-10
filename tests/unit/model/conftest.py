"""Tiny randomly initialized reference models, saved as real safetensors checkpoints.

They exercise the full loader + model path against `transformers` in milliseconds, without any
download, and cover architecture variants the pinned real checkpoints do not (untied heads,
Llama attention bias, llama3/linear RoPE scaling).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

TINY: dict[str, Any] = {
    "vocab_size": 97,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,  # GQA fan-out 4
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
}

VARIANTS: dict[str, dict[str, Any]] = {
    "llama": {"arch": "llama"},
    "llama-tied": {"arch": "llama", "tie_word_embeddings": True},
    "llama-bias": {"arch": "llama", "attention_bias": True, "mlp_bias": True},
    "llama3-rope": {
        "arch": "llama",
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 32,
        },
    },
    "linear-rope": {"arch": "llama", "rope_scaling": {"rope_type": "linear", "factor": 2.0}},
    "mha": {"arch": "llama", "num_key_value_heads": 8},
    "qwen2": {"arch": "qwen2", "tie_word_embeddings": True},
}

TinyFactory = Callable[[str], tuple[torch.nn.Module, Path]]


@pytest.fixture
def tiny_model(tmp_path: Path) -> TinyFactory:
    """Return a factory: variant name -> (reference HF model in eval/fp32, checkpoint dir)."""

    def make(variant: str) -> tuple[torch.nn.Module, Path]:
        spec = dict(VARIANTS[variant])
        arch = spec.pop("arch")
        torch.manual_seed(1234)
        kwargs = {**TINY, **spec}
        model: torch.nn.Module
        if arch == "llama":
            model = LlamaForCausalLM(LlamaConfig(**kwargs))
        else:
            model = Qwen2ForCausalLM(Qwen2Config(**kwargs))
        # Random init draws small weights; perturb norms/biases so they are not trivially 1/0.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith(("norm.weight", ".bias")):
                    p.add_(0.1 * torch.randn_like(p))
        out = tmp_path / variant
        model.save_pretrained(out)
        return model.eval(), out

    return make
