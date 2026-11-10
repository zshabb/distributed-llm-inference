from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from dli.model import CheckpointError, ModelConfig, UnsupportedModelError, load_model
from dli.model.kv_cache import ContiguousKVCache

from .conftest import TinyFactory

HUB_LLAMA: dict[str, Any] = {  # Hub-style config.json (legacy rope keys), SmolLM2-like
    "model_type": "llama",
    "vocab_size": 49152,
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "rms_norm_eps": 1e-5,
    "rope_theta": 100000,
    "rope_scaling": None,
    "tie_word_embeddings": True,
    "hidden_act": "silu",
}


def test_hub_llama_config() -> None:
    cfg = ModelConfig.from_hf_dict(HUB_LLAMA)
    assert (cfg.head_dim, cfg.num_queries_per_kv) == (64, 3)
    assert cfg.rope.theta == 100000
    assert cfg.rope.rope_type == "default"
    assert not cfg.qkv_bias
    assert not cfg.o_bias


def test_qwen2_bias_layout() -> None:
    cfg = ModelConfig.from_hf_dict(
        {
            **HUB_LLAMA,
            "model_type": "qwen2",
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "use_sliding_window": False,
        }
    )
    assert cfg.qkv_bias
    assert not cfg.o_bias


def test_v5_rope_parameters_and_legacy_type_key() -> None:
    v5 = ModelConfig.from_hf_dict(
        {
            **HUB_LLAMA,
            "rope_theta": None,
            "rope_parameters": {
                "rope_type": "llama3",
                "rope_theta": 5e5,
                "factor": 8.0,
                "original_max_position_embeddings": 8192,
            },
        }
    )
    assert (v5.rope.rope_type, v5.rope.theta, v5.rope.factor) == ("llama3", 5e5, 8.0)
    legacy = ModelConfig.from_hf_dict(
        {**HUB_LLAMA, "rope_scaling": {"type": "linear", "factor": 2}}
    )
    assert legacy.rope.rope_type == "linear"


@pytest.mark.parametrize(
    "override",
    [
        {"model_type": "mistral"},
        {"hidden_act": "gelu"},
        {"use_sliding_window": True},
        {"layer_types": ["full_attention", "sliding_attention"]},
        {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
        {"pretraining_tp": 2},
        {"num_key_value_heads": 4},  # 9 heads not divisible by 4
    ],
)
def test_unsupported_features_are_rejected(override: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedModelError):
        ModelConfig.from_hf_dict({**HUB_LLAMA, **override})


def _ckpt_file(ckpt: Path) -> Path:
    return next(ckpt.glob("*.safetensors"))


def test_pickle_only_checkpoint_is_refused(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    f = _ckpt_file(ckpt)
    f.rename(ckpt / "pytorch_model.bin")
    with pytest.raises(CheckpointError, match="pickle"):
        load_model(ckpt)


def test_missing_tensor_is_an_error(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    f = _ckpt_file(ckpt)
    state = load_file(f)
    del state["model.layers.1.mlp.up_proj.weight"]
    save_file(state, f)
    with pytest.raises(CheckpointError, match="missing"):
        load_model(ckpt)


def test_shape_mismatch_is_an_error(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    f = _ckpt_file(ckpt)
    state = load_file(f)
    state["model.norm.weight"] = torch.ones(63)
    save_file(state, f)
    with pytest.raises(CheckpointError, match="shape"):
        load_model(ckpt)


def test_tied_checkpoint_with_divergent_lm_head_is_rejected(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama-tied")
    f = _ckpt_file(ckpt)
    state = load_file(f)
    state["lm_head.weight"] = torch.randn_like(state["model.embed_tokens.weight"])
    save_file(state, f)
    with pytest.raises(CheckpointError, match="tie_word_embeddings"):
        load_model(ckpt)


def test_config_on_disk_round_trip(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("qwen2")
    raw = json.loads((ckpt / "config.json").read_text())
    assert load_model(ckpt).config == ModelConfig.from_hf_dict(raw)


def test_kv_cache_rejects_overflow_and_gaps() -> None:
    cache = ContiguousKVCache(1, 2, 4, max_len=8, dtype=torch.float32, device=torch.device("cpu"))
    kv = torch.zeros(2, 3, 4)
    cache.write(0, 0, kv, kv)
    with pytest.raises(ValueError, match="non-contiguous"):
        cache.write(0, 5, kv, kv)
    cache.advance(3)
    with pytest.raises(ValueError, match="overflow"):
        cache.write(0, 3, torch.zeros(2, 6, 4), torch.zeros(2, 6, 4))
