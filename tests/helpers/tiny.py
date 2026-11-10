"""Tiny, fully real test artifacts: random-weight checkpoints and matching tokenizers.

The character-level tokenizer has exactly 97 tokens, the tiny models' vocabulary, so a tiny
checkpoint directory can be served end to end over HTTP in milliseconds without downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

TINY: dict[str, Any] = {
    "vocab_size": 97,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
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

CHAT_TEMPLATE = (
    "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)


def build_tiny_model(variant: str, out: Path) -> torch.nn.Module:
    """Save a random tiny HF checkpoint for ``variant`` in ``out``; return the reference model."""
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    spec = dict(VARIANTS[variant])
    arch = spec.pop("arch")
    torch.manual_seed(1234)
    kwargs = {**TINY, **spec}
    model: torch.nn.Module
    if arch == "llama":
        model = LlamaForCausalLM(LlamaConfig(**kwargs))
    else:
        model = Qwen2ForCausalLM(Qwen2Config(**kwargs))
    with torch.no_grad():  # perturb norms/biases so they are not trivially 1/0
        for name, p in model.named_parameters():
            if name.endswith(("norm.weight", ".bias")):
                p.add_(0.1 * torch.randn_like(p))
    model.save_pretrained(out)
    return model.eval()


def write_char_tokenizer(out: Path) -> None:
    """97-token character tokenizer: <unk>, <s>, </s>, then printable ASCII 32..125."""
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for c in range(32, 126):
        vocab[chr(c)] = len(vocab)
    assert len(vocab) == 97
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(["<unk>", "<s>", "</s>"])
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>", "chat_template": CHAT_TEMPLATE})
    )
    (out / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))


def write_byte_bpe_tokenizer(out: Path) -> Tokenizer:
    """Small byte-level BPE trained in memory: multi-byte characters split across tokens."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["</s>"],
    )
    tok.train_from_iterator(["hello world", "the quick brown fox", "naive cafe"] * 20, trainer)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({"eos_token": "</s>"}))
    return tok
