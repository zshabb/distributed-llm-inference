"""Strict checkpoint loading from ``safetensors`` files.

Security: only ``safetensors`` is accepted. PyTorch ``.bin`` checkpoints are pickles, and loading
them can execute arbitrary code, which is unacceptable for weights fetched from the internet.

Correctness: names and shapes must match the module exactly. A silently skipped tensor would
leave randomly initialized (or, with meta-device init, unallocated) weights in the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from dli.model.config import ModelConfig
from dli.model.llama import LlamaForCausalLM


class CheckpointError(ValueError):
    """The checkpoint directory is unusable (format, missing or unexpected tensors, shapes)."""


def _safetensor_files(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if files:
        return files
    if any(model_dir.glob("*.bin")) or any(model_dir.glob("*.pt")):
        raise CheckpointError(
            f"{model_dir} only has pickle-based weights (.bin/.pt); refusing to load. "
            "Use a safetensors revision of the model."
        )
    raise CheckpointError(f"no *.safetensors files in {model_dir}")


def load_config(model_dir: str | Path) -> ModelConfig:
    """Parse ``config.json`` in ``model_dir``."""
    return ModelConfig.from_hf_dict(json.loads((Path(model_dir) / "config.json").read_text()))


def load_model(
    model_dir: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> LlamaForCausalLM:
    """Load a supported checkpoint directory into an eval-mode :class:`LlamaForCausalLM`.

    The module is built on the ``meta`` device (no allocation, no random init), and real tensors
    are then *assigned* from the checkpoint, so peak memory is about one copy of the weights.

    Args:
        model_dir: Directory with ``config.json`` and one or more ``*.safetensors`` shards.
        dtype: Parameter dtype after loading (checkpoint tensors are cast).
        device: Target device.

    Raises:
        CheckpointError: Pickle-only checkpoint, missing/unexpected tensors, or shape mismatch.
        UnsupportedModelError: The architecture is not supported (see ``ModelConfig``).
    """
    model_dir = Path(model_dir)
    cfg = load_config(model_dir)
    device = torch.device(device)

    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)

    state: dict[str, torch.Tensor] = {}
    for f in _safetensor_files(model_dir):
        for name, tensor in load_file(str(f), device=str(device)).items():
            state[name] = tensor.to(dtype)

    if cfg.tie_word_embeddings and "lm_head.weight" in state:
        # Some tied checkpoints still serialize lm_head;
        # accept it only if it really is the embedding.
        lm_head = state.pop("lm_head.weight")
        if not torch.equal(lm_head, state["model.embed_tokens.weight"]):
            raise CheckpointError("tie_word_embeddings=true but lm_head differs from embeddings")

    expected = {name: p.shape for name, p in model.state_dict().items()}
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    if missing or unexpected:
        raise CheckpointError(f"missing tensors: {missing[:5]} ... unexpected: {unexpected[:5]}")
    bad = [n for n, shape in expected.items() if state[n].shape != shape]
    if bad:
        raise CheckpointError(f"shape mismatch for {bad[:5]}")

    model.load_state_dict(state, strict=True, assign=True)
    model.reset_rope(device)
    return model.eval()
