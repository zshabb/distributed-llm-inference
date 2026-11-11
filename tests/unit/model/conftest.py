"""Tiny randomly initialized reference models, saved as real safetensors checkpoints.

They exercise the full loader + model path against `transformers` in milliseconds, without any
download, and cover architecture variants the pinned real checkpoints do not (untied heads,
Llama attention bias, llama3/linear RoPE scaling). Builders live in ``tests/helpers/tiny.py`` so the
server tests can reuse them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from tests.helpers.tiny import TINY, VARIANTS, build_tiny_model

__all__ = ["TINY", "VARIANTS", "TinyFactory"]

TinyFactory = Callable[[str], tuple[torch.nn.Module, Path]]


@pytest.fixture
def tiny_model(tmp_path: Path) -> TinyFactory:
    """Return a factory: variant name -> (reference HF model in eval/fp32, checkpoint dir)."""

    def make(variant: str) -> tuple[torch.nn.Module, Path]:
        out = tmp_path / variant
        return build_tiny_model(variant, out), out

    return make
