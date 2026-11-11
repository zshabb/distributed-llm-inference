"""Decode-attention kernel selection without a GPU: auto falls back to SDPA, triton is refused."""

from __future__ import annotations

import pytest

from dli.engine import LLMEngine
from dli.engine.cuda_graph import BENCHMARK_SHAPES, LONG_CONTEXT_SHAPES, default_shapes
from dli.model import load_model, paged_attention
from dli.server.__main__ import build, parse_args

from .conftest import TinyFactory


def test_auto_resolves_to_sdpa_on_cpu() -> None:
    assert paged_attention.resolve("auto", "cpu") == "sdpa"
    assert paged_attention.resolve("sdpa", "cpu") == "sdpa"
    with pytest.raises(ValueError, match="needs a CUDA device"):
        paged_attention.resolve("triton", "cpu")
    with pytest.raises(ValueError, match="unknown decode_attention"):
        paged_attention.resolve("flash", "cpu")


def test_engine_on_cpu_uses_sdpa_and_refuses_triton(tiny_model: TinyFactory) -> None:
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    assert LLMEngine.for_model(model, num_blocks=16, block_size=4).runner.decode_attention == "sdpa"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="needs a CUDA device"):
        LLMEngine.for_model(model, num_blocks=16, block_size=4, decode_attention="triton")


def test_wide_graph_buckets_only_with_the_triton_kernel() -> None:
    assert default_shapes("sdpa") == BENCHMARK_SHAPES  # padding to 160 blocks would cost SDPA
    assert default_shapes("triton") == LONG_CONTEXT_SHAPES
    assert max(w for _, w in LONG_CONTEXT_SHAPES) * 16 >= 2048 + 512  # longest workload contexts


def test_server_flag(tiny_model: TinyFactory) -> None:
    assert parse_args(["--model", "m"]).decode_attention == "auto"
    d, _ = tiny_model("llama")
    args = parse_args(
        ["--model", str(d), "--no-auth", "--device", "cpu", "--decode-attention", "triton"]
    )
    with pytest.raises(SystemExit, match="needs a CUDA device"):
        build(args)
