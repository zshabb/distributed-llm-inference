"""Assemble a complete server (model, engine, async bridge, app) from a model directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI

from dli.engine import LLMEngine, SchedulerConfig
from dli.model import LlamaForCausalLM, load_model
from dli.server.admission import AdmissionConfig
from dli.server.app import ServerConfig, create_app
from dli.server.async_engine import AsyncEngine
from dli.server.text import TextCodec


@dataclass
class ServerBundle:
    app: FastAPI
    engine: AsyncEngine
    codec: TextCodec
    model: LlamaForCausalLM
    config: ServerConfig


@dataclass(frozen=True)
class BuildOptions:
    model_name: str
    num_blocks: int = 512
    block_size: int = 16
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    default_max_tokens: int = 16
    max_tokens_limit: int = 4096
    request_timeout_s: float = 600.0
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)


def build_server(model_dir: str | Path, opts: BuildOptions) -> ServerBundle:
    """Load weights and tokenizer from ``model_dir`` and wire every serving component.

    The engine is constructed here (caller's thread) and handed to the AsyncEngine thread on start.
    """
    model = load_model(model_dir, dtype=opts.dtype, device=opts.device)
    codec = TextCodec(model_dir)
    engine = LLMEngine.for_model(
        model,
        num_blocks=opts.num_blocks,
        block_size=opts.block_size,
        config=opts.scheduler,
        eos_token_ids=codec.eos_token_ids,
    )
    aengine = AsyncEngine(engine)
    config = ServerConfig(
        model_name=opts.model_name,
        max_model_len=model.config.max_position_embeddings,
        kv_capacity_tokens=opts.num_blocks * opts.block_size,
        vocab_size=model.config.vocab_size,
        default_max_tokens=opts.default_max_tokens,
        max_tokens_limit=opts.max_tokens_limit,
        request_timeout_s=opts.request_timeout_s,
    )
    app = create_app(aengine, codec, config, opts.admission)
    return ServerBundle(app, aengine, codec, model, config)
