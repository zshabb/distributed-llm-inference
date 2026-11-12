"""Assemble a complete server (model, engine, async bridge, app) from a model directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI

from dli.engine import LLMEngine, SchedulerConfig
from dli.engine.cuda_graph import Shape
from dli.model import LlamaForCausalLM, load_config, load_model
from dli.server.admin import create_admin_app
from dli.server.admission import AdmissionConfig
from dli.server.app import ServerConfig, create_app
from dli.server.async_engine import AsyncEngine, EngineClient
from dli.server.auth import ApiKeyAuth
from dli.server.metrics import ServerMetrics
from dli.server.text import TextCodec


@dataclass
class ServerBundle:
    app: FastAPI
    admin_app: FastAPI
    engine: EngineClient
    codec: TextCodec
    model: LlamaForCausalLM | None  # None when the engine (and the weights) live in a child process
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
    auth: ApiKeyAuth | None = None
    max_body_bytes: int = 2 * 1024 * 1024
    admin_token: str | None = None
    enable_prefix_caching: bool = True
    cuda_graphs: tuple[Shape, ...] | None = None  # decode CUDA graph buckets (None: eager only)
    engine_process: bool = False  # run the engine in a child process (no GIL shared with HTTP)
    decode_attention: str = "auto"  # batched decode kernel: auto (Triton on CUDA), triton, sdpa


def build_server(model_dir: str | Path, opts: BuildOptions) -> ServerBundle:
    """Load weights and tokenizer from ``model_dir`` and wire every serving component.

    The engine is constructed here (caller's thread) and handed to the AsyncEngine thread on start.
    """
    codec = TextCodec(model_dir)
    model: LlamaForCausalLM | None = None
    aengine: EngineClient
    if opts.engine_process:
        from dli.server.engine_process import EngineSpec, ProcessEngine

        cfg = load_config(model_dir)
        aengine = ProcessEngine(
            EngineSpec(
                model_dir=str(model_dir),
                num_blocks=opts.num_blocks,
                block_size=opts.block_size,
                device=opts.device,
                dtype=opts.dtype,
                scheduler=opts.scheduler,
                eos_token_ids=tuple(codec.eos_token_ids),
                enable_prefix_caching=opts.enable_prefix_caching,
                cuda_graphs=opts.cuda_graphs,
                decode_attention=opts.decode_attention,
            ),
            kv_total_blocks=opts.num_blocks,
        )
    else:
        model = load_model(model_dir, dtype=opts.dtype, device=opts.device)
        cfg = model.config
        engine = LLMEngine.for_model(
            model,
            num_blocks=opts.num_blocks,
            block_size=opts.block_size,
            config=opts.scheduler,
            eos_token_ids=codec.eos_token_ids,
            enable_prefix_caching=opts.enable_prefix_caching,
            cuda_graphs=opts.cuda_graphs,
            decode_attention=opts.decode_attention,
        )
        aengine = AsyncEngine(engine)
    config = ServerConfig(
        model_name=opts.model_name,
        max_model_len=cfg.max_position_embeddings,
        kv_capacity_tokens=opts.num_blocks * opts.block_size,
        vocab_size=cfg.vocab_size,
        default_max_tokens=opts.default_max_tokens,
        max_tokens_limit=opts.max_tokens_limit,
        request_timeout_s=opts.request_timeout_s,
        max_body_bytes=opts.max_body_bytes,
    )
    metrics = ServerMetrics(aengine)
    app = create_app(aengine, codec, config, opts.admission, opts.auth, metrics)
    admin = create_admin_app(app, aengine, metrics, opts.admin_token)
    return ServerBundle(app, admin, aengine, codec, model, config)
