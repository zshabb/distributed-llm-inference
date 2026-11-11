"""Server entry point: ``python -m dli.server --model <dir-or-pinned-repo> [options]``.

Safe defaults: binds to 127.0.0.1, admin listener on loopback, and it refuses to start without API
keys on a non-loopback address. Shutdown (SIGINT/SIGTERM) drains first: new requests get 503
while in-flight ones finish (bounded by ``--drain-timeout``), then the engine stops.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import logging
import os
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import uvicorn

from dli.engine import SchedulerConfig
from dli.engine.cuda_graph import BENCHMARK_SHAPES
from dli.kv.paged_store import PagedKVStore
from dli.model import hub, load_config
from dli.server.admission import AdmissionConfig
from dli.server.auth import ApiKeyAuth
from dli.server.factory import BuildOptions, ServerBundle, build_server

log = logging.getLogger("dli.server")


class _ManagedServer(uvicorn.Server):
    """uvicorn server whose shutdown is driven by our drain logic.

    uvicorn's serve() installs its own SIGINT/SIGTERM handlers that exit at once, which would skip
    draining. Overriding capture_signals (its signal hook) leaves the signals to serve() below.
    """

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class _DropScrapes(logging.Filter):
    """Keep periodic `GET /metrics` scrapes out of the access log; log everything else."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()
        return not (len(args) >= 3 and args[1] == "GET" and args[2] == "/metrics")


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="dli-serve", description=__doc__)
    p.add_argument("--model", required=True, help="model directory, or a pinned Hub repo id")
    p.add_argument("--served-model-name", default=None, help="id clients send (default: --model)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--admin-host", default="127.0.0.1")
    p.add_argument("--admin-port", type=int, default=8001)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    kv = p.add_mutually_exclusive_group()
    kv.add_argument("--num-blocks", type=int, default=None)
    kv.add_argument("--kv-cache-gb", type=float, default=None, help="KV pool size in GiB")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="disable KV prefix reuse across requests (for ablations)",
    )
    p.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="replay decode steps through CUDA graphs (buckets of 16/64 sequences x 24 KV blocks, "
        "captured at start-up); steps that fit no bucket run eagerly. CUDA devices only. Off by "
        "default",
    )
    p.add_argument(
        "--engine",
        choices=["process", "thread"],
        default="process",
        help="where the engine runs: its own process (default; its Python work never waits for "
        "the HTTP event loop's GIL, and the API process does not load the weights) or a thread "
        "of the API process (the original design)",
    )
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--max-queued-requests", type=int, default=256)
    p.add_argument("--max-inflight-requests", type=int, default=512)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--drain-timeout", type=float, default=30.0)
    p.add_argument(
        "--api-keys-file",
        type=Path,
        default=None,
        help="JSON key records from `python -m dli.server.auth new <tenant>`",
    )
    p.add_argument("--no-auth", action="store_true", help="disable auth (loopback hosts only)")
    p.add_argument(
        "--admin-token-env",
        default="DLI_ADMIN_TOKEN",
        help="environment variable holding the admin bearer token (optional)",
    )
    return p.parse_args(argv)


def build(args: argparse.Namespace) -> ServerBundle:
    if args.no_auth and not _is_loopback(args.host):
        raise SystemExit("refusing --no-auth on a non-loopback --host: configure --api-keys-file")
    if not args.no_auth and args.api_keys_file is None:
        raise SystemExit(
            "authentication is on by default: pass --api-keys-file (or --no-auth "
            "for local development on a loopback host)"
        )
    if args.cuda_graphs and not args.device.startswith("cuda"):
        raise SystemExit("--cuda-graphs needs a CUDA --device")
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        model_dir = hub.download(args.model)  # pinned revisions only (KeyError if unknown)
    dtype = {
        "auto": torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    num_blocks = args.num_blocks
    if num_blocks is None:
        cfg = load_config(model_dir)
        gib = args.kv_cache_gb if args.kv_cache_gb is not None else 1.0
        num_blocks = PagedKVStore.num_blocks_for_budget(
            cfg, args.block_size, dtype, int(gib * 2**30)
        )
    opts = BuildOptions(
        model_name=args.served_model_name or args.model,
        num_blocks=num_blocks,
        block_size=args.block_size,
        device=args.device,
        dtype=dtype,
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens
        ),
        request_timeout_s=args.request_timeout,
        admission=AdmissionConfig(
            max_inflight_requests=args.max_inflight_requests,
            max_queued_requests=args.max_queued_requests,
        ),
        auth=None if args.no_auth else ApiKeyAuth.from_file(args.api_keys_file),
        admin_token=os.environ.get(args.admin_token_env) or None,
        enable_prefix_caching=not args.no_prefix_caching,
        cuda_graphs=BENCHMARK_SHAPES if args.cuda_graphs else None,
        engine_process=args.engine == "process",
    )
    return build_server(model_dir, opts)


def warm_up(bundle: ServerBundle) -> None:
    """One short forward before accepting traffic: warms kernels, surfaces load failures early.

    With ``--cuda-graphs`` the decode graphs are captured here too, before the engine thread starts
    (so the capture neither delays the first request nor runs next to other GPU work). A bucket
    that fails its memory check or capture is logged and its steps run eagerly.
    """
    if bundle.model is None:
        return  # the engine process loads, warms up and captures on its own before it is ready
    with torch.inference_mode():
        ids = torch.tensor([1, 2, 3], device=bundle.model.inv_freq.device)
        bundle.model(ids, torch.arange(3, device=ids.device), last_token_only=True)
    graphs = getattr(bundle.engine, "decode_graphs", None)
    if graphs is not None:
        with torch.inference_mode():
            graphs.capture()
        log.info("decode CUDA graphs: %s", graphs.summary())


async def serve(bundle: ServerBundle, args: argparse.Namespace) -> None:
    api = _ManagedServer(
        uvicorn.Config(bundle.app, host=args.host, port=args.port, log_level="info", lifespan="off")
    )
    # uvicorn's loggers are process-wide and every Config reconfigures them (level, and access_log
    # =False removes the access handler for BOTH servers). So both use the same settings, and
    # Prometheus scrapes are filtered out of the access log instead.
    admin = _ManagedServer(
        uvicorn.Config(
            bundle.admin_app,
            host=args.admin_host,
            port=args.admin_port,
            log_level="info",
            lifespan="off",
        )
    )
    logging.getLogger("uvicorn.access").addFilter(_DropScrapes())
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    tasks = [asyncio.create_task(api.serve()), asyncio.create_task(admin.serve())]
    await stop.wait()
    log.info("shutdown requested: draining (new requests get 503)")
    bundle.app.state.draining = True
    deadline = time.monotonic() + args.drain_timeout
    while bundle.app.state.admission.inflight and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    left = bundle.app.state.admission.inflight
    if left:
        log.warning("drain timeout: %d request(s) still in flight are being cut off", left)
    api.should_exit = admin.should_exit = True
    await asyncio.gather(*tasks, return_exceptions=True)
    bundle.engine.stop()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    bundle = build(args)
    warm_up(bundle)
    bundle.engine.start()
    info = getattr(bundle.engine, "startup_info", None)
    if info:
        log.info("engine process ready: %s", info)
    log.info(
        "serving %s on http://%s:%d (admin http://%s:%d), KV pool %d blocks x %d",
        bundle.config.model_name,
        args.host,
        args.port,
        args.admin_host,
        args.admin_port,
        bundle.config.kv_capacity_tokens // args.block_size,
        args.block_size,
    )
    asyncio.run(serve(bundle, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
