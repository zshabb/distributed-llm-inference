"""The server's opt-in decode CUDA graphs: flag, default off, eager fallback, metrics, GPU parity.

CPU tests run everywhere (CI): the flag is off by default, refused on CPU devices, and a server
built with graphs on a CPU device serves normally with every decode step counted as a fallback.
The ``gpu`` tests serve the tiny model from CUDA through the graphs and compare with eager.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest
import torch

from dli.engine import SchedulerConfig
from dli.engine.cuda_graph import BENCHMARK_SHAPES
from dli.server.__main__ import build, parse_args, warm_up
from dli.server.factory import BuildOptions, build_server

from .conftest import MODEL, LiveServer, ServerFactory, start_server, stop_server
from .test_metrics_admin import scrape


def _complete(live: LiveServer, prompt: str, max_tokens: int = 12) -> str:
    with live.client() as c:
        r = c.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        )
    assert r.status_code == 200, r.text
    text: str = r.json()["choices"][0]["text"]
    return text


def test_cuda_graphs_are_off_by_default(server: LiveServer) -> None:
    assert parse_args(["--model", "m"]).cuda_graphs is False
    assert parse_args(["--model", "m", "--cuda-graphs"]).cuda_graphs is True
    _complete(server, "hello")
    m = scrape(server)
    assert not any(k.startswith("dli_cuda_graph") for k in m), "graph metrics without graphs"
    with server.admin() as a:
        assert a.get("/admin/state").json()["engine"]["cuda_graph_status"] == "off"


def test_cuda_graphs_flag_needs_a_cuda_device(tiny_dir: Path) -> None:
    args = parse_args(["--model", str(tiny_dir), "--no-auth", "--device", "cpu", "--cuda-graphs"])
    with pytest.raises(SystemExit, match="needs a CUDA --device"):
        build(args)


def test_graphs_on_a_cpu_device_fall_back_to_eager(make_server: ServerFactory) -> None:
    """The factory accepts graphs on any device; on CPU they disable themselves, serve eagerly."""
    plain = make_server()
    graphed = make_server(cuda_graphs=BENCHMARK_SHAPES)
    for i in range(3):
        assert _complete(graphed, f"p{i}") == _complete(plain, f"p{i}")
    m = scrape(graphed)
    assert m["dli_cuda_graph_captured{}"] == 0
    assert m["dli_cuda_graph_replays_total{}"] == 0
    assert m["dli_cuda_graph_fallbacks_total{reason=disabled: not a CUDA device}"] > 0
    with graphed.admin() as a:
        state = a.get("/admin/state").json()["engine"]
    assert state["cuda_graph_status"] == "disabled: not a CUDA device"


# ---------------------------------------------------------------------------- GPU


def _cuda_server(tiny_dir: Path, graphs: bool) -> LiveServer:
    opts = BuildOptions(
        model_name=MODEL,
        num_blocks=1024,
        block_size=4,
        device="cuda",
        scheduler=SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=512),
        cuda_graphs=BENCHMARK_SHAPES if graphs else None,
    )
    bundle = build_server(tiny_dir, opts)
    warm_up(bundle)  # what the server process does before serving: captures the graphs
    return start_server(bundle)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_graph_server_matches_eager_and_reports_usage(tiny_dir: Path) -> None:
    eager, graphed = _cuda_server(tiny_dir, False), _cuda_server(tiny_dir, True)
    try:
        # One request at a time: identical batches on both servers, so greedy text must match.
        prompts = [f"q{i} " * (1 + i % 5) for i in range(6)]
        assert [_complete(graphed, p, 20) for p in prompts] == [
            _complete(eager, p, 20) for p in prompts
        ]
        # A concurrent burst (batches of up to 40 rows: the 64-row bucket, shrinking into the
        # 16-row one as requests finish) must succeed and be served by replays.
        with concurrent.futures.ThreadPoolExecutor(40) as pool:
            texts = list(pool.map(lambda i: _complete(graphed, f"b{i}", 8 + i % 9), range(40)))
        assert all(texts)
        m = scrape(graphed)
        assert m["dli_cuda_graph_captured{}"] == 1
        assert m["dli_cuda_graph_replays_total{}"] > 20
        assert m["dli_cuda_graph_memory_bytes{}"] > 0
        assert m["dli_engine_preemptions_total{}"] == 0
        with graphed.admin() as a:
            assert a.get("/admin/state").json()["engine"]["cuda_graph_status"] == "captured"
    finally:
        stop_server(eager)
        stop_server(graphed)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_engine_process_with_graphs_matches_the_threaded_server(tiny_dir: Path) -> None:
    """--engine process --cuda-graphs: graphs captured in the child, same greedy text, usage
    and GPU memory reported through the snapshot."""
    threaded = _cuda_server(tiny_dir, True)
    opts = BuildOptions(
        model_name=MODEL,
        num_blocks=1024,
        block_size=4,
        device="cuda",
        scheduler=SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=512),
        cuda_graphs=BENCHMARK_SHAPES,
        engine_process=True,
    )
    proc = start_server(build_server(tiny_dir, opts))
    try:
        assert proc.bundle.model is None
        prompts = [f"z{i} " * (1 + i % 4) for i in range(5)]
        assert [_complete(proc, p, 20) for p in prompts] == [
            _complete(threaded, p, 20) for p in prompts
        ]
        with concurrent.futures.ThreadPoolExecutor(40) as pool:
            assert all(pool.map(lambda i: _complete(proc, f"c{i}", 8 + i % 9), range(40)))
        m = scrape(proc)
        assert m["dli_cuda_graph_captured{}"] == 1
        assert m["dli_cuda_graph_replays_total{}"] > 20
        assert m["dli_gpu_memory_reserved_bytes{}"] > 0
    finally:
        stop_server(proc)
        stop_server(threaded)
