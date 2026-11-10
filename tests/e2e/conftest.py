"""Real HTTP end-to-end harness: a uvicorn server on a real TCP socket, backed by a real engine.

The model is a tiny random Llama checkpoint with a matching 97-token character tokenizer, so every
layer runs for real (HTTP parsing, validation, tokenization, chat template, scheduler, paged KV,
model forward, sampling, detokenization, SSE) in milliseconds, with no downloads.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import uvicorn

from dli.engine import SchedulerConfig
from dli.server.factory import BuildOptions, ServerBundle, build_server
from tests.helpers.tiny import build_tiny_model, write_char_tokenizer

MODEL = "tiny-llama"


@dataclass
class LiveServer:
    url: str
    bundle: ServerBundle
    server: uvicorn.Server
    thread: threading.Thread

    def client(self, **kw: object) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=30.0, **kw)  # type: ignore[arg-type]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def start_server(bundle: ServerBundle) -> LiveServer:
    port = _free_port()
    config = uvicorn.Config(bundle.app, host="127.0.0.1", port=port, log_level="warning",
                            lifespan="off")  # fmt: skip
    server = uvicorn.Server(config)
    bundle.engine.start()
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.01)
    return LiveServer(f"http://127.0.0.1:{port}", bundle, server, thread)


def stop_server(live: LiveServer) -> None:
    live.server.should_exit = True
    live.thread.join(10)
    live.bundle.engine.stop()


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("tiny-server")
    build_tiny_model("llama", d)
    write_char_tokenizer(d)
    return d


ServerFactory = Callable[..., LiveServer]


@pytest.fixture
def make_server(tiny_dir: Path) -> Iterator[ServerFactory]:
    started: list[LiveServer] = []

    def make(**overrides: object) -> LiveServer:
        sched = overrides.pop(
            "scheduler", SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=128)
        )
        opts = BuildOptions(
            model_name=MODEL,
            num_blocks=int(overrides.pop("num_blocks", 256)),  # type: ignore[call-overload]
            block_size=4,
            scheduler=sched,
            **overrides,
        )  # type: ignore[arg-type]
        live = start_server(build_server(tiny_dir, opts))
        started.append(live)
        return live

    yield make
    for live in started:
        stop_server(live)


@pytest.fixture
def server(make_server: ServerFactory) -> LiveServer:
    return make_server()


def wait_for(pred: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached in time")
