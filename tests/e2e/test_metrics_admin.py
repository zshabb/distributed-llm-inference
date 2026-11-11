"""Prometheus metrics and the admin listener over real HTTP."""

from __future__ import annotations

from prometheus_client.parser import text_string_to_metric_families

from .conftest import MODEL, LiveServer, ServerFactory


def scrape(live: LiveServer) -> dict[str, float]:
    """Flatten the exposition into {"name{label=value,...}": value}."""
    with live.admin() as a:
        r = a.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    out: dict[str, float] = {}
    for fam in text_string_to_metric_families(r.text):
        for sample in fam.samples:
            labels = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
            out[f"{sample.name}{{{labels}}}"] = sample.value
    return out


def test_metrics_reflect_traffic(server: LiveServer) -> None:
    with server.client() as c:
        for i in range(3):
            assert (
                c.post(
                    "/v1/completions",
                    json={"model": MODEL, "prompt": f"m{i}", "max_tokens": 8, "ignore_eos": True},
                ).status_code
                == 200
            )
        with c.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "s"}],
                "max_tokens": 5,
                "stream": True,
                "ignore_eos": True,
            },
        ) as r:
            list(r.iter_lines())
        assert (
            c.post("/v1/completions", json={"model": MODEL, "prompt": "x", "n": 3}).status_code
            == 400
        )
    m = scrape(server)
    assert m["dli_requests_total{endpoint=/v1/completions,status=200}"] == 3
    assert m["dli_requests_total{endpoint=/v1/chat/completions,status=200}"] == 1
    assert m["dli_requests_total{endpoint=/v1/completions,status=400}"] == 1
    assert m["dli_request_errors_total{type=invalid_request_error}"] == 1
    assert m["dli_generation_tokens_total{}"] == 3 * 8 + 5
    assert m["dli_time_to_first_token_seconds_count{}"] == 4
    assert m["dli_inter_token_latency_seconds_count{}"] == 3 * 7 + 4
    assert m["dli_request_latency_seconds_count{}"] == 4
    assert m["dli_prompt_tokens_total{}"] > 0
    assert m["dli_kv_blocks_total{}"] == 256
    assert m["dli_engine_healthy{}"] == 1
    assert m["dli_engine_steps_total{}"] > 0
    assert "process_resident_memory_bytes{}" in m


def test_rejections_are_counted_by_reason(make_server: ServerFactory) -> None:
    live = make_server()
    live.bundle.app.state.draining = True
    with live.client() as c:
        assert c.post("/v1/completions", json={"model": MODEL, "prompt": "d"}).status_code == 503
    assert scrape(live)["dli_admission_rejections_total{reason=draining}"] == 1


def test_admin_state_and_drain(server: LiveServer) -> None:
    with server.admin() as a, server.client() as c:
        state = a.get("/admin/state").json()
        assert state["draining"] is False
        assert state["engine"]["kv_total_blocks"] == 256
        assert a.post("/admin/drain").json()["draining"] is True
        assert c.get("/readyz").status_code == 503
        assert c.post("/v1/completions", json={"model": MODEL, "prompt": "x"}).status_code == 503
        assert a.post("/admin/undrain").json()["draining"] is False
        assert c.get("/readyz").status_code == 200


def test_admin_token_is_enforced(make_server: ServerFactory) -> None:
    live = make_server(admin_token="s3cret-admin")
    with live.admin() as a:
        assert a.get("/metrics").status_code == 401
        assert a.post("/admin/drain", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = a.get("/metrics", headers={"Authorization": "Bearer s3cret-admin"})
        assert ok.status_code == 200
    assert live.bundle.app.state.draining is False


def test_metrics_are_not_exposed_on_the_public_port(server: LiveServer) -> None:
    with server.client() as c:
        assert c.get("/metrics").status_code == 404
        assert c.post("/admin/drain").status_code == 404


def test_prefix_caching_can_be_disabled(make_server: ServerFactory) -> None:
    """Ablation switch: with prefix caching off, a repeated prompt gets no cache hits."""
    body = {"model": MODEL, "prompt": "shared prefix " * 8, "max_tokens": 2, "temperature": 0}
    on, off = make_server(), make_server(enable_prefix_caching=False)
    for live in (on, off):
        with live.client() as c:
            for _ in range(3):
                assert c.post("/v1/completions", json=body).status_code == 200
    assert scrape(on)["dli_prefix_cache_hit_ratio{}"] > 0
    assert scrape(off)["dli_prefix_cache_hit_ratio{}"] == 0
