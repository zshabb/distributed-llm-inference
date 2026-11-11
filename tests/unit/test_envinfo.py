import json
from pathlib import Path

import pytest

from dli import envinfo


def test_collect_is_json_serializable_with_stable_top_level_keys() -> None:
    env = envinfo.collect()
    json.dumps(env)  # must not raise
    assert env["schema"] == "dli-env/v1"
    assert {"revision", "os", "cpu", "memory_gib", "python", "torch", "gpu"} <= env.keys()
    assert env["torch"] is not None
    assert env["torch"]["version"]


def test_missing_tools_are_recorded_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulates a CPU-only machine: no nvidia-smi on PATH must not raise.
    monkeypatch.setattr(envinfo.shutil, "which", lambda _name: None)
    assert envinfo._gpu_info() is None


def test_main_writes_manifest_file(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "env.json"
    assert envinfo.main([str(out)]) == 0
    assert json.loads(out.read_text())["schema"] == "dli-env/v1"


def test_explicit_start_revision_is_recorded_not_the_current_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression: a long run must report the revision captured at its start, even if HEAD moved
    # (commits made during the first M1 parity run were wrongly attributed to it).
    monkeypatch.setattr(envinfo, "current_revision", lambda: "moved-on-head")
    assert envinfo.collect(revision="start-rev")["revision"] == "start-rev"
    out = tmp_path / "env.json"
    envinfo.main([str(out)], revision="start-rev")
    assert json.loads(out.read_text())["revision"] == "start-rev"
    assert envinfo.collect()["revision"] == "moved-on-head"


def test_untracked_files_mark_the_revision_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression (M2): git describe --dirty ignores untracked files, so a run of an uncommitted
    # new test file was labeled with a clean revision.
    outputs = {"describe": "abc123", "status": "?? tests/new_test.py"}
    monkeypatch.setattr(
        envinfo,
        "_run",
        lambda cmd, timeout=20.0: outputs["describe" if cmd[1] == "describe" else "status"],
    )
    monkeypatch.setattr(envinfo.Path, "is_file", lambda self: False)
    assert envinfo.current_revision() == "abc123-dirty"
    outputs["status"] = ""
    assert envinfo.current_revision() == "abc123"


def _cgroup_v1(root: Path, quota: int, period: int) -> None:
    (root / "cpu").mkdir(parents=True)
    (root / "cpu" / "cpu.cfs_quota_us").write_text(f"{quota}\n")
    (root / "cpu" / "cpu.cfs_period_us").write_text(f"{period}\n")


def test_env_records_container_limits_but_never_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cgroup_v1(tmp_path, 765000, 100000)
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text(str(50 * 2**30))
    monkeypatch.setenv("RUNPOD_DC_ID", "EU-SE-1")
    monkeypatch.setenv("RUNPOD_PUBLIC_IP", "203.0.113.7")
    monkeypatch.setenv("RUNPOD_TCP_PORT_22", "22124")
    info = envinfo._container_info(tmp_path)
    assert info["cpu_quota"] == 7.65
    assert info["memory_limit_gib"] == 50.0
    assert info["provider"]["RUNPOD_DC_ID"] == "EU-SE-1"
    assert "203.0.113.7" not in repr(info)
    assert "22124" not in repr(info)
