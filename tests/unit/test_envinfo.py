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
