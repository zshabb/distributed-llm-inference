"""Provider-independent GPU host scripts: profiles, SSH options, CPU quota."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GPU = REPO / "scripts" / "gpu"


def bash(script: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, **env},
        check=False,
    )


@pytest.mark.parametrize("script", sorted(p.name for p in GPU.glob("*.sh")))
def test_scripts_parse(script: str) -> None:
    assert bash(f"bash -n scripts/gpu/{script}").returncode == 0


def _cgroup_v1(root: Path, quota: int, period: int) -> None:
    (root / "cpu").mkdir(parents=True)
    (root / "cpu" / "cpu.cfs_quota_us").write_text(f"{quota}\n")
    (root / "cpu" / "cpu.cfs_period_us").write_text(f"{period}\n")


def test_cpu_threads_follow_the_quota_not_the_visible_cores(tmp_path: Path) -> None:
    visible = os.cpu_count() or 1
    _cgroup_v1(tmp_path / "v1", 765000, 100000)  # the RunPod A5000 pod: 7.65 CPUs of 96 visible
    out = bash("bash scripts/gpu/cpu_threads.sh", CGROUP_ROOT=str(tmp_path / "v1"))
    assert int(out.stdout) == min(7, visible)
    (tmp_path / "v2").mkdir()
    (tmp_path / "v2" / "cpu.max").write_text("150000 100000\n")
    assert (
        int(bash("bash scripts/gpu/cpu_threads.sh", CGROUP_ROOT=str(tmp_path / "v2")).stdout) == 1
    )
    (tmp_path / "v2" / "cpu.max").write_text("max 100000\n")  # no quota: a plain VM
    assert (
        int(bash("bash scripts/gpu/cpu_threads.sh", CGROUP_ROOT=str(tmp_path / "v2")).stdout)
        == visible
    )


SHOW = (
    "source scripts/gpu/common.sh; "
    'echo "$GPU_HOST|$GPU_RESULT_LABEL|$GPU_RESULTS_DIR|$GPU_HOURLY_USD"; '
    'printf "%s\\n" "${SSH_OPTS[@]}"'
)


def test_profile_selection_and_alias_identity(tmp_path: Path) -> None:
    (tmp_path / "pod.env").write_text(
        "GPU_HOST=my-alias\nGPU_SSH_PUBKEY=\nGPU_RESULTS_DIR=/workspace/dli/results\n"
        "GPU_RESULT_LABEL=GPU-X1-test\nGPU_HOURLY_USD=0.28\n"
    )
    out = bash(SHOW, GPU_PROFILE="pod", GPU_PROFILE_DIR=str(tmp_path))
    assert out.returncode == 0, out.stderr
    first, *opts = out.stdout.splitlines()
    assert first == "my-alias|GPU-X1-test|/workspace/dli/results|0.28"
    assert "-i" not in opts  # identity comes from the ssh_config alias
    for safety in (
        "StrictHostKeyChecking=yes",
        "ForwardAgent=no",
        "IdentitiesOnly=yes",
        "BatchMode=yes",
    ):
        assert safety in opts


def test_profile_with_public_key_and_defaults(tmp_path: Path) -> None:
    (tmp_path / "vm.env").write_text("GPU_HOST=root@203.0.113.10\nGPU_SSH_PUBKEY=/k.pub\n")
    out = bash(SHOW, GPU_PROFILE="vm", GPU_PROFILE_DIR=str(tmp_path))
    first, *opts = out.stdout.splitlines()
    host, label, results, price = first.split("|")
    assert (host, label, price) == ("root@203.0.113.10", "GPU-A16x1-vultr", "0.471")
    assert results == "work/distributed-llm-inference/benchmarks/results"
    assert opts[opts.index("-i") + 1] == "/k.pub"


def test_missing_profile_is_an_error(tmp_path: Path) -> None:
    out = bash(SHOW, GPU_PROFILE="nope", GPU_PROFILE_DIR=str(tmp_path))
    assert out.returncode == 64
    assert "no host profile" in out.stderr


def test_real_profiles_are_never_committed() -> None:
    out = subprocess.run(
        ["git", "check-ignore", "scripts/gpu/hosts/anything.env"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0  # ignored
    tracked = subprocess.run(
        ["git", "ls-files", "scripts/gpu/hosts"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert all(f.endswith((".example", ".md")) for f in tracked)


def test_scripts_are_committed_executable() -> None:
    """Rewriting a script can drop its mode; remote.sh lost +x once during the M5 migration work."""
    out = subprocess.run(
        ["git", "ls-files", "-s", "scripts/gpu/*.sh"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in out.stdout.splitlines():
        mode, *_, path = line.split()
        assert mode == "100755", path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")
def test_pulling_results_never_overwrites_committed_evidence(tmp_path: Path) -> None:
    """A stale host copy of a committed result must not replace it (it happened twice in M6)."""
    repo, host = tmp_path / "repo", tmp_path / "host"
    res = repo / "benchmarks" / "results"
    (res / "run-a").mkdir(parents=True)
    (res / "run-a" / "summary.json").write_text("committed\n")
    (res / "run-a" / "same.json").write_text("same\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "r")
    (res / "run-b").mkdir()
    (res / "run-b" / "partial.json").write_text("in progress\n")  # untracked: a running campaign
    for rel, text in {
        "run-a/summary.json": "stale host copy\n",  # differs from the committed file
        "run-a/same.json": "same\n",
        "run-b/partial.json": "finished\n",
        "run-c/new.json": "new result\n",
    }.items():
        (host / rel).parent.mkdir(parents=True, exist_ok=True)
        (host / rel).write_text(text)

    out = subprocess.run(
        [str(GPU / "pull_results.sh"), str(host), str(repo)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert (res / "run-a" / "summary.json").read_text() == "committed\n"
    assert (res / "run-b" / "partial.json").read_text() == "finished\n"
    assert (res / "run-c" / "new.json").read_text() == "new result\n"
    kept = res / ".pod-conflicts" / "run-a" / "summary.json"
    assert kept.read_text() == "stale host copy\n"  # unique host data is preserved, not dropped
    assert not (res / ".pod-conflicts" / "run-a" / "same.json").exists()
    assert "1 committed result file(s) differ" in out.stderr
    assert "run-a/summary.json" in out.stderr
    assert (host / "run-a" / "summary.json").read_text() == "stale host copy\n"  # host untouched
