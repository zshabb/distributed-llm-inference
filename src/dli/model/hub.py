"""Download supported models at pinned revisions.

Revisions are commit hashes, not branch names: ``main`` can change under us, which would silently
invalidate parity results and benchmarks. Only open, ungated (Apache-2.0) models are listed, so no
Hugging Face token is ever needed, on the laptop or on the GPU host.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

# repo id -> pinned commit (verified 2026-11-10 via the Hub API).
PINNED_REVISIONS: dict[str, str] = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    # M5 benchmark ladder (verified 2026-11-11, Apache-2.0, ungated). Qwen2.5-3B-Instruct, named
    # in the spec, is excluded: its license is the non-Apache "Qwen Research" license.
    # SmolLM2-1.7B takes its place (no GQA: ~7x the KV bytes per token of Qwen2.5-1.5B).
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
    # Baseline-only (vLLM): AWQ 4-bit weights. The native engine has no quantized path yet.
    "Qwen/Qwen2.5-7B-Instruct-AWQ": "b25037543e9394b818fdfca67ab2a00ecc7dd641",
}

# Only what inference needs. Excluding *.bin guarantees no pickle checkpoint is ever downloaded.
_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def download(repo_id: str, cache_dir: str | Path | None = None) -> Path:
    """Fetch ``repo_id`` at its pinned revision and return the local snapshot directory.

    Raises:
        KeyError: ``repo_id`` has no pinned revision (pin it deliberately before use).
    """
    revision = PINNED_REVISIONS[repo_id]
    path = snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=_ALLOW_PATTERNS,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(path)
