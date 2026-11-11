"""M6 batched decode attention vs the per-sequence reference path (fp32, tiny checkpoints).

The two paths run the same engine on the same requests; every decode step's logits are compared
row by row, so any indexing slip (wrong slot, wrong block, unmasked padding, wrong GQA head
mapping) shows up as a numeric difference, not just as a different sampled token.
"""

from __future__ import annotations

import random

import pytest
import torch
from torch.nn import functional as F

from dli.engine import LLMEngine, SchedulerConfig
from dli.engine.runner import ModelRunner
from dli.model import LlamaForCausalLM, load_model
from dli.sampling import SamplingParams

from .conftest import TinyFactory


class Recorder:
    """Wraps a runner: records each step's logits and whether the batched decode path ran."""

    def __init__(self, inner: ModelRunner) -> None:
        self.inner = inner
        self.steps: list[torch.Tensor] = []
        self.batched = 0
        orig = inner._execute_decode

        def spy(sched: object) -> torch.Tensor:
            self.batched += 1
            return orig(sched)  # type: ignore[arg-type]

        inner._execute_decode = spy  # type: ignore[method-assign]

    def execute(self, sched: object) -> torch.Tensor:
        out = self.inner.execute(sched)  # type: ignore[arg-type]
        self.steps.append(out.detach().clone())
        return out


def _run(
    model: LlamaForCausalLM,
    *,
    batched: bool,
    num_blocks: int,
    budget: int,
    seed: int,
    shared_prefix: bool = False,
) -> tuple[list[list[int]], Recorder, int]:
    eng = LLMEngine.for_model(
        model,
        num_blocks=num_blocks,
        block_size=4,
        config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=budget),
        batched_decode=batched,
    )
    rec = Recorder(eng.runner)  # type: ignore[arg-type]
    eng.runner = rec  # type: ignore[assignment]
    rng = random.Random(seed)
    head = [rng.randrange(3, 97) for _ in range(9)]
    reqs = []
    for i in range(10):
        tail = [rng.randrange(3, 97) for _ in range(rng.randrange(1, 30))]
        prompt = head + tail if shared_prefix else tail
        reqs.append(
            eng.add_request(
                f"r{i}", prompt, SamplingParams(max_tokens=4 + (i * 5) % 17, ignore_eos=True)
            )
        )
    while eng.has_unfinished():
        eng.step()
    eng.bm.check_invariants()
    return [r.output_ids for r in reqs], rec, sum(r.num_preemptions for r in reqs)


@pytest.mark.parametrize(
    ("variant", "num_blocks", "budget", "shared"),
    [
        ("llama", 256, 512, False),  # GQA 8 heads / 2 KV heads, roomy
        ("qwen2", 256, 32, False),  # biases, chunked prefill interleaved with decode
        ("mha", 256, 512, False),  # no GQA (enable_gqa off)
        ("llama", 24, 64, False),  # tight pool: preemption + recompute, scattered block tables
        ("llama", 256, 512, True),  # prefix sharing: block tables share physical blocks
    ],
)
def test_batched_decode_matches_per_sequence_reference(
    tiny_model: TinyFactory, variant: str, num_blocks: int, budget: int, shared: bool
) -> None:
    _, ckpt = tiny_model(variant)
    model = load_model(ckpt)
    kw = {
        "num_blocks": num_blocks,
        "budget": budget,
        "seed": num_blocks + budget,
        "shared_prefix": shared,
    }
    ref_out, ref, _ = _run(model, batched=False, **kw)
    new_out, new, preemptions = _run(model, batched=True, **kw)
    assert ref.batched == 0
    assert new.batched > 0, "batched decode path never ran"
    assert new_out == ref_out  # identical greedy tokens
    assert len(new.steps) == len(ref.steps)  # identical schedules
    worst = max(float((a - b).abs().max()) for a, b in zip(ref.steps, new.steps, strict=True))
    assert worst < 1e-4, f"max logit difference {worst}"
    if num_blocks == 24:
        assert preemptions > 0, "preemption not exercised"


def test_one_attention_call_per_layer_whatever_the_batch(
    tiny_model: TinyFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launch-count claim, checked structurally: SDPA calls per decode step do not grow."""
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    layers = model.config.num_layers
    calls = {"n": 0}
    real = F.scaled_dot_product_attention

    def counting(*a: object, **k: object) -> torch.Tensor:
        calls["n"] += 1
        return real(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(F, "scaled_dot_product_attention", counting)
    for batched, expected in ((True, lambda b: layers), (False, lambda b: layers * b)):
        for bsz in (2, 6):
            eng = LLMEngine.for_model(
                model,
                num_blocks=128,
                block_size=4,
                config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=512),
                batched_decode=batched,
            )
            for i in range(bsz):
                eng.add_request(
                    f"r{i}", [5 + i, 6, 7], SamplingParams(max_tokens=3, ignore_eos=True)
                )
            eng.step()  # prefill step
            calls["n"] = 0
            eng.step()  # steady-state decode: every request has one new token
            assert calls["n"] == expected(bsz), (batched, bsz, calls["n"])
