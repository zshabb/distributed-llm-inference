"""M6 mixed steps: batched single-token rows + per-sequence prefill vs the reference path (fp32).

A step that schedules a prefill chunk next to decode rows used to send every row through the
per-sequence path. Now the single-token rows are attended in one batched call and only the prefill
chunks run per sequence. Both engines run the same requests with the same schedules; every step's
logits are compared row by row, and the KV pools are compared at the end.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest
import torch
from torch.nn import functional as F

from dli.engine import LLMEngine, SchedulerConfig
from dli.engine.runner import ModelRunner
from dli.engine.scheduler import SchedulerOutput
from dli.model import LlamaForCausalLM, load_model
from dli.sampling import SamplingParams

from .conftest import TinyFactory


class Recorder:
    """Wraps a runner: records every step's logits and counts mixed-path steps."""

    def __init__(self, inner: ModelRunner) -> None:
        self.inner = inner
        self.steps: list[torch.Tensor] = []
        self.mixed = 0
        orig = inner._execute_mixed

        def spy(sched: SchedulerOutput) -> torch.Tensor:
            self.mixed += 1
            return orig(sched)

        inner._execute_mixed = spy  # type: ignore[method-assign]

    def execute(self, sched: SchedulerOutput) -> torch.Tensor:
        out = self.inner.execute(sched)
        self.steps.append(out.detach().clone())
        return out


Reorder = Callable[[SchedulerOutput], SchedulerOutput]


def _run(
    model: LlamaForCausalLM,
    *,
    batched: bool,
    num_blocks: int,
    budget: int,
    seed: int,
    shared_prefix: bool = False,
    reorder: Reorder | None = None,
) -> tuple[list[list[int]], Recorder, LLMEngine, int]:
    eng = LLMEngine.for_model(
        model,
        num_blocks=num_blocks,
        block_size=4,
        config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=budget),
        batched_decode=batched,
    )
    rec = Recorder(eng.runner)  # type: ignore[arg-type]
    eng.runner = rec  # type: ignore[assignment]
    if reorder is not None:
        schedule = eng.scheduler.schedule
        eng.scheduler.schedule = lambda: reorder(schedule())  # type: ignore[method-assign]
    rng = random.Random(seed)
    head = [rng.randrange(3, 97) for _ in range(9)]
    reqs = []
    # Staggered arrivals: new prompts keep landing while earlier requests decode (mixed steps).
    for i in range(14):
        tail = [rng.randrange(3, 97) for _ in range(rng.randrange(2, 30))]
        prompt = head + tail if shared_prefix else tail
        params = SamplingParams(max_tokens=4 + (i * 5) % 17, ignore_eos=True)
        reqs.append(eng.add_request(f"r{i}", prompt, params))
        for _ in range(i % 3):
            if eng.has_unfinished():
                eng.step()
    while eng.has_unfinished():
        eng.step()
    eng.bm.check_invariants()
    return [r.output_ids for r in reqs], rec, eng, sum(r.num_preemptions for r in reqs)


def _reverse(sched: SchedulerOutput) -> SchedulerOutput:
    """Schedule order with prefill and decode chunks interleaved differently from the default."""
    sched.chunks.reverse()
    return sched


@pytest.mark.parametrize(
    ("variant", "num_blocks", "budget", "shared", "reorder"),
    [
        ("llama", 256, 64, False, None),  # GQA 8 heads / 2 KV heads
        ("qwen2", 256, 24, False, None),  # biases; small budget = chunked prefill next to decode
        ("mha", 256, 64, False, None),  # no GQA
        ("llama", 28, 32, False, None),  # tight pool: preemption, recompute, scattered tables
        ("llama", 256, 64, True, None),  # prefix sharing: shared physical blocks in mixed steps
        ("llama", 256, 64, False, _reverse),  # logits must follow schedule order, not row layout
    ],
)
def test_mixed_steps_match_the_per_sequence_reference(
    tiny_model: TinyFactory,
    variant: str,
    num_blocks: int,
    budget: int,
    shared: bool,
    reorder: Reorder | None,
) -> None:
    _, ckpt = tiny_model(variant)
    model = load_model(ckpt)
    kw = {
        "num_blocks": num_blocks,
        "budget": budget,
        "seed": num_blocks + budget,
        "shared_prefix": shared,
        "reorder": reorder,
    }
    ref_out, ref, ref_eng, _ = _run(model, batched=False, **kw)  # type: ignore[arg-type]
    new_out, new, new_eng, preemptions = _run(model, batched=True, **kw)  # type: ignore[arg-type]
    assert ref.mixed == 0
    assert new.mixed > 0, "mixed path never ran"
    assert new_out == ref_out  # identical greedy tokens
    assert len(new.steps) == len(ref.steps)  # identical schedules
    worst = max(float((a - b).abs().max()) for a, b in zip(ref.steps, new.steps, strict=True))
    assert worst < 1e-4, f"max logit difference {worst}"
    # Same KV everywhere: every slot written by either path holds the same keys and values.
    for name in ("k", "v"):
        a = getattr(ref_eng.runner.inner.store, name)  # type: ignore[attr-defined]
        b = getattr(new_eng.runner.inner.store, name)  # type: ignore[attr-defined]
        assert torch.allclose(a, b, atol=1e-5), f"KV pool {name} differs"
    if num_blocks == 28:
        assert preemptions > 0, "preemption not exercised"


def test_mixed_step_attention_calls_do_not_grow_with_decode_rows(
    tiny_model: TinyFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One batched call for all single-token rows plus one per prefill chunk, per layer."""
    _, ckpt = tiny_model("llama")
    model = load_model(ckpt)
    layers = model.config.num_layers
    calls = {"n": 0}
    real = F.scaled_dot_product_attention

    def counting(*a: object, **k: object) -> torch.Tensor:
        calls["n"] += 1
        return real(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(F, "scaled_dot_product_attention", counting)
    for decoding in (2, 6):
        eng = LLMEngine.for_model(
            model,
            num_blocks=128,
            block_size=4,
            config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=512),
        )
        for i in range(decoding):
            eng.add_request(f"d{i}", [5 + i, 6, 7], SamplingParams(max_tokens=4, ignore_eos=True))
        eng.step()  # prefill of the decoding requests
        eng.add_request("p", list(range(3, 15)), SamplingParams(max_tokens=2, ignore_eos=True))
        calls["n"] = 0
        eng.step()  # mixed: `decoding` single-token rows + one 12-token prefill chunk
        assert calls["n"] == 2 * layers, (decoding, calls["n"])
