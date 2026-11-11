# M4 HTTP parity, bf16 on one A16: criterion failed (analysis)

`[GPU:A16x1@vultr]` · code `7463dd4` · `tests/parity/test_http_real_checkpoint.py` · 24 prompts × ≤ 48 greedy
tokens, each sent twice (non-streaming and streaming), all 48 requests concurrent.

## What failed

The test, as written before the run, asserted for every dtype that each streamed response equals its non-streamed
twin. In bf16 that held for 11/24 (SmolLM2-135M) and 15/24 (Qwen2.5-0.5B) prompts. Pytest therefore failed for
both models. The result files are kept unmodified.

| | SmolLM2-135M | Qwen2.5-0.5B |
|---|---|---|
| Prompt token counts equal to `transformers` | 24/24 | 24/24 |
| Text identical to the unbatched reference | 13/24 | 9/24 |
| Reference top-2 logit gap at each divergence | all ≤ 0.25 (four exact ties, 0.0) | all ≤ 0.375 (four exact ties, 0.0) |
| Stream text == non-stream text | 11/24 | 15/24 |
| finish_reason and usage correct (identical prompts) | 13/13 | 9/9 |

## Explanation

This is the effect measured in M3 (`2026-11-10-GPU-A16x1-vultr-m3-engine-parity-bfloat16/ANALYSIS.md`): in bf16 the
result of a row depends on which other rows share its batch, because batched matmuls sum in a different order and
bf16 keeps only 8 mantissa bits. Greedy decoding then flips at near-ties, and the sequences diverge from there. The
streamed and non-streamed twins of a prompt are admitted at different times into different batches, so they are
two samples of this noise, not a test of the streaming code.

Why this is not a server defect:
- The same code, run the same way, gives stream == non-stream for 24/24 prompts and text identical to the reference
  for 24/24 prompts in fp32, both on this GPU and on the laptop CPU (`*-m4-http-parity-float32/`). The stream path
  (detokenizer, stop matcher, SSE) is dtype-independent text processing.
- Every divergence from the reference is at a reference near-tie, within the M1/M3 bf16 rule (gap ≤ 1.0).

What this does **not** show: the gaps at the stream-vs-non-stream divergences were not recorded (the reference gap is
known only for divergences from the reference). Their attribution to batch noise rests on the fp32 control, not on
direct measurement.

## Decision

The stream-equality assertion was a mistake for bf16: M3 had already shown that bf16 output depends on batch
composition, and I did not carry that into this test. It now asserts stream equality only in fp32 and records it in
bf16, where the reference near-tie rule remains the pass/fail criterion. That change is a separate commit made after this
run, and this failed run remains the recorded bf16 result. Quality-level bf16 evaluation is planned in
`docs/evaluation-plan-m5.md`. Batch-invariant bf16 decoding (a fixed reduction order, so the same request gives the same
tokens in any batch) is not planned yet and is recorded as known issue #29.
