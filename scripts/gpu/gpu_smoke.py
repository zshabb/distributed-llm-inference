"""Bounded GPU smoke validation: is this device usable and numerically sane?

This is NOT a benchmark. The timing numbers use one problem size and few iterations; they only
detect gross problems (e.g. a throttled, unlicensed vGPU or a wrong device). Benchmarks live in
benchmarks/ and follow benchmarks/methodology.md.

Writes a JSON report to the path given as argv[1] (default: stdout). Exit code 1 if any
correctness check fails.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got.double() - ref.double()).abs().max() / ref.double().abs().max())


def _matmul_tflops(dev: torch.device, dtype: torch.dtype, n: int = 4096, iters: int = 20) -> float:
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * iters / (time.perf_counter() - t0) / 1e12


def _copy_gbps(dev: torch.device, nbytes: int = 256 * 2**20, iters: int = 20) -> float:
    x = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    x.clone()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        x.clone()
    torch.cuda.synchronize()
    return 2 * nbytes * iters / (time.perf_counter() - t0) / 1e9  # read + write


def _triton_check(dev: torch.device) -> dict[str, Any]:
    """Compile and run a trivial Triton kernel: verifies the JIT toolchain (needs a C compiler)."""
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        return {"ok": False, "error": f"import failed: {exc}"}

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, block: tl.constexpr):  # type: ignore[no-untyped-def]
        offs = tl.program_id(0) * block + tl.arange(0, block)
        mask = offs < n
        tl.store(
            out_ptr + offs,
            tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
            mask=mask,
        )

    try:
        n = 98_765
        x, y = torch.randn(n, device=dev), torch.randn(n, device=dev)
        out = torch.empty_like(x)
        add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, block=1024)
        torch.cuda.synchronize()
        return {"ok": bool(torch.equal(out, x + y)), "version": triton.__version__}
    except Exception as exc:  # report, don't crash: this is a diagnostic
        return {
            "ok": False,
            "version": triton.__version__,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }


def main() -> int:
    report: dict[str, Any] = {
        "kind": "gpu-smoke (not a benchmark)",
        "checks": {},
        "timings_smoke": {},
    }
    if not torch.cuda.is_available():
        report["error"] = "torch.cuda.is_available() is False"
        print(json.dumps(report, indent=2))
        return 1
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    report["device"] = {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_gib": round(props.total_memory / 2**30, 2),
        "multiprocessors": props.multi_processor_count,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    torch.manual_seed(0)
    checks = report["checks"]

    # 1. fp32 matmul vs float64 CPU reference (TF32 off: measures plain fp32 numerics).
    torch.backends.cuda.matmul.allow_tf32 = False
    a, b = torch.randn(512, 512, dtype=torch.float64), torch.randn(512, 512, dtype=torch.float64)
    err = _rel_err((a.float().to(dev) @ b.float().to(dev)).cpu(), a @ b)
    checks["matmul_fp32_vs_fp64"] = {"max_rel_err": err, "ok": err < 1e-5}

    # 2. bf16 matmul: loose tolerance (8-bit mantissa).
    err = _rel_err((a.to(dev, torch.bfloat16) @ b.to(dev, torch.bfloat16)).cpu(), a @ b)
    checks["matmul_bf16_vs_fp64"] = {"max_rel_err": err, "ok": err < 5e-2}

    # 3. Attention kernel path the engine will use (SDPA, causal, GQA-expanded heads).
    q, k, v = (torch.randn(2, 8, 128, 64) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = F.scaled_dot_product_attention(q.to(dev), k.to(dev), v.to(dev), is_causal=True).cpu()
    err = _rel_err(got, ref)
    checks["sdpa_causal_fp32_vs_cpu"] = {"max_rel_err": err, "ok": err < 1e-4}

    # 4. Allocation near the advertised memory size (catches vGPU profiles smaller than reported).
    free, total = torch.cuda.mem_get_info()
    try:
        blob = torch.empty(int(free * 0.9), dtype=torch.uint8, device=dev)
        blob.fill_(1)
        torch.cuda.synchronize()
        del blob
        torch.cuda.empty_cache()
        alloc_ok = True
    except torch.cuda.OutOfMemoryError:
        alloc_ok = False
    checks["allocate_90pct_free"] = {
        "free_gib": round(free / 2**30, 2),
        "total_gib": round(total / 2**30, 2),
        "ok": alloc_ok,
    }

    checks["triton_vector_add"] = _triton_check(dev)

    # Smoke timings (single size, few iterations; NOT benchmark results).
    torch.backends.cuda.matmul.allow_tf32 = False
    report["timings_smoke"]["matmul4096_fp32_tflops"] = round(_matmul_tflops(dev, torch.float32), 2)
    torch.backends.cuda.matmul.allow_tf32 = True
    report["timings_smoke"]["matmul4096_tf32_tflops"] = round(_matmul_tflops(dev, torch.float32), 2)
    report["timings_smoke"]["matmul4096_fp16_tflops"] = round(_matmul_tflops(dev, torch.float16), 2)
    report["timings_smoke"]["matmul4096_bf16_tflops"] = round(
        _matmul_tflops(dev, torch.bfloat16), 2
    )
    report["timings_smoke"]["device_copy_gbps"] = round(_copy_gbps(dev), 1)
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() // 2**20

    required = [name for name in checks if name != "triton_vector_add"]
    report["ok"] = all(checks[name]["ok"] for name in required)
    text = json.dumps(report, indent=2)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text + "\n")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
