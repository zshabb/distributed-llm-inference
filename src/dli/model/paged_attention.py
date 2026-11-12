# mypy: disable-error-code="untyped-decorator,no-untyped-def"
"""GQA-aware paged decode attention in Triton (M6).

One new query token per sequence attends to that sequence's keys, read **directly from the paged
KV pool through its block table**, for a whole query-head group at once:

* no page gather (the SDPA path copies every padded page of every sequence first);
* each KV head is read once for its ``group`` query heads (the SDPA path, with ``enable_gqa`` and a
  key mask, re-reads it per query head);
* only ``seq_len`` keys are read, never the padding up to the block-table width.

Measured on the A5000 with Qwen2.5-0.5B shapes (14/2 heads, d=64; 24 layers in a CUDA graph,
``benchmarks/results/2026-11-11-GPU-A5000x1-runpod-m6-attention-audit/``): 4.38 -> 0.52 ms per
decode step at B=64 and ~354-token contexts, 23.3 -> 2.47 ms at B=64 and ~2,000 tokens, with the
live KV read at 535-640 GB/s (peak 768) instead of 56-68 GB/s.

Split-K ("flash decoding"): when ``batch x kv_heads`` programs would leave SMs idle (small batches,
long contexts) the keys of each sequence are split into ``splits`` chunks whose partial softmax
states are merged by a second kernel.

The SDPA path in :mod:`dli.model.llama` stays the reference and the fallback (CPU, no Triton).
"""

from __future__ import annotations

import functools
import math

import torch

try:  # Triton ships with CUDA builds of PyTorch only
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installs
    triton = None
    tl = None


def available() -> bool:
    """True when the Triton kernel can run here (Triton importable and a CUDA device present)."""
    return triton is not None and torch.cuda.is_available()


def resolve(choice: str, device: torch.device | str) -> str:
    """``auto`` -> ``triton`` on CUDA with Triton available, else ``sdpa``. Validates the others."""
    dev = torch.device(device)
    if choice == "auto":
        return "triton" if dev.type == "cuda" and available() else "sdpa"
    if choice == "triton":
        if dev.type != "cuda" or not available():
            raise ValueError("decode_attention='triton' needs a CUDA device and Triton")
        return choice
    if choice == "sdpa":
        return choice
    raise ValueError(f"unknown decode_attention {choice!r} (auto, triton or sdpa)")


@functools.cache
def _num_sms(index: int) -> int:
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def choose_splits(batch: int, kv_heads: int, max_keys: int, device: torch.device) -> int:
    """Split-K factor: enough programs to fill the SMs twice, never chunks under 64 keys.

    On the A5000 (64 SMs) this gives 1 split at B=64 and 4 at B=16 with 2 KV heads, the best
    settings measured in the audit.
    """
    programs = batch * kv_heads
    target = 2 * _num_sms(device.index if device.index is not None else torch.cuda.current_device())
    splits = 1
    while programs * splits < target and splits < 16 and max_keys // (splits * 2) >= 64:
        splits *= 2
    return splits


if triton is not None:

    @triton.jit
    def _decode_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        tables_ptr,
        lens_ptr,
        out_ptr,
        acc_ptr,
        m_ptr,
        l_ptr,
        stride_qb,
        stride_qh,
        stride_kblk,
        stride_ktok,
        stride_kh,
        stride_vblk,
        stride_vtok,
        stride_vh,
        stride_tb,
        stride_ob,
        stride_oh,
        sm_scale,
        G: tl.constexpr,  # noqa: N803 - query heads per KV head
        GP: tl.constexpr,  # noqa: N803 - G padded (>= 16 for tl.dot)
        D: tl.constexpr,  # noqa: N803 - head dim
        DP: tl.constexpr,  # noqa: N803 - D padded to a power of two
        BS: tl.constexpr,  # noqa: N803 - KV block size (tokens per page)
        BLOCK_N: tl.constexpr,  # noqa: N803 - keys per iteration
        CHUNK: tl.constexpr,  # noqa: N803 - keys per split
        SPLITS: tl.constexpr,  # noqa: N803
        KVH: tl.constexpr,  # noqa: N803
        IEEE: tl.constexpr,  # noqa: N803 - exact fp32 dots (no TF32) for fp32 pools
    ):
        b = tl.program_id(0)
        kvh = tl.program_id(1)
        s = tl.program_id(2)
        seq_len = tl.load(lens_ptr + b)
        start = s * CHUNK
        end = tl.minimum(start + CHUNK, seq_len)
        offs_g = tl.arange(0, GP)
        offs_d = tl.arange(0, DP)
        gmask = offs_g < G
        dmask = offs_d < D
        q = tl.load(
            q_ptr + b * stride_qb + (kvh * G + offs_g)[:, None] * stride_qh + offs_d[None, :],
            mask=gmask[:, None] & dmask[None, :],
            other=0.0,
        )
        m_i = tl.full([GP], float("-inf"), tl.float32)
        l_i = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, DP], tl.float32)
        for n0 in range(start, end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            nmask = offs_n < end
            page = tl.load(tables_ptr + b * stride_tb + offs_n // BS, mask=nmask, other=0)
            tok = offs_n % BS
            kvmask = nmask[:, None] & dmask[None, :]
            k = tl.load(
                k_ptr
                + page[:, None] * stride_kblk
                + tok[:, None] * stride_ktok
                + kvh * stride_kh
                + offs_d[None, :],
                mask=kvmask,
                other=0.0,
            )
            if IEEE:  # noqa: SIM108 - compile-time branch (Triton constexpr)
                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
            else:
                qk = tl.dot(q, tl.trans(k))
            qk = tl.where(nmask[None, :], qk * sm_scale, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(
                v_ptr
                + page[:, None] * stride_vblk
                + tok[:, None] * stride_vtok
                + kvh * stride_vh
                + offs_d[None, :],
                mask=kvmask,
                other=0.0,
            )
            if IEEE:
                acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32), input_precision="ieee")
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        if SPLITS == 1:
            out = acc / l_i[:, None]
            tl.store(
                out_ptr + b * stride_ob + (kvh * G + offs_g)[:, None] * stride_oh + offs_d[None, :],
                out.to(out_ptr.dtype.element_ty),
                mask=gmask[:, None] & dmask[None, :],
            )
        else:
            row = (b * KVH + kvh) * SPLITS + s
            tl.store(acc_ptr + (row * GP + offs_g)[:, None] * DP + offs_d[None, :], acc)
            tl.store(m_ptr + row * GP + offs_g, m_i)
            tl.store(l_ptr + row * GP + offs_g, l_i)

    @triton.jit
    def _merge_kernel(
        out_ptr,
        acc_ptr,
        m_ptr,
        l_ptr,
        stride_ob,
        stride_oh,
        G: tl.constexpr,  # noqa: N803
        GP: tl.constexpr,  # noqa: N803
        D: tl.constexpr,  # noqa: N803
        DP: tl.constexpr,  # noqa: N803
        SPLITS: tl.constexpr,  # noqa: N803
        KVH: tl.constexpr,  # noqa: N803
    ):
        b = tl.program_id(0)
        kvh = tl.program_id(1)
        base = (b * KVH + kvh) * SPLITS
        offs_s = tl.arange(0, SPLITS)
        offs_g = tl.arange(0, GP)
        offs_d = tl.arange(0, DP)
        m = tl.load(m_ptr + (base + offs_s)[:, None] * GP + offs_g[None, :])  # [S, GP]
        lsum = tl.load(l_ptr + (base + offs_s)[:, None] * GP + offs_g[None, :])
        m_max = tl.max(m, 0)
        w = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_max[None, :]))
        denom = tl.sum(lsum * w, 0)
        out = tl.zeros([GP, DP], tl.float32)
        for i in tl.static_range(SPLITS):
            a = tl.load(acc_ptr + ((base + i) * GP + offs_g)[:, None] * DP + offs_d[None, :])
            mi = tl.load(m_ptr + (base + i) * GP + offs_g)
            out += a * tl.where(mi == float("-inf"), 0.0, tl.exp(mi - m_max))[:, None]
        out = out / denom[:, None]
        tl.store(
            out_ptr + b * stride_ob + (kvh * G + offs_g)[:, None] * stride_oh + offs_d[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=(offs_g < G)[:, None] & (offs_d < D)[None, :],
        )


def paged_decode_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    splits: int | None = None,
    block_n: int = 64,
) -> torch.Tensor:
    """Decode attention over paged KV.

    Args:
        q: ``[B, heads, d]`` (any strides with a contiguous last dimension).
        k_pool, v_pool: One layer's pools ``[num_blocks, block_size, kv_heads, d]``.
        block_tables: ``[B, width]`` int64 physical block ids; entries past a sequence's length
            are never read.
        seq_lens: ``[B]`` int64 keys per sequence, including the new token (>= 1).
        splits: Split-K factor (power of two); chosen from the shape when ``None``.

    Returns:
        ``[B, heads, d]`` attention output, scaled by ``1/sqrt(d)`` as in SDPA.
    """
    if triton is None:
        raise RuntimeError("Triton is not available")
    bsz, heads, d = q.shape
    _, block_size, kv_heads, _ = k_pool.shape
    if heads % kv_heads:
        raise ValueError(f"{heads} query heads do not group onto {kv_heads} KV heads")
    if q.stride(-1) != 1 or k_pool.stride(-1) != 1 or v_pool.stride(-1) != 1:
        raise ValueError("last dimensions must be contiguous")
    group = heads // kv_heads
    gp = max(16, triton.next_power_of_2(group))
    dp = max(16, triton.next_power_of_2(d))
    max_keys = block_tables.shape[1] * block_size
    if splits is None:
        splits = choose_splits(bsz, kv_heads, max_keys, q.device)
    chunk = triton.cdiv(triton.cdiv(max_keys, splits), block_n) * block_n
    out = torch.empty(bsz, heads, d, device=q.device, dtype=q.dtype)
    if splits > 1:
        rows = bsz * kv_heads * splits * gp
        acc = torch.empty(rows, dp, device=q.device, dtype=torch.float32)
        m = torch.empty(rows, device=q.device, dtype=torch.float32)
        lsum = torch.empty(rows, device=q.device, dtype=torch.float32)
    else:
        acc = m = lsum = out  # unused by the single-split kernel
    _decode_kernel[(bsz, kv_heads, splits)](
        q, k_pool, v_pool, block_tables, seq_lens, out, acc, m, lsum,
        q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        1.0 / math.sqrt(d),
        G=group, GP=gp, D=d, DP=dp, BS=block_size, BLOCK_N=block_n, CHUNK=chunk,
        SPLITS=splits, KVH=kv_heads, IEEE=k_pool.dtype == torch.float32,
        num_warps=4, num_stages=2,
    )  # fmt: skip
    if splits > 1:
        _merge_kernel[(bsz, kv_heads)](
            out, acc, m, lsum, out.stride(0), out.stride(1),
            G=group, GP=gp, D=d, DP=dp, SPLITS=splits, KVH=kv_heads, num_warps=4,
        )  # fmt: skip
    return out
