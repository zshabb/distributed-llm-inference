# ruff: noqa: N803, E501, E741, B023  (audit prototype kept as run: Triton-style names, long lines)
"""Paged GQA decode attention: current path vs query reshaping vs a Triton paged kernel (M6 audit).

This is the prototype the audit measured; the production kernel is dli.model.paged_attention.

Isolated attention for one decode step of ``B`` sequences over paged KV laid out exactly like
``PagedKVStore`` (``[layers, blocks, block_size, kv_heads, head_dim]``, scattered block tables),
Qwen2.5-0.5B shapes by default (14 query heads, 2 KV heads, head_dim 64, 24 layers). Each variant
is captured in a CUDA graph of 24 calls (one per layer's pool) so the timing is GPU time, not
launch overhead. Every variant is checked against a float64 reference first.

Variants:
* ``current``: gather the pages, SDPA with a key mask and ``enable_gqa`` (the M6 engine path).
* ``reshape``: same gather; queries reshaped to ``[B, kv_heads, group, d]`` so each KV head is
  attended by its whole query group in one SDPA "query block" (no GQA expansion).
* ``triton-sN``: a GQA-aware paged decode kernel reading K/V straight from the pool through the
  block table (no gather, no padding beyond each sequence's length), ``N`` splits of the sequence
  (split-K / flash-decoding) with an in-kernel or a second-kernel reduction.

The KV write of the new token is the same in every variant and is not timed.
Usage: uv run --frozen python benchmarks/scripts/decode_attention_bench.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def _paged_gqa_decode(
    Q,
    K,
    V,
    TABLES,
    LENS,
    OUT,
    ACC,
    MS,
    LS,
    stride_qb,
    stride_qh,
    stride_kblk,
    stride_ktok,
    stride_kh,
    stride_tb,
    stride_ob,
    stride_oh,
    sm_scale,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    SPLITS: tl.constexpr,
    KVH: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    s = tl.program_id(2)
    seq_len = tl.load(LENS + b)
    start = s * CHUNK
    end = tl.minimum(start + CHUNK, seq_len)
    offs_g = tl.arange(0, GP)
    offs_d = tl.arange(0, D)
    gmask = offs_g < G
    q = tl.load(
        Q + b * stride_qb + (kvh * G + offs_g)[:, None] * stride_qh + offs_d[None, :],
        mask=gmask[:, None],
        other=0.0,
    )
    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < end
        page = tl.load(TABLES + b * stride_tb + offs_n // BS, mask=nmask, other=0)
        kv_off = (
            page[:, None] * stride_kblk
            + (offs_n % BS)[:, None] * stride_ktok
            + kvh * stride_kh
            + offs_d[None, :]
        )
        k = tl.load(K + kv_off, mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(nmask[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + kv_off, mask=nmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    if SPLITS == 1:
        out = acc / l_i[:, None]
        tl.store(
            OUT + b * stride_ob + (kvh * G + offs_g)[:, None] * stride_oh + offs_d[None, :],
            out.to(OUT.dtype.element_ty),
            mask=gmask[:, None],
        )
    else:
        row = (b * KVH + kvh) * SPLITS + s
        tl.store(ACC + (row * GP + offs_g)[:, None] * D + offs_d[None, :], acc)
        tl.store(MS + row * GP + offs_g, m_i)
        tl.store(LS + row * GP + offs_g, l_i)


@triton.jit
def _reduce_splits(
    OUT,
    ACC,
    MS,
    LS,
    stride_ob,
    stride_oh,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    KVH: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    offs_s = tl.arange(0, SPLITS)
    offs_g = tl.arange(0, GP)
    offs_d = tl.arange(0, D)
    base = (b * KVH + kvh) * SPLITS
    m = tl.load(MS + (base + offs_s)[:, None] * GP + offs_g[None, :])  # [S, GP]
    l = tl.load(LS + (base + offs_s)[:, None] * GP + offs_g[None, :])
    m_max = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_max[None, :]))
    denom = tl.sum(l * w, 0)
    out = tl.zeros([GP, D], tl.float32)
    for i in tl.static_range(SPLITS):
        a = tl.load(ACC + ((base + i) * GP + offs_g)[:, None] * D + offs_d[None, :])
        mi = tl.load(MS + (base + i) * GP + offs_g)
        wi = tl.where(mi == float("-inf"), 0.0, tl.exp(mi - m_max))
        out += a * wi[:, None]
    out = out / denom[:, None]
    gmask = offs_g < G
    tl.store(
        OUT + b * stride_ob + (kvh * G + offs_g)[:, None] * stride_oh + offs_d[None, :],
        out.to(OUT.dtype.element_ty),
        mask=gmask[:, None],
    )


class Case:
    def __init__(
        self,
        bsz: int,
        ctx: int,
        width: int,
        layers: int,
        heads: int,
        kvh: int,
        d: int,
        bs: int,
        seed: int,
    ) -> None:
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.bsz, self.width, self.layers, self.heads, self.kvh, self.d, self.bs = (
            bsz,
            width,
            layers,
            heads,
            kvh,
            d,
            bs,
        )
        lo = max(1, int(ctx * 0.85))
        self.lens = torch.randint(lo, ctx + 1, (bsz,), device="cuda", generator=g)
        assert int(self.lens.max()) <= width * bs
        num_blocks = bsz * width + 8
        self.k = torch.randn(
            layers, num_blocks, bs, kvh, d, device="cuda", dtype=torch.bfloat16, generator=g
        )
        self.v = torch.randn(
            layers, num_blocks, bs, kvh, d, device="cuda", dtype=torch.bfloat16, generator=g
        )
        perm = torch.randperm(num_blocks, device="cuda", generator=g)[: bsz * width]
        self.tables = perm.view(bsz, width).contiguous()  # scattered physical blocks
        q = torch.randn(heads, bsz, d, device="cuda", dtype=torch.bfloat16, generator=g)
        self.q = q.transpose(0, 1)  # [B, H, d], the engine's non-contiguous layout
        keys = torch.arange(width * bs, device="cuda")
        self.mask = (keys[None, :] < self.lens[:, None])[:, None, None, :]
        self.out = torch.empty(bsz, heads, d, device="cuda", dtype=torch.bfloat16)
        self.scale = 1.0 / math.sqrt(d)

    # ------------------------------------------------------------------ variants
    def gather(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.k[layer][self.tables].view(self.bsz, -1, self.kvh, self.d).transpose(1, 2)
        v = self.v[layer][self.tables].view(self.bsz, -1, self.kvh, self.d).transpose(1, 2)
        return k, v

    def current(self, layer: int) -> torch.Tensor:
        k, v = self.gather(layer)
        out = F.scaled_dot_product_attention(
            self.q.unsqueeze(2), k, v, attn_mask=self.mask, enable_gqa=True
        )
        return out.squeeze(2)

    def reshape(self, layer: int) -> torch.Tensor:
        k, v = self.gather(layer)
        g = self.heads // self.kvh
        q = self.q.reshape(self.bsz, self.kvh, g, self.d)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)
        return out.reshape(self.bsz, self.heads, self.d)

    def gather_only(self, layer: int) -> torch.Tensor:
        k, _ = self.gather(layer)  # both gathers are recorded in the graph
        return k

    def triton(self, layer: int, splits: int, block_n: int = 64) -> torch.Tensor:
        g = self.heads // self.kvh
        gp = max(16, triton.next_power_of_2(g))
        maxlen = self.width * self.bs
        chunk = triton.cdiv(triton.cdiv(maxlen, splits), block_n) * block_n
        kl, vl = self.k[layer], self.v[layer]
        bufs = getattr(self, "_bufs", {})
        self._bufs = bufs
        if splits > 1 and splits not in bufs:  # split-K scratch, sized for this split count
            n = self.bsz * self.kvh * splits * gp
            bufs[splits] = (
                torch.empty(n, self.d, device="cuda", dtype=torch.float32),
                torch.empty(n, device="cuda", dtype=torch.float32),
                torch.empty(n, device="cuda", dtype=torch.float32),
            )
        acc, ms, ls = bufs.get(splits, (self.out, self.out, self.out))
        _paged_gqa_decode[(self.bsz, self.kvh, splits)](
            self.q,
            kl,
            vl,
            self.tables,
            self.lens,
            self.out,
            acc,
            ms,
            ls,
            self.q.stride(0),
            self.q.stride(1),
            kl.stride(0),
            kl.stride(1),
            kl.stride(2),
            self.tables.stride(0),
            self.out.stride(0),
            self.out.stride(1),
            self.scale,
            G=g,
            GP=gp,
            D=self.d,
            BS=self.bs,
            BLOCK_N=block_n,
            CHUNK=chunk,
            SPLITS=splits,
            KVH=self.kvh,
            num_warps=4,
            num_stages=2,
        )
        if splits > 1:
            _reduce_splits[(self.bsz, self.kvh)](
                self.out,
                acc,
                ms,
                ls,
                self.out.stride(0),
                self.out.stride(1),
                G=g,
                GP=gp,
                D=self.d,
                SPLITS=splits,
                KVH=self.kvh,
                num_warps=4,
            )
        return self.out

    def reference(self, layer: int) -> torch.Tensor:
        k, v = self.gather(layer)
        g = self.heads // self.kvh
        k = k.double().repeat_interleave(g, dim=1)
        v = v.double().repeat_interleave(g, dim=1)
        q = self.q.double().unsqueeze(2)
        s = (q @ k.transpose(-1, -2)) * self.scale
        s = s.masked_fill(~self.mask, float("-inf"))
        return (s.softmax(-1) @ v).squeeze(2)


def time_graph(fn, layers: int, reps: int = 30) -> float:
    """GPU ms for ``layers`` back-to-back calls, from a CUDA graph replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for layer in range(layers):
                fn(layer)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(layers):
            fn(layer)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=24)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    configs = [  # (batch, max context, block-table width): widths as the engine pads them
        (64, 383, 24),
        (16, 383, 24),
        (64, 1150, 72),
        (16, 1150, 72),
        (16, 2175, 136),
        (64, 2175, 136),
    ]
    rows = []
    for bsz, ctx, width in configs:
        c = Case(bsz, ctx, width, a.layers, 14, 2, 64, 16, seed=bsz * 1000 + ctx)
        variants = {"current": c.current, "reshape": c.reshape}
        for splits in (1, 4, 8, 16):
            variants[f"triton-s{splits}"] = (lambda sp: lambda layer: c.triton(layer, sp))(splits)
        ref = c.reference(3)
        row = {
            "batch": bsz,
            "max_context": ctx,
            "width_blocks": width,
            "mean_len": round(float(c.lens.float().mean()), 1),
            "kv_bytes_per_layer_live": int(c.lens.sum()) * 2 * 2 * 64 * 2,
            "kv_bytes_per_layer_padded": bsz * width * 16 * 2 * 2 * 64 * 2,
        }
        for name, fn in variants.items():
            err = float((fn(3).double() - ref).abs().max())
            ms = time_graph(fn, a.layers)
            row[name] = {"ms_per_step": round(ms, 4), "max_abs_err": err}
        row["gather_only"] = {"ms_per_step": round(time_graph(c.gather_only, a.layers), 4)}
        live_gb = row["kv_bytes_per_layer_live"] * a.layers / 1e9
        for name in variants:
            row[name]["effective_gbs_live_kv"] = round(
                live_gb / (row[name]["ms_per_step"] / 1e3), 1
            )
        rows.append(row)
        best = min(
            (n for n in variants if n.startswith("triton")), key=lambda n: row[n]["ms_per_step"]
        )
        print(
            f"B={bsz:3d} ctx<={ctx:5d} W={width:3d}: current {row['current']['ms_per_step']:.3f} ms, "
            f"reshape {row['reshape']['ms_per_step']:.3f}, gather-only {row['gather_only']['ms_per_step']:.3f}, "
            f"best triton {best} {row[best]['ms_per_step']:.3f} | errs cur {row['current']['max_abs_err']:.1e} "
            f"resh {row['reshape']['max_abs_err']:.1e} tri {row[best]['max_abs_err']:.1e}"
        )
    (a.out / "decode_attention_bench.json").write_text(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "layers": a.layers,
                "rows": rows,
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
