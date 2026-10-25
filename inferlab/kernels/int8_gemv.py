"""Fused int8 weight-only GEMV in Triton:  y = (x @ Wq^T) * scale (+ bias)

Why this kernel exists
----------------------
LLM decode multiplies a tiny activation (1..few rows) by a huge weight matrix, so
the cost is the time to *read the weights from DRAM*. int8 weights are half the
bytes of fp16, so the ceiling is ~2x faster. The plain-PyTorch version

    w = qweight.to(fp16)          # kernel 1: reads N*K bytes, WRITES 2*N*K bytes
    y = F.linear(x, w)            # kernel 2: reads 2*N*K bytes again

moves ~5*N*K bytes, which is MORE traffic than fp16 (2*N*K). This kernel reads
each int8 weight once, converts it in registers, accumulates in fp32, and applies
the per-row scale at the end: ~1*N*K bytes of weight traffic.

Design
------
* Grid = (cdiv(N, BLOCK_N), M). Each program owns BLOCK_N output rows for one
  activation row and streams along K, so weight reads are contiguous (coalesced).
* fp32 accumulation; scale/bias applied once in the epilogue.
* Intended for small M (decode). For large M (prefill) the matmul is compute-bound
  and cuBLAS tensor cores win, so callers should fall back (see MAX_FUSED_M).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

MAX_FUSED_M = 16  # above this many activation rows, use the dense path

_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"


@triton.jit
def _int8_gemv_kernel(
    x_ptr, w_ptr, s_ptr, b_ptr, y_ptr,
    N, K,
    stride_xm, stride_wn, stride_ym,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(x_ptr + m * stride_xm + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    scale = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc * scale
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + m * stride_ym + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


_CONFIGS = [
    triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw)
    for bn in (4, 8, 16, 32)
    for bk in (256, 512, 1024)
    for nw in (2, 4, 8)
]

# Autotune on real GPUs (cached per (N, K)); the interpreter does not need it.
_kernel = (_int8_gemv_kernel if _INTERPRET
           else triton.autotune(configs=_CONFIGS, key=["N", "K"])(_int8_gemv_kernel))


def int8_gemv(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
    """x: [..., K] fp16/bf16/fp32, qweight: [N, K] int8, scale: [N], bias: [N] or None."""
    assert qweight.dtype == torch.int8 and qweight.is_contiguous()
    N, K = qweight.shape
    assert x.shape[-1] == K
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    kwargs = {}
    if _INTERPRET:
        kwargs = dict(BLOCK_N=16, BLOCK_K=64)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]) if "BLOCK_N" in meta else triton.cdiv(N, 16), M)
    _kernel[grid](
        x2, qweight, scale, bias if bias is not None else scale, y,
        N, K,
        x2.stride(0), qweight.stride(0), y.stride(0),
        HAS_BIAS=bias is not None,
        **kwargs,
    )
    return y.reshape(*lead, N)
