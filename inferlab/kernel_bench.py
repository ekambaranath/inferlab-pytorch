"""Kernel-level microbenchmark: fused int8 GEMV vs fp16 cuBLAS vs unfused int8.

    python -m inferlab.kernel_bench            # LLM-shaped matrices, M = 1 and 4
    python -m inferlab.kernel_bench --peak-bw 320

Why microbenchmark in addition to end-to-end runs: whole-model tokens/s mixes in
attention, sampling, Python overhead and launch gaps. A single-op benchmark isolates
exactly what the kernel changed, which is the cleanest evidence for a kernel claim.

Reported per shape:
  * time per call (median of many iterations, CUDA-synchronized)
  * effective bandwidth = bytes of weights that MUST be read / time
  * speedup vs fp16 and vs the unfused PyTorch int8 path
  * max relative error vs an fp32 reference (correctness guard)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from . import hardware
from .kernels import int8_gemv
from .quant import quantize_per_channel_int8

# (name, N=out_features, K=in_features) taken from real model configs.
SHAPES = [
    ("qwen0.5B q_proj / o_proj", 896, 896),
    ("qwen0.5B mlp up/gate", 4864, 896),
    ("qwen0.5B mlp down", 896, 4864),
    ("qwen1.5B mlp up/gate", 8960, 1536),
    ("llama7B-class qkv/o", 4096, 4096),
    ("llama7B-class mlp up", 11008, 4096),
]


def _bench(fn, iters: int = 200, warmup: int = 20) -> float:
    """Median milliseconds per call using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def run(ms: list[int], peak_bw: float | None, dtype: torch.dtype) -> list[dict]:
    dev = "cuda"
    rows = []
    for name, N, K in SHAPES:
        w32 = torch.randn(N, K, device=dev)
        q, scale = quantize_per_channel_int8(w32)
        q, scale16 = q.contiguous(), scale.to(dtype)
        w16 = w32.to(dtype)
        for M in ms:
            x = torch.randn(M, K, device=dev, dtype=dtype)
            ref = F.linear(x.float(), w32)

            def fp16():
                return F.linear(x, w16)

            def unfused():
                return F.linear(x, q.to(dtype)) * scale16

            def fused():
                return int8_gemv(x, q, scale16)

            err = {k: ((f().float() - ref).norm() / ref.norm()).item()
                   for k, f in (("fp16", fp16), ("unfused_int8", unfused), ("fused_int8", fused))}
            t = {"fp16": _bench(fp16), "unfused_int8": _bench(unfused), "fused_int8": _bench(fused)}
            nbytes = {"fp16": 2 * N * K, "unfused_int8": N * K, "fused_int8": N * K}
            row = {"shape": name, "N": N, "K": K, "M": M}
            for k in t:
                gbps = nbytes[k] / (t[k] * 1e-3) / 1e9
                row[f"{k}_ms"] = round(t[k], 4)
                row[f"{k}_eff_gbps"] = round(gbps, 1)
                row[f"{k}_rel_err"] = round(err[k], 5)
                if peak_bw:
                    row[f"{k}_pct_peak"] = round(100 * gbps / peak_bw, 1)
            row["fused_vs_fp16"] = round(t["fp16"] / t["fused_int8"], 2)
            row["fused_vs_unfused"] = round(t["unfused_int8"] / t["fused_int8"], 2)
            rows.append(row)
            print(f"{name:28s} M={M}  fp16 {t['fp16']:.4f}ms  unfused {t['unfused_int8']:.4f}ms  "
                  f"fused {t['fused_int8']:.4f}ms  -> {row['fused_vs_fp16']}x vs fp16, "
                  f"{row['fused_vs_unfused']}x vs unfused  (err {err['fused_int8']:.4f})")
    return rows


def to_markdown(rows: list[dict]) -> str:
    h = ("| shape | N x K | M | fp16 (ms) | unfused int8 (ms) | fused int8 (ms) | fused vs fp16 | "
         "fused vs unfused | fused GB/s | % peak | rel err |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = "\n".join(
        f"| {r['shape']} | {r['N']}x{r['K']} | {r['M']} | {r['fp16_ms']} | {r['unfused_int8_ms']} | "
        f"{r['fused_int8_ms']} | {r['fused_vs_fp16']}x | {r['fused_vs_unfused']}x | "
        f"{r['fused_int8_eff_gbps']} | {r.get('fused_int8_pct_peak', '-')} | {r['fused_int8_rel_err']} |"
        for r in rows)
    return h + body + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m", default="1,4", help="comma list of activation rows (decode batch sizes)")
    p.add_argument("--peak-bw", type=float, default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("kernel_bench needs a GPU (CUDA or ROCm). Use pytest for CPU correctness checks.")
    info = hardware.device_info()
    peak = a.peak_bw or hardware.lookup_peak_bw(info["name"])
    print(info, "peak_bw", peak)
    dtype = torch.bfloat16 if a.bf16 else torch.float16
    rows = run([int(x) for x in a.m.split(",")], peak, dtype)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel_bench.json").write_text(json.dumps({"device": info, "peak_bw": peak, "rows": rows}, indent=2))
    (out / "kernel_bench.md").write_text(to_markdown(rows))
    print("\n" + to_markdown(rows))


if __name__ == "__main__":
    main()
