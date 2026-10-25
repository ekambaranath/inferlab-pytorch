"""torch.profiler wrapper: where does the time go, per phase?

Produces (a) a Chrome/Perfetto trace you can open at ui.perfetto.dev and
(b) a compact table of the top kernels by self device time, split into
prefill and decode so the two regimes can be compared side by side.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

from .bench import _generate


def _activities(device: str):
    acts = [ProfilerActivity.CPU]
    if device == "cuda":
        acts.append(ProfilerActivity.CUDA)  # also covers ROCm/HIP builds
    return acts


def _top_ops(prof, device: str, k: int) -> list[dict]:
    key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    rows = []
    for ev in prof.key_averages():
        # Newer torch renamed *_cuda_* -> *_device_*; support both.
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, key, 0)
        if device != "cuda":
            t = ev.self_cpu_time_total
        rows.append({"op": ev.key[:80], "calls": ev.count, "self_time_us": round(float(t), 1)})
    rows.sort(key=lambda r: r["self_time_us"], reverse=True)
    total = sum(r["self_time_us"] for r in rows) or 1.0
    top = rows[:k]
    for r in top:
        r["pct"] = round(100.0 * r["self_time_us"] / total, 1)
    return top


def profile_phases(model, input_ids: torch.Tensor, static_cache: bool, device: str, out_dir: str,
                   tag: str, decode_tokens: int = 16, top_k: int = 10) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for _ in range(2):  # warm
        _generate(model, input_ids, decode_tokens, static_cache)

    results = {}
    for phase, n in (("prefill", 1), ("decode", decode_tokens)):
        with profile(activities=_activities(device), record_shapes=True) as prof:
            _generate(model, input_ids, n, static_cache)
            if device == "cuda":
                torch.cuda.synchronize()
        trace = out / f"{tag}_{phase}.trace.json"
        prof.export_chrome_trace(str(trace))
        results[phase] = {"trace": str(trace), "top_ops": _top_ops(prof, device, top_k)}
    return results
