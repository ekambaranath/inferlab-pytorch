"""Peak memory-bandwidth lookup used by the roofline / MBU analysis.

Values are vendor-published peak DRAM bandwidth in GB/s. They are *peaks*; real
achievable bandwidth is typically 70-90% of these. Override with --peak-bw on
the CLI when your device is not listed (e.g. an AMD or Qualcomm part).
"""

from __future__ import annotations

import torch

# Substring of torch.cuda.get_device_name() -> peak GB/s
PEAK_BW_GBPS = {
    "T4": 320.0,
    "L4": 300.0,
    "A10": 600.0,
    "A100": 1555.0,      # 40GB; the 80GB part is ~2039
    "H100": 3350.0,      # SXM
    "RTX 4090": 1008.0,
    "RTX 3090": 936.0,
    "MI210": 1638.0,     # AMD Instinct, for ROCm users
    "MI250": 3277.0,
    "MI300X": 5300.0,
}


def device_info() -> dict:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        backend = "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory
        return {"device": "cuda", "name": name, "backend": backend,
                "compute_capability": f"{cap[0]}.{cap[1]}", "total_memory_gb": round(total / 1e9, 2)}
    return {"device": "cpu", "name": "CPU", "backend": "cpu", "compute_capability": None,
            "total_memory_gb": None}


def lookup_peak_bw(name: str) -> float | None:
    for key, bw in PEAK_BW_GBPS.items():
        if key.lower() in name.lower():
            return bw
    return None


def supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
