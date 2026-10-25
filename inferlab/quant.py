"""Weight-only int8 quantization, written from scratch in plain PyTorch.

Scheme: symmetric, per-output-channel. For a weight W of shape [out, in]:

    scale[o] = max(|W[o, :]|) / 127
    Wq[o, i] = round(W[o, i] / scale[o])        (int8)

At inference the int8 weight is dequantized on the fly and multiplied against
fp16/bf16 activations. Weights are stored in 1 byte instead of 2 (fp16) or 4
(fp32), which matters because LLM decode is memory-bandwidth bound: every
generated token has to stream all weights from DRAM once.

Honest caveat (also reported in the README): in eager PyTorch the dequantize
step is a separate kernel that materializes a full fp16 weight, so this layer
saves *memory* but may not save *time*. A fused dequant-GEMV kernel (Triton /
CUDA / vendor library) is what turns the byte savings into speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (int8 weight, fp32 per-row scale) for a [out, in] weight."""
    w = weight.detach().float()
    max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(1)


class Int8WeightOnlyLinear(nn.Module):
    """Drop-in replacement for nn.Linear storing int8 weights."""

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None,
                 compute_dtype: torch.dtype):
        super().__init__()
        self.out_features, self.in_features = qweight.shape
        self.compute_dtype = compute_dtype
        self.fused = False  # set True to use the Triton fused dequant-GEMV for small batches
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale.to(compute_dtype))
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(compute_dtype))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, compute_dtype: torch.dtype | None = None) -> "Int8WeightOnlyLinear":
        compute_dtype = compute_dtype or linear.weight.dtype
        q, scale = quantize_per_channel_int8(linear.weight)
        return cls(q, scale, linear.bias, compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused and x.numel() // self.in_features <= _max_fused_m():
            from .kernels import int8_gemv
            return int8_gemv(x, self.qweight, self.scale.to(x.dtype),
                             None if self.bias is None else self.bias.to(x.dtype))
        # Dequantize the full weight BEFORE the matmul: w = q * scale.
        # (Do not matmul with raw int8 values and scale the output afterwards: raw
        # values are up to 127 and are ~1/scale times larger than the real weights, so
        # in fp16 the products overflow to inf/NaN on models with large activations.
        # That was a real bug here, found on Qwen2.5 on a T4; see tests/test_quant.py.)
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)[:, None]
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return torch.nn.functional.linear(x, w, bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8, per-channel"


def _max_fused_m() -> int:
    from .kernels import MAX_FUSED_M
    return MAX_FUSED_M


def quantize_model_int8(model: nn.Module, skip: tuple[str, ...] = ("lm_head",),
                        compute_dtype: torch.dtype | None = None, fused: bool = False) -> int:
    """Replace nn.Linear modules in-place. Returns number of layers replaced.

    lm_head is skipped by default: it is the layer most sensitive to
    quantization error (it directly produces the logits) and is often tied to
    the embedding matrix.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                q = Int8WeightOnlyLinear.from_linear(child, compute_dtype)
                q.fused = fused
                setattr(module, child_name, q)
                replaced += 1
    return replaced


def model_weight_bytes(model: nn.Module) -> int:
    """Bytes of parameters + buffers, counting shared (tied) storage once."""
    seen: set[int] = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        key = t.untyped_storage().data_ptr()
        if key in seen:
            continue
        seen.add(key)
        total += t.untyped_storage().nbytes()
    return total
