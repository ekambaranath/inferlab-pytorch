"""Correctness of the Triton kernel. On machines without a GPU it runs in Triton's CPU
interpreter (TRITON_INTERPRET=1), so the logic, masking and epilogue are verified anywhere.
The env var must be set before inferlab.kernels is imported."""

import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

pytest.importorskip("triton")

from inferlab.kernels import int8_gemv  # noqa: E402
from inferlab.quant import Int8WeightOnlyLinear, quantize_per_channel_int8  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ref(x, q, scale, bias=None):
    y = x.float() @ (q.float() * scale.float()[:, None]).T
    return y + bias.float() if bias is not None else y


@pytest.mark.parametrize("N,K,M", [(37, 200, 1), (70, 300, 3), (16, 64, 2)])  # non-multiples test masking
def test_kernel_matches_reference(N, K, M):
    torch.manual_seed(0)
    q, s = quantize_per_channel_int8(torch.randn(N, K))
    q, s = q.to(DEV), s.to(DEV)
    x = torch.randn(M, K, device=DEV, dtype=torch.float16)
    out = int8_gemv(x, q, s.half())
    ref = _ref(x, q, s)
    assert out.shape == (M, N)
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 2e-3  # fp16 output rounding only


def test_kernel_bias_and_leading_dims():
    torch.manual_seed(1)
    N, K = 40, 130
    q, s = quantize_per_channel_int8(torch.randn(N, K))
    b = torch.randn(N, device=DEV, dtype=torch.float16)
    x = torch.randn(2, 3, K, device=DEV, dtype=torch.float16)
    out = int8_gemv(x, q.to(DEV), s.to(DEV).half(), b)
    assert out.shape == (2, 3, N)
    ref = _ref(x.reshape(-1, K), q.to(DEV), s.to(DEV), b).reshape(2, 3, N)
    assert ((out.float() - ref).norm() / ref.norm()) < 2e-3


def test_layer_fused_equals_unfused():
    torch.manual_seed(2)
    lin = torch.nn.Linear(192, 80, bias=True).to(DEV).half()
    layer = Int8WeightOnlyLinear.from_linear(lin, torch.float16)
    layer = layer.to(DEV)
    x = torch.randn(1, 192, device=DEV, dtype=torch.float16)
    unfused = layer(x)
    layer.fused = True
    fused = layer(x)
    assert ((fused.float() - unfused.float()).norm() / unfused.float().norm()) < 5e-3
