import torch
import torch.nn as nn

from inferlab.quant import (Int8WeightOnlyLinear, model_weight_bytes, quantize_model_int8,
                            quantize_per_channel_int8)


def test_quantize_range_and_dtype():
    w = torch.randn(64, 128)
    q, s = quantize_per_channel_int8(w)
    assert q.dtype == torch.int8
    assert q.abs().max() <= 127
    assert s.shape == (64,)


def test_roundtrip_error_is_bounded_by_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    q, s = quantize_per_channel_int8(w)
    recon = q.float() * s[:, None]
    # Rounding error per element is at most scale/2 for symmetric round-to-nearest.
    assert ((recon - w).abs() <= s[:, None] / 2 + 1e-6).all()


def test_int8_linear_matches_fp_linear():
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=True)
    qlin = Int8WeightOnlyLinear.from_linear(lin, torch.float32)
    x = torch.randn(4, 256)
    ref, out = lin(x), qlin(x)
    rel = (ref - out).norm() / ref.norm()
    assert rel < 0.02  # int8 per-channel on gaussian weights: well under 2% relative error


def test_zero_row_does_not_nan():
    w = torch.zeros(4, 8)
    q, s = quantize_per_channel_int8(w)
    assert torch.isfinite(s).all() and (q == 0).all()


def test_quantize_model_skips_lm_head_and_saves_memory():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
            self.lm_head = nn.Linear(512, 100, bias=False)

        def forward(self, x):
            return self.lm_head(self.block(x))

    m = Toy()
    before = model_weight_bytes(m)
    n = quantize_model_int8(m, compute_dtype=torch.float32)
    assert n == 2
    assert isinstance(m.lm_head, nn.Linear)  # skipped
    assert model_weight_bytes(m) < before * 0.5
    assert m(torch.randn(2, 512)).shape == (2, 100)
