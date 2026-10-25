"""End-to-end smoke test on an offline random tiny Llama (no downloads, CPU ok)."""

import torch

from inferlab import hardware
from inferlab.bench import benchmark, make_prompt
from inferlab.quality import greedy_agreement, perplexity, ppl_tokens
from inferlab.quant import model_weight_bytes
from inferlab.variants import TINY, Variant, load_model, model_vocab_size


def test_int8_shrinks_weights_and_pipeline_runs():
    fp = Variant("fp32", dtype="float32", attn="eager")
    q8 = Variant("int8", dtype="float32", attn="sdpa", int8=True)
    m_fp, _ = load_model(TINY, fp, "cpu")
    m_q8, _ = load_model(TINY, q8, "cpu")
    assert model_weight_bytes(m_q8) < model_weight_bytes(m_fp)

    ids = make_prompt(None, model_vocab_size(m_fp), 16, 1, None, "cpu")
    res = benchmark(m_fp, ids, new_tokens=8, static_cache=False, device="cpu", warmup=1, repeats=2,
                    weight_bytes=model_weight_bytes(m_fp), peak_bw_gbps=50.0)
    assert res["decode_tokens_per_s"] > 0
    assert res["_generated"].shape == (1, 8)

    toks = ppl_tokens(None, model_vocab_size(m_fp), "", min_tokens=256)
    assert perplexity(m_fp, toks, 64, "cpu") > 1.0
    assert greedy_agreement(res["_generated"], res["_generated"]) == 1.0


def test_hardware_info_has_device():
    assert hardware.device_info()["device"] in ("cpu", "cuda")
