"""Optimization variants and model loading.

Each Variant is one cumulative step in the optimization ladder. Keeping them as
data (not scattered if-statements) makes the benchmark table reproducible and
makes it trivial to add a new rung.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaConfig, LlamaForCausalLM

from .quant import quantize_model_int8

TINY = "tiny-random"  # offline smoke-test model, no downloads
COMPILE_MODE = "reduce-overhead"  # set by the CLI (--compile-mode)


@dataclass(frozen=True)
class Variant:
    name: str
    dtype: str = "float16"      # float32 | float16 | bfloat16
    attn: str = "sdpa"          # eager | sdpa
    int8: bool = False
    compile: bool = False
    static_cache: bool = False
    fused: bool = False         # Triton fused int8 dequant-GEMV (decode path)

    def describe(self) -> dict:
        return asdict(self)


def ladder(use_bf16: bool = False) -> list[Variant]:
    half = "bfloat16" if use_bf16 else "float16"
    return [
        Variant("v0_fp32_eager_attn", dtype="float32", attn="eager"),
        Variant("v1_half_eager_attn", dtype=half, attn="eager"),
        Variant("v2_half_sdpa", dtype=half, attn="sdpa"),
        Variant("v3_half_sdpa_int8", dtype=half, attn="sdpa", int8=True),
        Variant("v4_half_sdpa_compile", dtype=half, attn="sdpa", compile=True, static_cache=True),
        Variant("v5_half_sdpa_int8_compile", dtype=half, attn="sdpa", int8=True, compile=True,
                static_cache=True),
        Variant("v6_half_sdpa_int8_fused", dtype=half, attn="sdpa", int8=True, fused=True),
    ]


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _build_tiny(dtype: torch.dtype, attn: str):
    cfg = LlamaConfig(vocab_size=1024, hidden_size=128, intermediate_size=352, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    cfg._attn_implementation = attn
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    return model.to(dtype)


def load_model(model_id: str, variant: Variant, device: str):
    """Return (model, tokenizer_or_None). Applies dtype, attention impl and int8."""
    dtype = _dtype(variant.dtype)
    if model_id == TINY:
        model = _build_tiny(dtype, variant.attn)
        tok = None
    else:
        kwargs = dict(attn_implementation=variant.attn)
        try:  # transformers >= 4.56 renamed torch_dtype -> dtype
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
        tok = AutoTokenizer.from_pretrained(model_id)
    model.eval().to(device)

    if variant.int8:
        n = quantize_model_int8(model, compute_dtype=dtype, fused=variant.fused)
        model.to(device)
        print(f"[{variant.name}] quantized {n} Linear layers to int8")

    if variant.compile:
        # HF-documented recipe: compile the forward pass, use a static KV cache so
        # shapes do not change between decode steps (avoids recompiles / enables CUDA graphs).
        # mode="reduce-overhead" uses CUDA graphs; if it misbehaves on your stack try
        # --compile-mode default.
        model.forward = torch.compile(model.forward, mode=COMPILE_MODE, fullgraph=False)
    return model, tok


def model_vocab_size(model) -> int:
    return model.get_input_embeddings().num_embeddings


def n_params(model) -> int:
    cfg = model.config
    # Count each tensor once even if tied.
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total
