"""Command line entry point:  python -m inferlab.cli --model Qwen/Qwen2.5-0.5B"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from . import hardware
from . import variants as variants_mod
from .bench import benchmark, make_prompt
from .profile import profile_phases
from .quality import greedy_agreement, load_text, perplexity, ppl_tokens
from .quant import model_weight_bytes
from .report import markdown_table, plot
from .variants import TINY, ladder, load_model, model_vocab_size


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM inference optimization lab")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                   help=f"HF model id, or '{TINY}' for an offline smoke test")
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--ppl-window", type=int, default=512)
    p.add_argument("--variants", default="all", help="comma list of variant-name prefixes, e.g. v0,v2,v5")
    p.add_argument("--peak-bw", type=float, default=None, help="peak DRAM GB/s (auto for known GPUs)")
    p.add_argument("--profile", default="", help="comma list of variant prefixes to profile, e.g. v2,v5")
    p.add_argument("--bf16", action="store_true", help="use bfloat16 instead of float16 (needs Ampere+/ROCm)")
    p.add_argument("--compile-mode", default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--out", default="results")
    return p.parse_args()


def main() -> None:
    a = parse()
    variants_mod.COMPILE_MODE = a.compile_mode
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = hardware.device_info()
    device = info["device"]
    peak_bw = a.peak_bw or hardware.lookup_peak_bw(info["name"])
    print(f"device: {info}  peak_bw={peak_bw}")

    variants = ladder(use_bf16=a.bf16)
    if a.variants != "all":
        wanted = tuple(a.variants.split(","))
        variants = [v for v in variants if v.name.startswith(wanted)]
    if device == "cpu":  # CPU has no fp16 matmul speed; keep the smoke test fast and valid
        variants = [v for v in variants if not v.compile]
        variants = [v for v in variants if not v.fused]
        print("CPU detected: skipping torch.compile and Triton-fused variants")
    to_profile = tuple(x for x in a.profile.split(",") if x)

    text = load_text()
    results, reference_tokens = [], None
    for v in variants:
        print(f"\n=== {v.name} ===")
        rec = {"variant": v.name, **v.describe()}
        try:
            model, tok = load_model(a.model, v, device)
            vocab = model_vocab_size(model)
            wbytes = model_weight_bytes(model)
            ids = make_prompt(tok, vocab, a.prompt_len, a.batch, text, device)
            rec["weights_gb"] = round(wbytes / 1e9, 3)

            rec.update(benchmark(model, ids, a.new_tokens, v.static_cache, device, a.warmup, a.repeats,
                                 wbytes, peak_bw))
            gen = rec.pop("_generated")
            if reference_tokens is None:
                reference_tokens = gen
            rec["greedy_match"] = round(greedy_agreement(reference_tokens, gen), 4)

            toks = ppl_tokens(tok, vocab, text)
            rec["perplexity"] = round(perplexity(model, toks, a.ppl_window, device), 3)

            if to_profile and v.name.startswith(to_profile):
                rec["profile"] = profile_phases(model, ids, v.static_cache, device, str(out / "traces"),
                                                v.name)
            print({k: x for k, x in rec.items() if k != "profile"})
        except Exception as e:  # keep going: one failing rung must not kill the whole run
            rec["error"] = f"{type(e).__name__}: {e}"
            print("FAILED:", rec["error"])
        results.append(rec)
        model = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    payload = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "model": a.model, "device": info,
               "peak_bandwidth_gbps": peak_bw, "torch": torch.__version__,
               "config": {"prompt_len": a.prompt_len, "new_tokens": a.new_tokens, "batch": a.batch,
                          "repeats": a.repeats}, "results": results}
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    baseline = variants[0].name
    md = markdown_table(results, baseline)
    (out / "results.md").write_text(md)
    chart = plot(results, str(out), f"{a.model} on {info['name']}")
    print("\n" + md)
    print(f"wrote {out}/results.json, results.md" + (f", {chart}" if chart else ""))


if __name__ == "__main__":
    main()
