"""Latency / throughput / memory benchmarking for autoregressive generation.

Methodology (the part reviewers care about):
  * Warm-up runs are discarded (kernel autotune, cudnn/cublas init, torch.compile).
  * Every timed region is bracketed by torch.cuda.synchronize(); CUDA is async, so
    timing without a sync measures kernel *launch*, not kernel *execution*.
  * Prefill and decode are measured separately, because they have opposite
    bottlenecks: prefill is compute-bound (big GEMMs), decode is memory-bound
    (one token at a time re-reads every weight).
        prefill_ms     = time(generate 1 new token)
        decode_ms/tok  = (time(generate N) - time(generate 1)) / (N - 1)
  * We report the median over repeats, plus min and stdev, never a single run.
"""

from __future__ import annotations

import statistics
import time

import torch


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def make_prompt(tokenizer, vocab_size: int, prompt_len: int, batch: int, text: str | None,
                device: str) -> torch.Tensor:
    if tokenizer is not None and text:
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        while ids.numel() < prompt_len:  # repeat if the sample text is short
            ids = torch.cat([ids, ids])
        ids = ids[:prompt_len]
    else:
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, vocab_size, (prompt_len,), generator=g)
    return ids.unsqueeze(0).repeat(batch, 1).to(device)


def _generate(model, input_ids, new_tokens: int, static_cache: bool):
    kwargs = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
                  eos_token_id=None, pad_token_id=0, use_cache=True)
    if static_cache:
        kwargs["cache_implementation"] = "static"
    with torch.inference_mode():
        return model.generate(input_ids, attention_mask=torch.ones_like(input_ids), **kwargs)


def _time(fn, device: str) -> float:
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0, out


def benchmark(model, input_ids: torch.Tensor, new_tokens: int, static_cache: bool, device: str,
              warmup: int = 3, repeats: int = 5, weight_bytes: int | None = None,
              peak_bw_gbps: float | None = None) -> dict:
    # Warm-up covers both shapes we later time (1 token and N tokens).
    for _ in range(warmup):
        _generate(model, input_ids, 1, static_cache)
        _generate(model, input_ids, new_tokens, static_cache)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    prefill, full, last = [], [], None
    for _ in range(repeats):
        t1, _ = _time(lambda: _generate(model, input_ids, 1, static_cache), device)
        tn, last = _time(lambda: _generate(model, input_ids, new_tokens, static_cache), device)
        prefill.append(t1)
        full.append(tn)

    decode_per_tok = [(tn - t1) / (new_tokens - 1) for t1, tn in zip(prefill, full)]
    batch = input_ids.shape[0]
    med_decode = statistics.median(decode_per_tok)
    tokens_per_s = batch * 1000.0 / med_decode

    res = {
        "prefill_ms_median": round(statistics.median(prefill), 2),
        "decode_ms_per_token_median": round(med_decode, 3),
        "decode_ms_per_token_min": round(min(decode_per_tok), 3),
        "decode_ms_per_token_stdev": round(statistics.pstdev(decode_per_tok), 3),
        "decode_tokens_per_s": round(tokens_per_s, 1),
        "prompt_len": input_ids.shape[1],
        "new_tokens": new_tokens,
        "batch": batch,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None,
        "_generated": last[:, input_ids.shape[1]:].cpu(),
    }
    # Model-bandwidth utilization: bytes of weights streamed per second / peak DRAM bandwidth.
    # Each decode step reads every weight once, independent of batch size (small batch).
    if weight_bytes:
        achieved_gbps = weight_bytes * (1000.0 / med_decode) / 1e9
        res["achieved_bandwidth_gbps"] = round(achieved_gbps, 1)
        if peak_bw_gbps:
            res["mbu_pct"] = round(100.0 * achieved_gbps / peak_bw_gbps, 1)
            res["decode_roofline_tokens_per_s"] = round(batch * peak_bw_gbps * 1e9 / weight_bytes, 1)
    return res
