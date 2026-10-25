# inferlab-pytorch: LLM inference optimization lab

A small, reproducible PyTorch study of **why LLM decoding is slow and what actually makes it faster**.
It takes an open LLM and climbs an optimization ladder, measuring every rung for latency, throughput,
memory, achieved memory bandwidth and output quality.

```
v0  fp32, eager attention            baseline
v1  fp16, eager attention            half precision
v2  fp16, SDPA (fused attention)     fused attention kernel
v3  fp16, SDPA, int8 weights         weight-only int8 (written from scratch, see inferlab/quant.py)
v4  fp16, SDPA, torch.compile        static KV cache + compiled forward (CUDA graphs)
v5  fp16, SDPA, int8, compile        everything combined
v6  fp16, SDPA, int8 + Triton kernel fused int8 dequant-GEMV (inferlab/kernels/int8_gemv.py)
```

## What makes this more than a benchmark script

- **Prefill and decode are measured separately.** Prefill is compute-bound, decode is memory-bound.
- **Model Bandwidth Utilization (MBU).** Each decode step streams all weights once, so
  `MBU = bytes_of_weights x tokens/s / peak_DRAM_bandwidth`. This tells you how close a variant is to the
  hardware roofline instead of just "it got faster".
- **Quality guard.** Every variant reports perplexity and greedy-token agreement with the baseline, so a
  speedup is never bought with silent accuracy loss.
- **Profiler traces.** `--profile v2,v5` exports Perfetto traces and a top-kernel table for prefill and decode.
- **A custom Triton kernel that fixes a real problem.** Plain-PyTorch int8 dequantizes in a separate kernel,
  which writes a full fp16 weight back to DRAM and re-reads it: ~5 bytes of traffic per weight versus 2 for fp16,
  so it can be *slower* than fp16 despite smaller weights. `inferlab/kernels/int8_gemv.py` reads each int8 weight
  once, converts in registers, accumulates in fp32 and applies the scale in the epilogue (~1 byte per weight).
  Ceiling: ~2x over fp16 for the weight-bound part of decode.
- **Backend agnostic.** Uses only `torch.cuda` APIs, which map to HIP on AMD ROCm builds of PyTorch. The same
  command runs on an AMD Instinct/Radeon GPU, and `--peak-bw` lets you supply that part's bandwidth.

## Run it (Google Colab, T4)

```bash
pip install -r requirements.txt
python -m inferlab.cli --model Qwen/Qwen2.5-0.5B --prompt-len 256 --new-tokens 64 --profile v2,v5
```

T4 notes: it has no native bf16, so the default is fp16. Stay at <= 1.5B parameters for the fp32 baseline
(`Qwen/Qwen2.5-1.5B` needs ~6 GB for v0). Outputs go to `results/`: `results.json`, `results.md`, `summary.png`,
and `traces/*.json` (open at https://ui.perfetto.dev).

Kernel-level microbenchmark (the cleanest evidence for the kernel claim; GPU required):

```bash
python -m inferlab.kernel_bench      # writes results/kernel_bench.md and .json
```

It compares fp16 cuBLAS, unfused int8 and the fused kernel on matrix shapes from real models (Qwen 0.5B/1.5B,
Llama-7B class) at M=1 and M=4, reporting time, effective GB/s, % of peak bandwidth and error vs fp32.

Offline smoke test (no downloads, runs on CPU): `python -m inferlab.cli --model tiny-random`
Tests: `pytest -q`

## Verification status

- Kernel logic (masking for non-multiple shapes, bias, batch rows, fp32 accumulate) is unit-tested against a
  reference in Triton's CPU interpreter: `pytest -q`.
- GPU performance numbers are **not** included until you run `kernel_bench` and the CLI on a GPU.

## Results

> Fill this in from your own run. Do not publish numbers you did not measure.

| variant | weights (GB) | prefill (ms) | decode (ms/tok) | tok/s | speedup | peak mem (GB) | MBU % | perplexity | greedy match |
|---|---|---|---|---|---|---|---|---|---|
| _paste results/results.md here_ | | | | | | | | | |

## Findings (write these after your run)

Answer each with a number from your table or a trace:

1. How much of v0 -> v2 speedup came from dtype vs fused attention?
2. Is decode memory-bound on your GPU? (compare MBU to 100% and to prefill)
3. Did int8 reduce latency in eager mode? Why or why not? (check the dequant kernels in the decode trace)
4. What did `torch.compile` + static cache remove? (kernel launch count / CPU gaps in the trace)
5. What perplexity / agreement cost did int8 add?

## Roadmap (clearly marked as not yet done)

- Int4 (packed) variant of the kernel, and group-wise scales for better accuracy.
- Fuse the kernel into a full decode step under `torch.compile` / CUDA graphs.
- Run the identical benchmark on an AMD GPU (ROCm) and publish both tables.
- Export the quantized model via ExecuTorch / ONNX and measure on-device-class hardware.

## Layout

```
inferlab/kernels/      Triton fused int8 dequant-GEMV (autotuned; runs in Triton's CPU interpreter for tests)
inferlab/kernel_bench.py  per-op microbenchmark: fp16 vs unfused int8 vs fused int8
inferlab/quant.py      int8 per-channel weight-only quantization + Int8WeightOnlyLinear
inferlab/variants.py   the optimization ladder and model loading
inferlab/bench.py      sync-correct prefill/decode timing, memory, MBU
inferlab/profile.py    torch.profiler traces + top-kernel tables per phase
inferlab/quality.py    perplexity and greedy-agreement checks
inferlab/hardware.py   device info and peak-bandwidth table (NVIDIA + AMD)
inferlab/report.py     markdown table + chart
tests/                 unit tests for quantization + offline end-to-end smoke test
```
