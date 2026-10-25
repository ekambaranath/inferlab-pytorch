| variant | weights (GB) | prefill (ms) | decode (ms/tok) | tok/s | speedup | peak mem (GB) | MBU % | perplexity | greedy match |
|---|---|---|---|---|---|---|---|---|---|
| v0_fp32_eager_attn | 1.98 | 52.7 | 29.48 | 33.9 | 1.00x | 2.02 | 20.9 | 16.19 | 1.000 |
| v2_half_sdpa | 0.99 | 37.2 | 32.22 | 31.0 | 0.91x | 1.02 | 9.6 | 16.19 | 1.000 |
| v3_half_sdpa_int8 | 0.63 | 62.6 | 52.59 | 19.0 | 0.56x | 0.67 | 3.7 | 16.18 | 1.000 |
| v6_half_sdpa_int8_fused | 0.63 | 63.7 | 58.38 | 17.1 | 0.50x | 0.67 | 3.4 | 16.18 | 1.000 |
