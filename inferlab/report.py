"""Turn raw results into a markdown table and charts."""

from __future__ import annotations

from pathlib import Path


def _fmt(v, nd=1):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def markdown_table(results: list[dict], baseline: str) -> str:
    base = next((r for r in results if r["variant"] == baseline and "error" not in r), None)
    head = ("| variant | weights (GB) | prefill (ms) | decode (ms/tok) | tok/s | speedup | "
            "peak mem (GB) | MBU % | perplexity | greedy match |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for r in results:
        if "error" in r:
            rows.append(f"| {r['variant']} | FAILED: {r['error'][:60]} |||||||||")
            continue
        sp = None
        if base:
            sp = base["decode_ms_per_token_median"] / r["decode_ms_per_token_median"]
        rows.append(
            f"| {r['variant']} | {_fmt(r['weights_gb'], 2)} | {_fmt(r['prefill_ms_median'])} | "
            f"{_fmt(r['decode_ms_per_token_median'], 2)} | {_fmt(r['decode_tokens_per_s'])} | "
            f"{_fmt(sp, 2) + 'x' if sp else '-'} | {_fmt(r.get('peak_memory_gb'), 2)} | "
            f"{_fmt(r.get('mbu_pct'))} | {_fmt(r.get('perplexity'), 2)} | "
            f"{_fmt(r.get('greedy_match'), 3)} |")
    return head + "\n".join(rows) + "\n"


def plot(results: list[dict], out_dir: str, title: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    ok = [r for r in results if "error" not in r]
    if not ok:
        return None
    names = [r["variant"].split("_", 1)[0] for r in ok]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    series = [("decode_tokens_per_s", "Decode throughput (tokens/s)"),
              ("weights_gb", "Weight memory (GB)"),
              ("perplexity", "Perplexity (lower = better)")]
    for ax, (key, label) in zip(axes, series):
        vals = [r.get(key) or 0 for r in ok]
        ax.bar(names, vals, color="#4C72B0")
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=0)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = str(Path(out_dir) / "summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
