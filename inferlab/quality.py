"""Quality checks so speedups are not bought with silent accuracy loss."""

from __future__ import annotations

import math
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data" / "sample.txt"


def load_text(path: str | None = None) -> str:
    return Path(path or DATA).read_text()


@torch.inference_mode()
def perplexity(model, token_ids: torch.Tensor, window: int, device: str) -> float:
    """Non-overlapping-window perplexity. token_ids: 1-D LongTensor."""
    nll, count = 0.0, 0
    for start in range(0, token_ids.numel() - 1, window):
        chunk = token_ids[start:start + window + 1]
        if chunk.numel() < 2:
            break
        x = chunk[:-1].unsqueeze(0).to(device)
        y = chunk[1:].unsqueeze(0).to(device)
        logits = model(x).logits.float()  # fp32 softmax for a stable loss
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                                 reduction="sum")
        nll += loss.item()
        count += y.numel()
    return math.exp(nll / count)


def ppl_tokens(tokenizer, vocab_size: int, text: str, min_tokens: int = 2048) -> torch.Tensor:
    if tokenizer is None:  # tiny-random smoke-test path
        g = torch.Generator().manual_seed(1)
        return torch.randint(0, vocab_size, (min_tokens,), generator=g)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    while ids.numel() < min_tokens:
        ids = torch.cat([ids, ids])
    return ids


def greedy_agreement(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Fraction of greedy-decoded tokens identical to the reference variant."""
    n = min(reference.numel(), candidate.numel())
    return float((reference.flatten()[:n] == candidate.flatten()[:n]).float().mean())
