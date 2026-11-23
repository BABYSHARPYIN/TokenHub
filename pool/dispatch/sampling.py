"""Greedy sampling on Mac. Logits are not sent to Windows."""

from __future__ import annotations

from pool.protocol.codec import TensorBlob


def greedy(logits: TensorBlob) -> int:
    flat = logits.flat()
    if not logits.shape:
        raise ValueError("logits are empty")
    vocab = logits.shape[-1]
    if vocab <= 0 or len(flat) < vocab:
        raise ValueError("logits do not cover a vocabulary row")
    row = flat[-vocab:]
    best = 0
    best_value = row[0]
    for index, value in enumerate(row):
        if value > best_value:
            best = index
            best_value = value
    return best
