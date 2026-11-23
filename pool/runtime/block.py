"""Reference language-tower block on plain lists.

This is the reference formula the Metal and CUDA adapters follow: RMSNorm,
QKV, RoPE, grouped-query attention, residual, SwiGLU. It is the layer-range
path. It is not a chat completion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LayerWeights:
    attn_norm: list
    q: list
    k: list
    v: list
    o: list
    mlp_norm: list
    gate: list
    up: list
    down: list
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_theta: float
    rms_eps: float

    @property
    def kind(self) -> str:
        if _is_mlx(self.q):
            return "mlx"
        if _is_torch(self.q):
            return "torch"
        return "list"


def _is_torch(value) -> bool:
    module = type(value).__module__
    return module.startswith("torch")


def _is_mlx(value) -> bool:
    module = type(value).__module__
    return module.startswith("mlx")


def rmsnorm(x: list[float], weight: list[float], eps: float) -> list[float]:
    mean_square = sum(v * v for v in x) / len(x)
    scale = (mean_square + eps) ** -0.5
    return [v * scale * weight[i] for i, v in enumerate(x)]


def matvec(weight: list[list[float]], x: list[float]) -> list[float]:
    if weight and len(weight[0]) != len(x):
        raise ValueError(f"linear expected width {len(weight[0])}, got {len(x)}")
    return [sum(w * xi for w, xi in zip(row, x)) for row in weight]


def silu(values: list[float]) -> list[float]:
    return [v / (1.0 + math.exp(-v)) for v in values]


def _split_heads(vector: list[float], n_heads: int, head_dim: int) -> list[list[float]]:
    return [vector[h * head_dim : (h + 1) * head_dim] for h in range(n_heads)]


def rope(heads: list[list[float]], position: int, theta: float) -> list[list[float]]:
    rotated = []
    for head in heads:
        out = [0.0] * len(head)
        for index in range(0, len(head), 2):
            angle = position * (theta ** (-index / len(head)))
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            a = head[index]
            b = head[index + 1]
            out[index] = a * cos_a - b * sin_a
            out[index + 1] = b * cos_a + a * sin_a
        rotated.append(out)
    return rotated


def _softmax(scores: list[float]) -> list[float]:
    peak = max(scores)
    exps = [math.exp(score - peak) for score in scores]
    total = sum(exps)
    return [item / total for item in exps]


def _attend(
    query_heads: list[list[float]],
    keys: list[list[list[float]]],
    values: list[list[list[float]]],
    n_kv_heads: int,
) -> list[float]:
    """keys/values are [past_and_present, kv_heads, head_dim], already causal."""
    n_heads = len(query_heads)
    head_dim = len(query_heads[0])
    group = n_heads // n_kv_heads
    scale = head_dim ** -0.5
    pieces: list[float] = []
    for head_index, query in enumerate(query_heads):
        kv_index = head_index // group
        scores = []
        for step in keys:
            key = step[kv_index]
            scores.append(sum(a * b for a, b in zip(query, key)) * scale)
        weights = _softmax(scores)
        mixed = [0.0] * head_dim
        for weight, step in zip(weights, values):
            value = step[kv_index]
            for dim in range(head_dim):
                mixed[dim] += weight * value[dim]
        pieces.extend(mixed)
    return pieces


def forward_layers_list(
    layers: list[LayerWeights],
    hidden: list[list[float]],
    past_keys: list[list[list[list[float]]]] | None,
    past_values: list[list[list[list[float]]]] | None,
    position_start: int,
) -> tuple[list[list[float]], list, list]:
    """Run a contiguous layer range.

    past_* is [n_layers, past_seq, kv_heads, head_dim] or None.
    Returned KV is only the new tokens, same layout.
    """
    rows = [list(row) for row in hidden]
    new_keys: list = []
    new_values: list = []
    for layer_index, layer in enumerate(layers):
        past_k = [] if not past_keys else [list(step) for step in past_keys[layer_index]]
        past_v = [] if not past_values else [list(step) for step in past_values[layer_index]]
        layer_new_k: list = []
        layer_new_v: list = []
        updated: list[list[float]] = []
        for seq_index, row in enumerate(rows):
            normed = rmsnorm(row, layer.attn_norm, layer.rms_eps)
            query = rope(
                _split_heads(matvec(layer.q, normed), layer.n_heads, layer.head_dim),
                position_start + seq_index,
                layer.rope_theta,
            )
            key = rope(
                _split_heads(matvec(layer.k, normed), layer.n_kv_heads, layer.head_dim),
                position_start + seq_index,
                layer.rope_theta,
            )
            value = _split_heads(matvec(layer.v, normed), layer.n_kv_heads, layer.head_dim)
            layer_new_k.append(key)
            layer_new_v.append(value)
            visible_k = past_k + layer_new_k
            visible_v = past_v + layer_new_v
            mixed = _attend(query, visible_k, visible_v, layer.n_kv_heads)
            attn_out = matvec(layer.o, mixed)
            residual = [a + b for a, b in zip(row, attn_out)]
            mlp_in = rmsnorm(residual, layer.mlp_norm, layer.rms_eps)
            gated = silu(matvec(layer.gate, mlp_in))
            up = matvec(layer.up, mlp_in)
            mlp_out = matvec(layer.down, [a * b for a, b in zip(gated, up)])
            updated.append([a + b for a, b in zip(residual, mlp_out)])
        rows = updated
        new_keys.append(layer_new_k)
        new_values.append(layer_new_v)
    return rows, new_keys, new_values


def apply_lm_head_list(
    hidden: list[list[float]],
    norm_weight: list[float],
    lm_weight: list[list[float]],
    eps: float,
) -> list[float]:
    """Project the last token. `norm_weight` is the final RMSNorm, not a decoder layer."""
    last = rmsnorm(hidden[-1], norm_weight, eps)
    return matvec(lm_weight, last)
