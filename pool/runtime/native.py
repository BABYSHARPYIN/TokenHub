"""Torch and MLX translations of the reference Qwen2 block.

Weights stay on the device that loaded them. Activations cross the pool boundary
as float32 blobs. These functions never call a chat or generate entrypoint.
"""

from __future__ import annotations

from pool.runtime.block import LayerWeights


def _torch():
    import torch

    return torch


def forward_layers_torch(layers: list[LayerWeights], hidden_rows, past_keys, past_values, position_start: int):
    torch = _torch()
    device = layers[0].q.device
    dtype = layers[0].q.dtype
    hidden = torch.tensor(hidden_rows, dtype=dtype, device=device)
    new_keys = []
    new_values = []
    for index, layer in enumerate(layers):
        past_k = _past_tensor(torch, past_keys, index, device, dtype, layer)
        past_v = _past_tensor(torch, past_values, index, device, dtype, layer)
        seq = hidden.shape[0]
        layer_k = []
        layer_v = []
        outputs = []
        for seq_index in range(seq):
            row = hidden[seq_index]
            normed = _rmsnorm_torch(torch, row, layer.attn_norm, layer.rms_eps)
            position = position_start + seq_index
            query = _rope_torch(torch, _heads(torch, _linear(torch, layer.q, normed), layer.n_heads, layer.head_dim), position, layer.rope_theta)
            key = _rope_torch(torch, _heads(torch, _linear(torch, layer.k, normed), layer.n_kv_heads, layer.head_dim), position, layer.rope_theta)
            value = _heads(torch, _linear(torch, layer.v, normed), layer.n_kv_heads, layer.head_dim)
            layer_k.append(key)
            layer_v.append(value)
            full_k = torch.cat([past_k, torch.stack(layer_k, dim=0)], dim=0) if past_k.numel() else torch.stack(layer_k, dim=0)
            full_v = torch.cat([past_v, torch.stack(layer_v, dim=0)], dim=0) if past_v.numel() else torch.stack(layer_v, dim=0)
            mixed = _attend_torch(torch, query, full_k, full_v, layer.n_kv_heads)
            residual = row + _linear(torch, layer.o, mixed)
            mlp_in = _rmsnorm_torch(torch, residual, layer.mlp_norm, layer.rms_eps)
            gated = torch.nn.functional.silu(_linear(torch, layer.gate, mlp_in))
            up = _linear(torch, layer.up, mlp_in)
            residual = residual + _linear(torch, layer.down, gated * up)
            outputs.append(residual)
        hidden = torch.stack(outputs, dim=0)
        new_keys.append(torch.stack(layer_k, dim=0).to(dtype=torch.float32).detach().cpu().tolist())
        new_values.append(torch.stack(layer_v, dim=0).to(dtype=torch.float32).detach().cpu().tolist())
    return hidden.to(dtype=torch.float32).detach().cpu().tolist(), new_keys, new_values


def apply_lm_head_torch(hidden_rows, norm_weight, lm_weight, eps: float):
    torch = _torch()
    device = lm_weight.device
    dtype = lm_weight.dtype
    last = torch.tensor(hidden_rows[-1], dtype=dtype, device=device)
    normed = _rmsnorm_torch(torch, last, norm_weight, eps)
    logits = _linear(torch, lm_weight, normed)
    return logits.to(dtype=torch.float32).detach().cpu().tolist()


def _linear(torch, weight, vector):
    return torch.nn.functional.linear(vector, weight.to(dtype=vector.dtype))


def _rmsnorm_torch(torch, vector, weight, eps: float):
    weight = weight if hasattr(weight, "to") else torch.tensor(weight, dtype=vector.dtype, device=vector.device)
    variance = vector.float().pow(2).mean()
    scale = torch.rsqrt(variance + eps)
    return (vector.float() * scale).to(dtype=weight.dtype) * weight


def _heads(torch, vector, n_heads: int, head_dim: int):
    return vector.view(n_heads, head_dim)


def _rope_torch(torch, heads, position: int, theta: float):
    head_dim = heads.shape[-1]
    half = torch.arange(0, head_dim, 2, device=heads.device, dtype=torch.float32)
    inv = torch.pow(torch.tensor(theta, device=heads.device), -half / head_dim)
    angles = position * inv
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    a = heads[..., 0::2].float()
    b = heads[..., 1::2].float()
    out = torch.empty_like(heads, dtype=torch.float32)
    out[..., 0::2] = a * cos_a - b * sin_a
    out[..., 1::2] = b * cos_a + a * sin_a
    return out.to(dtype=heads.dtype)


def _attend_torch(torch, query, keys, values, n_kv_heads: int):
    # query [heads, dim], keys [seq, kv_heads, dim]
    n_heads = query.shape[0]
    group = n_heads // n_kv_heads
    scale = query.shape[-1] ** -0.5
    pieces = []
    for head_index in range(n_heads):
        kv_index = head_index // group
        scores = torch.einsum("d,sd->s", query[head_index].float(), keys[:, kv_index].float()) * scale
        weights = torch.softmax(scores, dim=0)
        mixed = torch.einsum("s,sd->d", weights, values[:, kv_index].float())
        pieces.append(mixed.to(dtype=query.dtype))
    return torch.cat(pieces, dim=0)


def _past_tensor(torch, past, index: int, device, dtype, layer: LayerWeights):
    if not past:
        return torch.zeros((0, layer.n_kv_heads, layer.head_dim), dtype=dtype, device=device)
    return torch.tensor(past[index], dtype=dtype, device=device)


def forward_layers_mlx(layers: list[LayerWeights], hidden_rows, past_keys, past_values, position_start: int):
    import mlx.core as mx

    hidden = mx.array(hidden_rows, dtype=mx.float32)
    new_keys = []
    new_values = []
    for index, layer in enumerate(layers):
        past_k = _past_mlx(mx, past_keys, index, layer)
        past_v = _past_mlx(mx, past_values, index, layer)
        seq = int(hidden.shape[0])
        layer_k = []
        layer_v = []
        outputs = []
        for seq_index in range(seq):
            row = hidden[seq_index]
            normed = _rmsnorm_mlx(mx, row, layer.attn_norm, layer.rms_eps)
            position = position_start + seq_index
            query = _rope_mlx(mx, _heads_mlx(mx, _linear_mlx(mx, layer.q, normed), layer.n_heads, layer.head_dim), position, layer.rope_theta)
            key = _rope_mlx(mx, _heads_mlx(mx, _linear_mlx(mx, layer.k, normed), layer.n_kv_heads, layer.head_dim), position, layer.rope_theta)
            value = _heads_mlx(mx, _linear_mlx(mx, layer.v, normed), layer.n_kv_heads, layer.head_dim)
            layer_k.append(key)
            layer_v.append(value)
            full_k = mx.concatenate([past_k, mx.stack(layer_k, axis=0)], axis=0) if past_k.shape[0] else mx.stack(layer_k, axis=0)
            full_v = mx.concatenate([past_v, mx.stack(layer_v, axis=0)], axis=0) if past_v.shape[0] else mx.stack(layer_v, axis=0)
            mixed = _attend_mlx(mx, query, full_k, full_v, layer.n_kv_heads)
            residual = row + _linear_mlx(mx, layer.o, mixed)
            mlp_in = _rmsnorm_mlx(mx, residual, layer.mlp_norm, layer.rms_eps)
            gated = _silu_mlx(mx, _linear_mlx(mx, layer.gate, mlp_in))
            up = _linear_mlx(mx, layer.up, mlp_in)
            residual = residual + _linear_mlx(mx, layer.down, gated * up)
            outputs.append(residual)
        hidden = mx.stack(outputs, axis=0)
        mx.eval(hidden)
        new_keys.append(_mlx_kv_list(mx, layer_k))
        new_values.append(_mlx_kv_list(mx, layer_v))
    mx.eval(hidden)
    return hidden.tolist(), new_keys, new_values


def apply_lm_head_mlx(hidden_rows, norm_weight, lm_weight, eps: float):
    import mlx.core as mx

    last = mx.array(hidden_rows[-1], dtype=mx.float32)
    normed = _rmsnorm_mlx(mx, last, norm_weight, eps)
    logits = _linear_mlx(mx, lm_weight, normed)
    mx.eval(logits)
    return logits.tolist()


def _coerce_mlx(mx, weight):
    if type(weight).__module__.startswith("mlx"):
        return weight.astype(mx.float32)
    return mx.array(weight, dtype=mx.float32)


def _linear_mlx(mx, weight, vector):
    matrix = _coerce_mlx(mx, weight)
    return matrix @ vector


def _rmsnorm_mlx(mx, vector, weight, eps: float):
    norm = _coerce_mlx(mx, weight)
    variance = mx.mean(mx.square(vector))
    return vector * mx.rsqrt(variance + eps) * norm


def _silu_mlx(mx, vector):
    return vector * mx.sigmoid(vector)


def _heads_mlx(mx, vector, n_heads: int, head_dim: int):
    del mx
    return vector.reshape((n_heads, head_dim))


def _rope_mlx(mx, heads, position: int, theta: float):
    head_dim = int(heads.shape[-1])
    half = mx.arange(0, head_dim, 2, dtype=mx.float32)
    inv = mx.power(mx.array(theta, dtype=mx.float32), -half / head_dim)
    angles = position * inv
    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)
    a = heads[..., 0::2]
    b = heads[..., 1::2]
    out_even = a * cos_a - b * sin_a
    out_odd = b * cos_a + a * sin_a
    return mx.stack([out_even, out_odd], axis=-1).reshape(heads.shape)


def _attend_mlx(mx, query, keys, values, n_kv_heads: int):
    n_heads = int(query.shape[0])
    group = n_heads // n_kv_heads
    scale = int(query.shape[-1]) ** -0.5
    pieces = []
    for head_index in range(n_heads):
        kv_index = head_index // group
        scores = (keys[:, kv_index] @ query[head_index]) * scale
        weights = mx.softmax(scores, axis=0)
        mixed = weights @ values[:, kv_index]
        pieces.append(mixed)
    return mx.concatenate(pieces, axis=0)


def _past_mlx(mx, past, index: int, layer: LayerWeights):
    if not past:
        return mx.zeros((0, layer.n_kv_heads, layer.head_dim), dtype=mx.float32)
    return mx.array(past[index], dtype=mx.float32)


def _mlx_kv_list(mx, steps):
    stacked = mx.stack(steps, axis=0)
    mx.eval(stacked)
    return stacked.tolist()
