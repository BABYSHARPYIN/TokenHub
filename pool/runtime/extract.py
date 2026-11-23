"""Pull Qwen2 decoder weights out of a vLLM or MLX module tree.

The pool then runs a layer range itself. It does not call the engine chat API.
"""

from __future__ import annotations

from pool.errors import EngineUnavailable
from pool.protocol.constants import ModelSpec
from pool.runtime.block import LayerWeights


def _weight(module):
    if module is None:
        return None
    if hasattr(module, "weight"):
        return module.weight
    return module


def _child(module, *names):
    for name in names:
        if hasattr(module, name):
            found = getattr(module, name)
            if found is not None:
                return found
    return None


def _native(weight) -> bool:
    module = type(weight).__module__
    return module.startswith("torch") or module.startswith("mlx")


def _as_rows(weight, expected_out: int | None = None):
    if weight is None:
        raise EngineUnavailable("missing weight while exporting a decoder layer")
    if _native(weight):
        if expected_out is not None and int(weight.shape[0]) != expected_out:
            raise EngineUnavailable(f"expected {expected_out} rows, found {int(weight.shape[0])}")
        return weight
    if isinstance(weight, list):
        rows = weight
    elif hasattr(weight, "detach") and hasattr(weight, "tolist"):
        rows = weight.detach().to("cpu").float().tolist()
    else:
        raise EngineUnavailable(f"unsupported weight type {type(weight)!r}")
    if expected_out is not None and len(rows) != expected_out:
        raise EngineUnavailable(f"expected {expected_out} rows, found {len(rows)}")
    return rows


def _split_rows(weight, sizes: list[int]):
    if isinstance(weight, list):
        cursor = 0
        parts = []
        for size in sizes:
            parts.append(weight[cursor : cursor + size])
            cursor += size
        if cursor != len(weight):
            raise EngineUnavailable("fused weight does not match the Qwen2 head sizes")
        return parts
    if hasattr(weight, "split") and not type(weight).__module__.startswith("mlx"):
        return list(weight.split(sizes, dim=0))
    if type(weight).__module__.startswith("mlx"):
        return _mlx_split(weight, sizes)
    raise EngineUnavailable("cannot split a fused projection")


def _mlx_split(weight, sizes: list[int]):
    import mlx.core as mx

    parts = []
    cursor = 0
    for size in sizes:
        parts.append(weight[cursor : cursor + size])
        cursor += size
    del mx
    if cursor != int(weight.shape[0]):
        raise EngineUnavailable("fused MLX weight does not match the Qwen2 head sizes")
    return parts


def _vector(weight, expected: int):
    if weight is not None and _native(weight):
        if int(weight.shape[0]) != expected:
            raise EngineUnavailable(f"norm width {int(weight.shape[0])} != {expected}")
        return weight
    rows = _as_rows(weight)
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        if len(rows) == 1:
            rows = rows[0]
        elif len(rows) == expected and len(rows[0]) == 1:
            rows = [row[0] for row in rows]
    if isinstance(rows, list) and len(rows) != expected:
        raise EngineUnavailable(f"norm width {len(rows)} != {expected}")
    return rows


def looks_like_block(module) -> bool:
    return any(
        hasattr(module, name)
        for name in ("self_attn", "attention", "mlp", "input_layernorm", "attention_norm")
    )


def find_layer_list(root, depth: int = 6):
    """Bounded walk used after a vLLM engine has loaded. Does not call generate."""
    seen: set[int] = set()

    def walk(obj, level: int):
        if obj is None or level < 0 or id(obj) in seen:
            return None
        if isinstance(obj, (str, bytes, int, float, bool)):
            return None
        seen.add(id(obj))
        for name in ("layers", "h", "decoder_layers"):
            seq = getattr(obj, name, None)
            if isinstance(seq, (list, tuple)) and seq and looks_like_block(seq[0]):
                return list(seq)
        for name in (
            "model",
            "llm_engine",
            "engine_core",
            "model_executor",
            "driver_worker",
            "worker",
            "model_runner",
        ):
            if hasattr(obj, name):
                found = walk(getattr(obj, name), level - 1)
                if found is not None:
                    return found
        return None

    found = walk(root, depth)
    if found is None:
        raise EngineUnavailable(
            "decoder layers were not found inside the engine; refusing to fall back to chat"
        )
    return found


def export_layer(module, spec: ModelSpec) -> LayerWeights:
    attn = _child(module, "self_attn", "attention")
    mlp = _child(module, "mlp")
    if attn is None or mlp is None:
        raise EngineUnavailable("decoder block has no attention or mlp module")
    q_size = spec.n_heads * spec.head_dim
    kv_size = spec.n_kv_heads * spec.head_dim
    q_proj = _child(attn, "q_proj", "query_proj")
    k_proj = _child(attn, "k_proj", "key_proj")
    v_proj = _child(attn, "v_proj", "value_proj")
    if q_proj is None:
        fused = _weight(_child(attn, "qkv_proj", "wqkv"))
        if fused is None:
            raise EngineUnavailable("attention has neither q_proj nor qkv_proj")
        q, k, v = _split_rows(fused, [q_size, kv_size, kv_size])
    else:
        q = _as_rows(_weight(q_proj), q_size)
        k = _as_rows(_weight(k_proj), kv_size)
        v = _as_rows(_weight(v_proj), kv_size)
    o = _as_rows(_weight(_child(attn, "o_proj", "out_proj")), spec.hidden_size)
    gate = _child(mlp, "gate_proj")
    up = _child(mlp, "up_proj")
    if gate is None:
        fused_mlp = _weight(_child(mlp, "gate_up_proj"))
        if fused_mlp is None:
            raise EngineUnavailable("mlp has neither gate_proj nor gate_up_proj")
        gate_w, up_w = _split_rows(fused_mlp, [spec.intermediate, spec.intermediate])
    else:
        gate_w = _as_rows(_weight(gate), spec.intermediate)
        up_w = _as_rows(_weight(up), spec.intermediate)
    down = _as_rows(_weight(_child(mlp, "down_proj")), spec.hidden_size)
    attn_norm = _vector(
        _weight(_child(module, "input_layernorm", "attention_norm")),
        spec.hidden_size,
    )
    mlp_norm = _vector(
        _weight(_child(module, "post_attention_layernorm", "mlp_norm")),
        spec.hidden_size,
    )
    return LayerWeights(
        attn_norm=attn_norm,
        q=q,
        k=k,
        v=v,
        o=o,
        mlp_norm=mlp_norm,
        gate=gate_w,
        up=up_w,
        down=down,
        n_heads=spec.n_heads,
        n_kv_heads=spec.n_kv_heads,
        head_dim=spec.head_dim,
        rope_theta=spec.rope_theta,
        rms_eps=spec.rms_norm_eps,
    )


def export_range(root, start: int, end: int, spec: ModelSpec) -> list[LayerWeights]:
    layers = find_layer_list(root)
    if end > len(layers) or start < 0:
        raise EngineUnavailable(
            f"model exposes {len(layers)} layers; cannot export [{start}, {end})"
        )
    return [export_layer(layers[index], spec) for index in range(start, end)]


def export_head(root, spec: ModelSpec):
    """Return (embed, final_norm, lm_head) from a causal-LM module. Mac only."""
    model = getattr(root, "model", root)
    embed = _weight(_child(root, "embed_tokens") or _child(model, "embed_tokens"))
    norm = _weight(_child(model, "norm") or _child(root, "norm"))
    head = _weight(_child(root, "lm_head") or _child(model, "lm_head"))
    if embed is None or norm is None or head is None:
        raise EngineUnavailable("Mac runner requires embed_tokens, final norm, and lm_head")
    embed_rows = _as_rows(embed, spec.vocab_size)
    norm_vec = _vector(norm, spec.hidden_size)
    # Some checkpoints tie the head. A Linear exposes .weight; embed may already be rows.
    head_rows = _as_rows(head, spec.vocab_size)
    return embed_rows, norm_vec, head_rows
