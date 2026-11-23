from pool.errors import EngineUnavailable
from pool.protocol.constants import ModelSpec
from pool.runtime.block import LayerWeights, apply_lm_head_list, forward_layers_list
from pool.runtime.extract import drop_vision_module, export_head, export_layer, find_layer_list


def _spec() -> ModelSpec:
    return ModelSpec(
        name="tiny",
        num_layers=2,
        hidden_size=4,
        n_heads=2,
        n_kv_heads=1,
        head_dim=2,
        intermediate=4,
        vocab_size=5,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        eos_token_id=0,
    )


def _matrix(rows: int, cols: int, scale: float) -> list[list[float]]:
    return [[scale * ((row + 1) * 0.1 + (col + 1) * 0.01) for col in range(cols)] for row in range(rows)]


def _layer(spec: ModelSpec, scale: float = 0.2) -> LayerWeights:
    hidden = spec.hidden_size
    query = spec.n_heads * spec.head_dim
    key = spec.n_kv_heads * spec.head_dim
    return LayerWeights(
        attn_norm=[1.0] * hidden,
        q=_matrix(query, hidden, scale),
        k=_matrix(key, hidden, scale),
        v=_matrix(key, hidden, scale / 2),
        o=_matrix(hidden, query, scale),
        mlp_norm=[0.8] * hidden,
        gate=_matrix(spec.intermediate, hidden, scale),
        up=_matrix(spec.intermediate, hidden, scale / 3),
        down=_matrix(hidden, spec.intermediate, scale),
        n_heads=spec.n_heads,
        n_kv_heads=spec.n_kv_heads,
        head_dim=spec.head_dim,
        rope_theta=spec.rope_theta,
        rms_eps=spec.rms_norm_eps,
    )


def _close(left, right) -> bool:
    return all(abs(a - b) < 1e-6 for a, b in zip(left, right))


def test_prefill_matches_cached_decode():
    spec = _spec()
    layer = _layer(spec)
    hidden = [[0.2, -0.1, 0.3, 0.05], [0.4, 0.1, -0.2, 0.2]]
    full, _, _ = forward_layers_list([layer], hidden, None, None, 0)
    first, keys, values = forward_layers_list([layer], [hidden[0]], None, None, 0)
    second, _, _ = forward_layers_list([layer], [hidden[1]], keys, values, 1)
    assert _close(full[0], first[0])
    assert _close(full[1], second[0])


def test_two_layers_change_the_hidden_state():
    spec = _spec()
    hidden = [[0.2, -0.1, 0.3, 0.05]]
    one, _, _ = forward_layers_list([_layer(spec)], hidden, None, None, 0)
    two, _, _ = forward_layers_list([_layer(spec), _layer(spec, 0.15)], hidden, None, None, 0)
    assert not _close(one[0], two[0])


def test_extract_unfused_and_fused_qkv():
    spec = _spec()
    layer = _layer(spec)

    class Linear:
        def __init__(self, weight):
            self.weight = weight

    class Norm:
        def __init__(self, weight):
            self.weight = weight

    class Attn:
        def __init__(self):
            self.q_proj = Linear(layer.q)
            self.k_proj = Linear(layer.k)
            self.v_proj = Linear(layer.v)
            self.o_proj = Linear(layer.o)

    class Mlp:
        def __init__(self):
            self.gate_proj = Linear(layer.gate)
            self.up_proj = Linear(layer.up)
            self.down_proj = Linear(layer.down)

    class Block:
        def __init__(self):
            self.input_layernorm = Norm(layer.attn_norm)
            self.self_attn = Attn()
            self.post_attention_layernorm = Norm(layer.mlp_norm)
            self.mlp = Mlp()

    exported = export_layer(Block(), spec)
    assert exported.q[0] == layer.q[0]
    assert len(exported.k) == spec.n_kv_heads * spec.head_dim

    class FusedAttn:
        def __init__(self):
            self.qkv_proj = Linear(layer.q + layer.k + layer.v)
            self.o_proj = Linear(layer.o)

    class FusedBlock(Block):
        def __init__(self):
            super().__init__()
            self.self_attn = FusedAttn()

    fused = export_layer(FusedBlock(), spec)
    assert fused.q == layer.q
    assert fused.v == layer.v

    class Root:
        def __init__(self):
            self.model = type("Model", (), {"layers": [Block(), Block()]})()

    found = find_layer_list(Root())
    assert len(found) == 2


def test_language_tower_is_not_the_vision_encoder():
    class Block:
        def __init__(self, tag: str):
            self.tag = tag
            self.self_attn = tag
            self.mlp = tag

    class Language:
        def __init__(self):
            self.layers = [Block("language") for _ in range(36)]
            self.embed_tokens = [[float(token)] * 4 for token in range(5)]
            self.norm = [1.0] * 4

    class Vision:
        def __init__(self):
            self.blocks = [Block("vision") for _ in range(27)]
            self.embed_tokens = [[9.0] * 4]

    class Model:
        def __init__(self):
            self.visual = Vision()
            self.language_model = Language()

    class Root:
        def __init__(self):
            self.model = Model()
            self.lm_head = [[0.1] * 4 for _ in range(5)]

    root = Root()
    found = find_layer_list(root, expected_len=36)
    assert len(found) == 36
    assert all(block.tag == "language" for block in found)
    assert all(block.tag == "vision" for block in root.model.visual.blocks)

    class VisionOnly:
        def __init__(self):
            self.visual = Vision()

    try:
        find_layer_list(VisionOnly(), expected_len=36)
    except EngineUnavailable:
        pass
    else:
        raise AssertionError("vision blocks were accepted as the language tower")

    spec = ModelSpec(
        name="tiny-vl",
        num_layers=36,
        hidden_size=4,
        n_heads=2,
        n_kv_heads=1,
        head_dim=2,
        intermediate=4,
        vocab_size=5,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        eos_token_id=0,
    )
    embed, norm, head = export_head(root, spec)
    assert embed[0][0] == 0.0
    assert norm == [1.0, 1.0, 1.0, 1.0]
    assert len(head) == 5
    assert drop_vision_module(root) is True
    assert root.model.visual is None
    assert len(root.model.language_model.layers) == 36


def test_lm_head_uses_last_row_only():
    spec = _spec()
    hidden = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.0, -0.2, 0.1]]
    norm = [1.0] * spec.hidden_size
    weight = _matrix(spec.vocab_size, spec.hidden_size, 0.3)
    logits = apply_lm_head_list(hidden, norm, weight, spec.rms_norm_eps)
    assert len(logits) == spec.vocab_size
    again = apply_lm_head_list([hidden[1]], norm, weight, spec.rms_norm_eps)
    assert all(abs(a - b) < 1e-6 for a, b in zip(logits, again))
