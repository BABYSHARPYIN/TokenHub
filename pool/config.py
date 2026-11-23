"""Example-config checks. v1 rejects other models, uneven splits, and encryption."""

from __future__ import annotations

import json
from pathlib import Path

from pool.protocol.constants import (
    CHECKPOINT_FORMAT,
    EOS_TOKEN_ID,
    HEAD_DIM,
    HEALTH_TIMEOUT_S,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MAC_LAYER_END,
    MAC_LAYER_START,
    MODEL_NAME,
    N_HEADS,
    N_KV_HEADS,
    NUM_LAYERS,
    VISION_DEEPSTACK_INDEXES,
    VISION_DEPTH,
    VISION_OWNER,
    VOCAB_SIZE,
    WINDOWS_LAYER_END,
    WINDOWS_LAYER_START,
    assert_locked_split,
)

MAC_RANGE = [MAC_LAYER_START, MAC_LAYER_END]
WINDOWS_RANGE = [WINDOWS_LAYER_START, WINDOWS_LAYER_END]

_LANGUAGE_DIMS = {
    "hidden_size": HIDDEN_SIZE,
    "num_attention_heads": N_HEADS,
    "num_key_value_heads": N_KV_HEADS,
    "head_dim": HEAD_DIM,
    "intermediate_size": INTERMEDIATE_SIZE,
    "vocab_size": VOCAB_SIZE,
    "eos_token_id": EOS_TOKEN_ID,
}


def load_config(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(data)
    return data


def validate_config(data: dict) -> None:
    assert_locked_split()
    model = data.get("model")
    if model != MODEL_NAME or not isinstance(model, str) or "GGUF" in model:
        raise ValueError(f"v1 model must be the safetensors repo {MODEL_NAME}")
    if data.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("v1 checkpoint_format must be safetensors")
    if data.get("num_layers") != NUM_LAYERS:
        raise ValueError(f"v1 num_layers must be {NUM_LAYERS}")
    for key, expected in _LANGUAGE_DIMS.items():
        if data.get(key) != expected:
            raise ValueError(f"{key} must be {expected}")
    if data.get("vision_owner") != VISION_OWNER:
        raise ValueError("vision encoder must stay on mac")
    if data.get("vision_depth") != VISION_DEPTH:
        raise ValueError(f"vision depth must be {VISION_DEPTH}")
    if list(data.get("vision_deepstack_indexes") or []) != list(VISION_DEEPSTACK_INDEXES):
        raise ValueError("vision DeepStack indexes must be 8, 16, 24")
    if data.get("transport") != "cleartext-http":
        raise ValueError("v1 transport is cleartext-http with no encryption")
    if data.get("encryption") not in (False, None):
        raise ValueError("v1 does not encrypt hidden states")
    role = data.get("role")
    engine = data.get("engine")
    layer_range = data.get("layer_range")
    if role == "mac":
        if layer_range != MAC_RANGE:
            raise ValueError(f"Mac layer_range must be [{MAC_LAYER_START}, {MAC_LAYER_END}]")
        if engine not in {"mock", "vllm-metal"}:
            raise ValueError("Mac engine must be mock or vllm-metal")
    elif role == "windows":
        if layer_range != WINDOWS_RANGE:
            raise ValueError(
                f"Windows layer_range must be [{WINDOWS_LAYER_START}, {WINDOWS_LAYER_END}]"
            )
        if engine not in {"mock", "vllm-cuda"}:
            raise ValueError("Windows engine must be mock or vllm-cuda")
    else:
        raise ValueError("role must be mac or windows")
    timeout = data.get("health_timeout_s", HEALTH_TIMEOUT_S)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("health_timeout_s must be positive")


def _contract_fields() -> dict:
    return {
        "model": MODEL_NAME,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "num_layers": NUM_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTERMEDIATE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "eos_token_id": EOS_TOKEN_ID,
        "vision_owner": VISION_OWNER,
        "vision_depth": VISION_DEPTH,
        "vision_deepstack_indexes": list(VISION_DEEPSTACK_INDEXES),
        "transport": "cleartext-http",
        "encryption": False,
        "health_timeout_s": HEALTH_TIMEOUT_S,
    }


def mac_config(**overrides) -> dict:
    data = {
        "role": "mac",
        **_contract_fields(),
        "layer_range": MAC_RANGE,
        "engine": "mock",
        "mode": "auto",
        "host": "0.0.0.0",
        "port": 7410,
        "windows_kv_url": "",
        "windows_node_url": "",
        "max_new_tokens": 32,
        "max_model_len": 512,
    }
    data.update(overrides)
    validate_config(data)
    return data


def windows_config(**overrides) -> dict:
    data = {
        "role": "windows",
        **_contract_fields(),
        "layer_range": WINDOWS_RANGE,
        "engine": "mock",
        "host": "0.0.0.0",
        "kv_port": 7421,
        "node_port": 7422,
        "max_model_len": 512,
    }
    data.update(overrides)
    validate_config(data)
    return data
