"""Example-config checks. v1 rejects other models, uneven splits, and encryption."""

from __future__ import annotations

import json
from pathlib import Path

from pool.protocol.constants import (
    HEALTH_TIMEOUT_S,
    MAC_LAYER_END,
    MAC_LAYER_START,
    MODEL_NAME,
    NUM_LAYERS,
    WINDOWS_LAYER_END,
    WINDOWS_LAYER_START,
    assert_locked_split,
)

MAC_RANGE = [MAC_LAYER_START, MAC_LAYER_END]
WINDOWS_RANGE = [WINDOWS_LAYER_START, WINDOWS_LAYER_END]


def load_config(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(data)
    return data


def validate_config(data: dict) -> None:
    assert_locked_split()
    if data.get("model") != MODEL_NAME:
        raise ValueError(f"v1 model must be {MODEL_NAME}")
    if data.get("num_layers") != NUM_LAYERS:
        raise ValueError("v1 num_layers must be 28")
    if data.get("transport") != "cleartext-http":
        raise ValueError("v1 transport is cleartext-http with no encryption")
    if data.get("encryption") not in (False, None):
        raise ValueError("v1 does not encrypt hidden states")
    role = data.get("role")
    engine = data.get("engine")
    layer_range = data.get("layer_range")
    if role == "mac":
        if layer_range != MAC_RANGE:
            raise ValueError("Mac layer_range must be [0, 14]")
        if engine not in {"mock", "vllm-metal"}:
            raise ValueError("Mac engine must be mock or vllm-metal")
    elif role == "windows":
        if layer_range != WINDOWS_RANGE:
            raise ValueError("Windows layer_range must be [14, 28]")
        if engine not in {"mock", "vllm-cuda"}:
            raise ValueError("Windows engine must be mock or vllm-cuda")
    else:
        raise ValueError("role must be mac or windows")
    timeout = data.get("health_timeout_s", HEALTH_TIMEOUT_S)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("health_timeout_s must be positive")


def mac_config(**overrides) -> dict:
    data = {
        "role": "mac",
        "model": MODEL_NAME,
        "num_layers": NUM_LAYERS,
        "layer_range": MAC_RANGE,
        "engine": "mock",
        "mode": "auto",
        "host": "0.0.0.0",
        "port": 7410,
        "windows_kv_url": "",
        "windows_node_url": "",
        "health_timeout_s": HEALTH_TIMEOUT_S,
        "max_new_tokens": 32,
        "max_model_len": 512,
        "transport": "cleartext-http",
        "encryption": False,
    }
    data.update(overrides)
    validate_config(data)
    return data


def windows_config(**overrides) -> dict:
    data = {
        "role": "windows",
        "model": MODEL_NAME,
        "num_layers": NUM_LAYERS,
        "layer_range": WINDOWS_RANGE,
        "engine": "mock",
        "host": "0.0.0.0",
        "kv_port": 7421,
        "node_port": 7422,
        "max_model_len": 512,
        "transport": "cleartext-http",
        "encryption": False,
        "health_timeout_s": HEALTH_TIMEOUT_S,
    }
    data.update(overrides)
    validate_config(data)
    return data
