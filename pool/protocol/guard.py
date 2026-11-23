"""Allowlists for activation RPC.

Windows payloads may contain session ids, layer indexes, and float tensors.
They may not contain a prompt, token ids, decoded text, or logits.
"""

from __future__ import annotations

import re

from pool.errors import LayerRangeError, PlaintextLeak
from pool.protocol.constants import NUM_LAYERS, PROTOCOL

_SESSION = re.compile(r"^[a-f0-9]{32}$")
_TENSOR_KEYS = {"dtype", "shape", "data_b64"}

FORWARD_FIELDS = {
    "protocol",
    "session_id",
    "phase",
    "layer_start",
    "layer_end",
    "position_start",
    "seq_len",
    "hidden",
    "past_keys",
    "past_values",
    "return_logits",
}
KV_APPEND_FIELDS = {
    "protocol",
    "session_id",
    "layer_start",
    "layer_end",
    "position_start",
    "keys",
    "values",
}
KV_READ_FIELDS = {"protocol", "session_id", "layer_start", "layer_end"}
TENSOR_FIELDS = {"hidden", "past_keys", "past_values", "keys", "values"}


def _assert_tensor(value, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != _TENSOR_KEYS:
        raise PlaintextLeak(f"{field} must be a float32 tensor object")
    if value.get("dtype") != "float32":
        raise PlaintextLeak(f"{field} dtype must be float32")
    if not isinstance(value.get("shape"), list) or not all(isinstance(x, int) for x in value["shape"]):
        raise PlaintextLeak(f"{field} shape is invalid")
    if not isinstance(value.get("data_b64"), str):
        raise PlaintextLeak(f"{field} data is invalid")


def assert_allowlist(payload: dict, allowed: set[str]) -> None:
    if not isinstance(payload, dict):
        raise PlaintextLeak("payload must be a JSON object")
    extra = set(payload) - allowed
    if extra:
        raise PlaintextLeak(f"unexpected fields: {sorted(extra)}")
    if payload.get("protocol") != PROTOCOL:
        raise PlaintextLeak("unknown protocol")
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not _SESSION.fullmatch(session_id):
        raise PlaintextLeak("session_id is not a pool id")
    for key, value in payload.items():
        if key in TENSOR_FIELDS:
            _assert_tensor(value, key)
            continue
        if key in {"protocol", "session_id", "phase"}:
            if not isinstance(value, str):
                raise PlaintextLeak(f"{key} must be a string")
            continue
        if key == "return_logits":
            if not isinstance(value, bool):
                raise PlaintextLeak("return_logits must be a boolean")
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise PlaintextLeak(f"{key} must be an integer")


def assert_activation_payload(payload: dict) -> None:
    """Reject text-bearing fields on any node, including Mac."""
    assert_allowlist(payload, FORWARD_FIELDS)
    if payload["phase"] not in {"prefill", "decode"}:
        raise PlaintextLeak("phase must be prefill or decode")
    if payload["layer_end"] <= payload["layer_start"]:
        raise LayerRangeError("empty layer range")
    if payload["layer_start"] == 0 and payload["layer_end"] >= NUM_LAYERS:
        raise LayerRangeError(f"refusing a full-depth [0, {NUM_LAYERS}) forward")


def assert_windows_request(payload: dict, prompt: str) -> None:
    """Windows may receive activations only. The prompt string must not appear."""
    assert_activation_payload(payload)
    if payload["return_logits"]:
        raise PlaintextLeak("Windows cannot be asked for logits")
    if payload["layer_start"] < 0 or payload["layer_end"] > NUM_LAYERS:
        raise LayerRangeError("remote hop is outside the language tower")
    _reject_prompt(payload, prompt)


def assert_kv_append(payload: dict, prompt: str = "") -> None:
    assert_allowlist(payload, KV_APPEND_FIELDS)
    if payload["layer_end"] <= payload["layer_start"]:
        raise LayerRangeError("empty KV range")
    _reject_prompt(payload, prompt)


# Structural tokens are protocol machinery, not user text. Tensor bytes are skipped
# because they are activations; the allowlist already forbids a prompt field.
_STRUCTURAL_FIELDS = {"protocol", "phase", "dtype", "session_id", "data_b64"}


def _reject_prompt(payload: dict, prompt: str) -> None:
    if not prompt:
        return
    for text in _free_strings(payload, None):
        if prompt in text:
            raise PlaintextLeak("plaintext prompt present in activation payload")


def _free_strings(value, key: str | None):
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _free_strings(child, child_key)
        return
    if isinstance(value, list):
        for item in value:
            yield from _free_strings(item, key)
        return
    if isinstance(value, str) and key not in _STRUCTURAL_FIELDS:
        yield value


def scrub_tensors(value):
    """Drop tensor bytes so tests can search the remaining JSON for plaintext."""
    if isinstance(value, dict):
        return {key: ("" if key == "data_b64" else scrub_tensors(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_tensors(item) for item in value]
    return value
