"""RPC structs shared by the dispatcher, KV service, and layer nodes."""

from __future__ import annotations

from dataclasses import dataclass

from pool.protocol.codec import TensorBlob
from pool.protocol.constants import PROTOCOL
from pool.protocol.guard import assert_activation_payload


@dataclass(frozen=True)
class Health:
    ok: bool
    role: str
    layer_start: int | None = None
    layer_end: int | None = None
    detail: str = ""
    encryption: bool = False
    transport: str = "cleartext-http"

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "role": self.role,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "detail": self.detail,
            "encryption": self.encryption,
            "transport": self.transport,
            "protocol": PROTOCOL,
        }

    @classmethod
    def from_json(cls, raw: dict) -> "Health":
        return cls(
            ok=bool(raw.get("ok")),
            role=str(raw.get("role", "")),
            layer_start=raw.get("layer_start"),
            layer_end=raw.get("layer_end"),
            detail=str(raw.get("detail", "")),
            encryption=bool(raw.get("encryption", False)),
            transport=str(raw.get("transport", "cleartext-http")),
        )


@dataclass(frozen=True)
class ForwardRequest:
    session_id: str
    phase: str
    layer_start: int
    layer_end: int
    position_start: int
    seq_len: int
    hidden: TensorBlob
    return_logits: bool = False
    past_keys: TensorBlob | None = None
    past_values: TensorBlob | None = None

    def public_dict(self) -> dict:
        payload = {
            "protocol": PROTOCOL,
            "session_id": self.session_id,
            "phase": self.phase,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "position_start": self.position_start,
            "seq_len": self.seq_len,
            "hidden": self.hidden.to_json(),
            "past_keys": None if self.past_keys is None else self.past_keys.to_json(),
            "past_values": None if self.past_values is None else self.past_values.to_json(),
            "return_logits": self.return_logits,
        }
        assert_activation_payload(payload)
        return payload

    @classmethod
    def from_public_dict(cls, raw: dict) -> "ForwardRequest":
        assert_activation_payload(raw)
        hidden = TensorBlob.from_json(raw["hidden"])
        if hidden.shape[0] != int(raw["seq_len"]):
            raise ValueError("seq_len does not match hidden rows")
        past_keys = TensorBlob.from_json(raw["past_keys"]) if raw.get("past_keys") else None
        past_values = TensorBlob.from_json(raw["past_values"]) if raw.get("past_values") else None
        return cls(
            session_id=raw["session_id"],
            phase=raw["phase"],
            layer_start=int(raw["layer_start"]),
            layer_end=int(raw["layer_end"]),
            position_start=int(raw["position_start"]),
            seq_len=int(raw["seq_len"]),
            hidden=hidden,
            return_logits=bool(raw["return_logits"]),
            past_keys=past_keys,
            past_values=past_values,
        )


@dataclass(frozen=True)
class ForwardResponse:
    hidden: TensorBlob
    keys: TensorBlob
    values: TensorBlob
    layer_start: int
    layer_end: int
    logits: TensorBlob | None = None

    def to_json(self, *, include_logits: bool) -> dict:
        payload = {
            "protocol": PROTOCOL,
            "hidden": self.hidden.to_json(),
            "keys": self.keys.to_json(),
            "values": self.values.to_json(),
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
        }
        if include_logits:
            payload["logits"] = None if self.logits is None else self.logits.to_json()
        return payload


@dataclass(frozen=True)
class GenerateResult:
    answer: str
    thinking: str
    degraded: str
    mode_used: str
    prompt_tokens: int
    completion_tokens: int
    session_id: str
    warning: str

    def to_json(self) -> dict:
        return {
            "answer": self.answer,
            "thinking": self.thinking,
            "degraded": self.degraded,
            "mode_used": self.mode_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "session_id": self.session_id,
            "warning": self.warning,
            "protocol": PROTOCOL,
        }
