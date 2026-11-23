"""Float32 tensor codec.

Hidden states and KV travel as little-endian float32 bytes inside JSON.
The bytes are not encrypted.
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass

from pool.protocol.constants import PROTOCOL


@dataclass(frozen=True)
class TensorBlob:
    dtype: str
    shape: tuple[int, ...]
    data_b64: str

    def to_json(self) -> dict:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_b64": self.data_b64,
        }

    @classmethod
    def from_json(cls, raw: dict) -> "TensorBlob":
        if not isinstance(raw, dict):
            raise ValueError("tensor payload must be an object")
        if set(raw) - {"dtype", "shape", "data_b64"}:
            raise ValueError("tensor payload has unexpected fields")
        if raw.get("dtype") != "float32":
            raise ValueError("v1 tensors are float32")
        shape = tuple(int(x) for x in raw["shape"])
        if any(x < 0 for x in shape):
            raise ValueError("negative dimension")
        blob = cls(dtype="float32", shape=shape, data_b64=str(raw["data_b64"]))
        flat = blob.flat()
        expect = 1
        for dim in shape:
            expect *= dim
        if len(flat) != expect:
            raise ValueError(f"tensor has {len(flat)} values, shape wants {expect}")
        return blob

    def flat(self) -> list[float]:
        raw = base64.b64decode(self.data_b64.encode("ascii"), validate=True)
        count = len(raw) // 4
        if len(raw) != count * 4:
            raise ValueError("float32 payload is not aligned")
        if count == 0:
            return []
        return list(struct.unpack("<" + "f" * count, raw))

    def nested(self):
        return reshape(self.flat(), self.shape)


def reshape(flat: list[float], shape: tuple[int, ...]):
    if not shape:
        if len(flat) != 1:
            raise ValueError("scalar shape mismatch")
        return flat[0]
    expect = 1
    for dim in shape:
        expect *= dim
    if len(flat) != expect:
        raise ValueError("cannot reshape")
    if len(shape) == 1:
        return list(flat)

    def rec(values: list[float], dims: tuple[int, ...]):
        if len(dims) == 1:
            return list(values)
        step = 1
        for dim in dims[1:]:
            step *= dim
        return [rec(values[i * step : (i + 1) * step], dims[1:]) for i in range(dims[0])]

    return rec(flat, shape)


def _shape_of(value) -> list[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        inner = _shape_of(value[0])
        for item in value[1:]:
            if _shape_of(item) != inner:
                raise ValueError("ragged tensor")
        return [len(value), *inner]
    return []


def _flatten(value) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [float(value)]


def as_blob(values, *, shape: tuple[int, ...] | None = None) -> TensorBlob:
    if shape is None:
        if isinstance(values, list) and not values:
            raise ValueError("empty tensors need an explicit shape")
        shape = tuple(_shape_of(values))
        flat = _flatten(values)
    else:
        flat = _flatten(values) if not isinstance(values, list) or values else _flatten(values)
        # Allow an explicit empty shape with no values.
        if any(dim == 0 for dim in shape):
            flat = []
        else:
            flat = _flatten(values)
    expect = 1
    for dim in shape:
        expect *= dim
    if len(flat) != expect:
        raise ValueError(f"blob has {len(flat)} values, shape {shape} wants {expect}")
    raw = struct.pack("<" + "f" * len(flat), *flat) if flat else b""
    return TensorBlob(
        dtype="float32",
        shape=tuple(int(x) for x in shape),
        data_b64=base64.b64encode(raw).decode("ascii"),
    )


def protocol_envelope(payload: dict) -> dict:
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol must be {PROTOCOL}")
    return payload
