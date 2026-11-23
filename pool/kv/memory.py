"""Session-scoped KV store.

Each layer keeps a list of per-token key and value vectors. The store is the
Windows service in a dual session and a local Mac stand-in during front-half smoke.
"""

from __future__ import annotations

from pool.errors import PoolError
from pool.protocol.codec import TensorBlob, as_blob
from pool.protocol.constants import NUM_LAYERS
from pool.protocol.guard import assert_kv_append
from pool.protocol.types import Health


class InMemoryKVStore:
    def __init__(self, *, reachable: bool = True, num_layers: int = NUM_LAYERS):
        self.reachable = reachable
        self.num_layers = num_layers
        self.sessions: dict[str, dict[str, list]] = {}
        self.deleted: list[str] = []
        self.writes: list[tuple] = []

    def health(self, timeout: float = 3.0) -> Health:
        del timeout
        if not self.reachable:
            raise TimeoutError("kv unreachable")
        return Health(ok=True, role="kv", detail="in-memory")

    def create(self, session_id: str) -> None:
        self._require()
        if session_id in self.sessions:
            raise PoolError(f"session already exists: {session_id}")
        self.sessions[session_id] = {
            "k": [[] for _ in range(self.num_layers)],
            "v": [[] for _ in range(self.num_layers)],
        }

    def append(
        self,
        session_id: str,
        layer_start: int,
        layer_end: int,
        position_start: int,
        keys: TensorBlob,
        values: TensorBlob,
        *,
        prompt: str = "",
    ) -> None:
        self._require()
        payload = {
            "protocol": "pool.v1",
            "session_id": session_id,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "position_start": position_start,
            "keys": keys.to_json(),
            "values": values.to_json(),
        }
        assert_kv_append(payload, prompt)
        if session_id not in self.sessions:
            raise PoolError(f"unknown session: {session_id}")
        key_rows = keys.nested()
        value_rows = values.nested()
        if len(key_rows) != layer_end - layer_start:
            raise PoolError("KV layer count does not match the requested range")
        session = self.sessions[session_id]
        for offset, layer in enumerate(range(layer_start, layer_end)):
            if position_start != len(session["k"][layer]):
                raise PoolError(
                    f"KV position gap at layer {layer}: have {len(session['k'][layer])}, "
                    f"append starts at {position_start}"
                )
            session["k"][layer].extend(key_rows[offset])
            session["v"][layer].extend(value_rows[offset])
        self.writes.append((session_id, layer_start, layer_end, position_start))

    def read(self, session_id: str, layer_start: int, layer_end: int):
        self._require()
        session = self.sessions[session_id]
        keys = [session["k"][layer] for layer in range(layer_start, layer_end)]
        values = [session["v"][layer] for layer in range(layer_start, layer_end)]
        if not keys or not keys[0]:
            return None, None
        return as_blob(keys), as_blob(values)

    def delete(self, session_id: str) -> None:
        self.deleted.append(session_id)
        self.sessions.pop(session_id, None)

    def _require(self) -> None:
        if not self.reachable:
            raise ConnectionError("kv unreachable")
