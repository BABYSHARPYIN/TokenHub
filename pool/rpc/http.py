"""Cleartext HTTP for KV and layer RPC. There is no TLS in v1."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

from pool.errors import LayerRangeError, PlaintextLeak, PoolError
from pool.protocol.codec import TensorBlob
from pool.protocol.constants import PROTOCOL
from pool.protocol.guard import assert_kv_append, assert_windows_request
from pool.protocol.types import ForwardRequest, Health


class JsonServer(ThreadingHTTPServer):
    def __init__(self, host: str, port: int, app):
        self.app = app
        super().__init__((host, port), JsonHandler)


class JsonHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        if raw:
            self.server.app.raw_bodies.append(raw.decode("utf-8"))
        path = self.path.split("?", 1)[0]
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(payload, dict):
                raise PlaintextLeak("JSON body must be an object")
            body, status = self.server.app.handle(method, path, payload)
        except (PlaintextLeak, LayerRangeError, PoolError, ValueError, KeyError) as exc:
            body, status = {"error": str(exc), "protocol": PROTOCOL}, 400
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class App:
    def __init__(self):
        self.raw_bodies: list[str] = []

    def handle(self, method: str, path: str, payload: dict) -> tuple[dict, int]:
        raise NotImplementedError


def serve_in_thread(app: App, host: str = "127.0.0.1", port: int = 0) -> JsonServer:
    import threading

    server = JsonServer(host, port, app)
    thread = threading.Thread(target=server.serve_forever, name=app.__class__.__name__, daemon=True)
    thread.start()
    return server


class KVApp(App):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def handle(self, method: str, path: str, payload: dict) -> tuple[dict, int]:
        if method == "GET" and path == "/health":
            return self.store.health().to_json(), 200
        if method == "POST" and path == "/v1/sessions":
            self.store.create(payload["session_id"])
            return {"ok": True, "protocol": PROTOCOL}, 200
        if method == "POST" and path == "/v1/sessions/append":
            assert_kv_append(payload, "")
            self.store.append(
                payload["session_id"],
                int(payload["layer_start"]),
                int(payload["layer_end"]),
                int(payload["position_start"]),
                TensorBlob.from_json(payload["keys"]),
                TensorBlob.from_json(payload["values"]),
            )
            return {"ok": True, "protocol": PROTOCOL}, 200
        if method == "POST" and path == "/v1/sessions/read":
            keys, values = self.store.read(
                payload["session_id"],
                int(payload["layer_start"]),
                int(payload["layer_end"]),
            )
            return {
                "protocol": PROTOCOL,
                "keys": None if keys is None else keys.to_json(),
                "values": None if values is None else values.to_json(),
            }, 200
        if method == "POST" and path == "/v1/sessions/delete":
            self.store.delete(payload["session_id"])
            return {"ok": True, "protocol": PROTOCOL}, 200
        return {"error": "not found", "protocol": PROTOCOL}, 404


class NodeApp(App):
    def __init__(self, node):
        super().__init__()
        self.node = node

    def handle(self, method: str, path: str, payload: dict) -> tuple[dict, int]:
        if method == "GET" and path == "/health":
            return self.node.health().to_json(), 200
        if method == "POST" and path == "/v1/forward":
            assert_windows_request(payload, "")
            response = self.node.forward(ForwardRequest.from_public_dict(payload))
            if response.logits is not None:
                raise PlaintextLeak("Windows forward response must not carry logits")
            return response.to_json(include_logits=False), 200
        return {"error": "not found", "protocol": PROTOCOL}, 404


def _call(url: str, method: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ConnectionError(detail or exc.reason) from exc
    except error.URLError as exc:
        raise ConnectionError(str(exc.reason)) from exc


class HttpKVClient:
    def __init__(self, base_url: str, *, forward_timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.forward_timeout_s = forward_timeout_s

    def health(self, timeout: float = 3.0) -> Health:
        return Health.from_json(_call(self.base_url + "/health", "GET", None, timeout))

    def create(self, session_id: str) -> None:
        _call(
            self.base_url + "/v1/sessions",
            "POST",
            {"session_id": session_id, "protocol": PROTOCOL},
            self.forward_timeout_s,
        )

    def append(self, session_id, layer_start, layer_end, position_start, keys, values, prompt: str = "") -> None:
        payload = {
            "protocol": PROTOCOL,
            "session_id": session_id,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "position_start": position_start,
            "keys": keys.to_json(),
            "values": values.to_json(),
        }
        assert_kv_append(payload, prompt)
        _call(self.base_url + "/v1/sessions/append", "POST", payload, self.forward_timeout_s)

    def read(self, session_id, layer_start, layer_end):
        payload = {
            "protocol": PROTOCOL,
            "session_id": session_id,
            "layer_start": layer_start,
            "layer_end": layer_end,
        }
        raw = _call(self.base_url + "/v1/sessions/read", "POST", payload, self.forward_timeout_s)
        keys = TensorBlob.from_json(raw["keys"]) if raw.get("keys") else None
        values = TensorBlob.from_json(raw["values"]) if raw.get("values") else None
        return keys, values

    def delete(self, session_id: str) -> None:
        _call(
            self.base_url + "/v1/sessions/delete",
            "POST",
            {"session_id": session_id, "protocol": PROTOCOL},
            self.forward_timeout_s,
        )


class HttpLayerClient:
    def __init__(self, base_url: str, *, forward_timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.forward_timeout_s = forward_timeout_s
        self.seen: list[dict] = []

    def health(self, timeout: float = 3.0) -> Health:
        return Health.from_json(_call(self.base_url + "/health", "GET", None, timeout))

    def forward(self, forward_request: ForwardRequest):
        payload = forward_request.public_dict()
        self.seen.append(payload)
        raw = _call(self.base_url + "/v1/forward", "POST", payload, self.forward_timeout_s)
        from pool.protocol.types import ForwardResponse

        return ForwardResponse(
            hidden=TensorBlob.from_json(raw["hidden"]),
            keys=TensorBlob.from_json(raw["keys"]),
            values=TensorBlob.from_json(raw["values"]),
            layer_start=int(raw["layer_start"]),
            layer_end=int(raw["layer_end"]),
            logits=None,
        )
