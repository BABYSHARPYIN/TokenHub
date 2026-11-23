"""Mac agent: console, dispatcher, local front-half node, optional Windows clients."""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pool.config import mac_config
from pool.dispatch.dispatcher import Dispatcher
from pool.dispatch.tokenizer import ByteTokenizer, HuggingFaceTokenizer
from pool.errors import EngineUnavailable, SessionFailed
from pool.kv.memory import InMemoryKVStore
from pool.protocol.constants import (
    FRONT_HALF_WARNING,
    MAC_LAYER_END,
    MAC_LAYER_START,
    MODEL_NAME,
    PROTOCOL,
    VISION_DEEPSTACK_INDEXES,
    VISION_DEPTH,
    VISION_OWNER,
    WINDOWS_LAYER_END,
    WINDOWS_LAYER_START,
)
from pool.rpc.http import HttpKVClient, HttpLayerClient
from pool.runtime.mock import MockEmbedder
from pool.runtime.node import LayerNode, build_mock_mac_node

STATIC = Path(__file__).resolve().parent / "static"
FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/style.css": "style.css",
}


class MacApp:
    def __init__(self, dispatcher: Dispatcher, config: dict):
        self.dispatcher = dispatcher
        self.config = config
        self.lock = threading.Lock()

    def status(self) -> dict:
        peers = self.dispatcher.peer_status()
        return {
            "protocol": PROTOCOL,
            "role": "mac",
            "model": self.config["model"],
            "engine": self.config["engine"],
            "mode": self.config["mode"],
            "mac_layers": [MAC_LAYER_START, MAC_LAYER_END],
            "windows_layers": [WINDOWS_LAYER_START, WINDOWS_LAYER_END],
            "vision_owner": VISION_OWNER,
            "vision_depth": VISION_DEPTH,
            "vision_deepstack_indexes": list(VISION_DEEPSTACK_INDEXES),
            "checkpoint_format": "safetensors",
            "windows_kv": peers["windows_kv"],
            "windows_node": peers["windows_node"],
            "transport": "cleartext-http",
            "encryption": False,
            "resolved_mode": peers["resolved_mode"],
            "degraded_if_auto": peers["degraded_if_auto"],
            "front_half_warning": FRONT_HALF_WARNING,
        }

    def generate(self, payload: dict) -> tuple[dict, int]:
        prompt = payload.get("prompt", "")
        mode = payload.get("mode") or self.config["mode"]
        limit = int(payload.get("max_new_tokens") or self.config["max_new_tokens"])
        try:
            with self.lock:
                result = self.dispatcher.generate(prompt, mode=mode, max_new_tokens=limit)
        except SessionFailed as exc:
            return {"error": str(exc), "code": exc.code, "failed": True, "protocol": PROTOCOL}, 409
        except ValueError as exc:
            return {"error": str(exc), "failed": True, "protocol": PROTOCOL}, 400
        return result.to_json(), 200


def build_mac_stack(config: dict | None = None) -> MacApp:
    config = config or mac_config()
    engine = config["engine"]
    if engine == "mock":
        embedder = MockEmbedder()
        mac = build_mock_mac_node()
        tokenizer = ByteTokenizer()
        lm_head = mac.lm_head
    elif engine == "vllm-metal":
        from pool.runtime.vllm_metal import VllmMetalHiddenStateRunner

        runner = VllmMetalHiddenStateRunner(config["model"], config.get("max_model_len", 512))
        runner.load()
        mac = LayerNode("mac", MAC_LAYER_START, MAC_LAYER_END, runner, lm_head=runner)
        embedder = runner
        tokenizer = HuggingFaceTokenizer(config["model"])
        lm_head = runner
    else:
        raise EngineUnavailable("Mac agent accepts mock or vllm-metal")
    kv_url = (config.get("windows_kv_url") or "").strip()
    node_url = (config.get("windows_node_url") or "").strip()
    placeholder = "WINDOWS_PUBLIC_IP"
    if kv_url and node_url and placeholder not in kv_url and placeholder not in node_url:
        windows_kv = HttpKVClient(kv_url)
        windows_node = HttpLayerClient(node_url)
    else:
        windows_kv = None
        windows_node = None
    dispatcher = Dispatcher(
        mac_worker=mac,
        embedder=embedder,
        lm_head=lm_head,
        tokenizer=tokenizer,
        local_kv=InMemoryKVStore(),
        windows_worker=windows_node,
        windows_kv=windows_kv,
        health_timeout_s=float(config.get("health_timeout_s", 3.0)),
    )
    return MacApp(dispatcher, config)


class MacHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            self._json(200, self.server.app.status())
            return
        if path == "/health":
            self._json(200, {"ok": True, "role": "mac", "model": MODEL_NAME, "encryption": False})
            return
        name = FILES.get(path)
        if name is None:
            self._json(404, {"error": "not found"})
            return
        body = (STATIC / name).read_bytes()
        if name.endswith(".html"):
            mime = "text/html; charset=utf-8"
        elif name.endswith(".js"):
            mime = "text/javascript; charset=utf-8"
        elif name.endswith(".css"):
            mime = "text/css; charset=utf-8"
        else:
            mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        if path != "/api/generate":
            self._json(404, {"error": "not found"})
            return
        body, status = self.server.app.generate(payload)
        self._json(status, body)

    def _json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def serve_mac(config: dict) -> ThreadingHTTPServer:
    app = build_mac_stack(config)
    server = ThreadingHTTPServer((config["host"], int(config["port"])), MacHandler)
    server.app = app
    return server
