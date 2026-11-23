import json
import threading
from urllib import error, request

import pytest

from pool.agent.mac import serve_mac
from pool.config import mac_config
from pool.dispatch.assembler import assemble
from pool.dispatch.dispatcher import Dispatcher
from pool.dispatch.tokenizer import ByteTokenizer
from pool.kv.memory import InMemoryKVStore
from pool.protocol.guard import scrub_tensors
from pool.rpc.http import HttpKVClient, HttpLayerClient, KVApp, NodeApp, serve_in_thread
from pool.runtime.mock import MockEmbedder
from pool.runtime.node import build_mock_mac_node, build_mock_windows_node
from tests.harness import SENTINEL


def test_cleartext_http_dual_path_has_no_plaintext():
    store = InMemoryKVStore()
    node = build_mock_windows_node()
    kv_app = KVApp(store)
    node_app = NodeApp(node)
    kv_server = serve_in_thread(kv_app, "127.0.0.1", 0)
    node_server = serve_in_thread(node_app, "127.0.0.1", 0)
    kv_port = kv_server.server_address[1]
    node_port = node_server.server_address[1]
    mac = build_mock_mac_node()
    dispatcher = Dispatcher(
        mac_worker=mac,
        embedder=MockEmbedder(),
        lm_head=mac.lm_head,
        tokenizer=ByteTokenizer(),
        local_kv=InMemoryKVStore(),
        windows_worker=HttpLayerClient(f"http://127.0.0.1:{node_port}"),
        windows_kv=HttpKVClient(f"http://127.0.0.1:{kv_port}"),
        health_timeout_s=2.0,
    )
    result = dispatcher.generate(SENTINEL, mode="dual", max_new_tokens=2)
    assert result.degraded == ""
    bodies = kv_app.raw_bodies + node_app.raw_bodies
    assert bodies
    for raw in bodies:
        payload = json.loads(raw)
        scrubbed = json.dumps(scrub_tensors(payload), ensure_ascii=False)
        assert SENTINEL not in scrubbed
        assert "prompt" not in payload
    assert all(item["return_logits"] is False for item in node.seen)
    kv_server.shutdown()
    node_server.shutdown()


def test_console_front_half_and_dual_failure():
    config = mac_config(host="127.0.0.1", port=0, mode="auto")
    server = serve_mac(config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    with request.urlopen(base + "/", timeout=3) as response:
        page = response.read().decode("utf-8")
    assert "TokenHub" in page
    assert "共享分层推理计算池" in page
    assert "Qwen/Qwen3-VL-8B-Instruct" in page
    assert "[0, 18)" in page
    assert "[18, 36)" in page
    assert "视觉" in page
    assert "27" in page
    assert "质量不计" in page or "mac_front_half" in page
    assert "Qwen2.5-" + "7B" not in page
    with request.urlopen(base + "/api/status", timeout=3) as response:
        status = json.loads(response.read().decode("utf-8"))
    assert status["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert status["mac_layers"] == [0, 18]
    assert status["windows_layers"] == [18, 36]
    assert status["vision_owner"] == "mac"
    assert status["vision_depth"] == 27
    assert status["vision_deepstack_indexes"] == [8, 16, 24]
    payload = json.dumps(
        {"prompt": SENTINEL, "mode": "auto", "max_new_tokens": 2},
    ).encode("utf-8")
    req = request.Request(base + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=5) as response:
        body = json.loads(response.read().decode("utf-8"))
    assert body["degraded"] == "mac_front_half"
    assert "does not count" in body["warning"]
    dual = json.dumps({"prompt": SENTINEL, "mode": "dual", "max_new_tokens": 2}).encode("utf-8")
    dual_req = request.Request(base + "/api/generate", data=dual, headers={"Content-Type": "application/json"})
    with pytest.raises(error.HTTPError) as caught:
        request.urlopen(dual_req, timeout=5)
    assert caught.value.code == 409
    server.shutdown()


def test_assembler_splits_thinking():
    thinking, answer = assemble("<think>内部</think>可见回答")
    assert thinking == "内部"
    assert answer == "可见回答"
    assert assemble("只有回答") == ("", "只有回答")
