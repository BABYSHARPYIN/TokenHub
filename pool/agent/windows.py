"""Windows agent: KV service plus the back-half layer node.

This process does not turn text into ids and does not return decoded text.
"""

from __future__ import annotations

from pool.config import windows_config
from pool.errors import EngineUnavailable
from pool.kv.memory import InMemoryKVStore
from pool.rpc.http import KVApp, NodeApp, serve_in_thread
from pool.runtime.node import LayerNode, build_mock_windows_node


def build_windows_node(config: dict):
    engine = config["engine"]
    if engine == "mock":
        return build_mock_windows_node()
    if engine == "vllm-cuda":
        from pool.runtime.vllm_cuda import VllmCudaHiddenStateRunner

        runner = VllmCudaHiddenStateRunner(config["model"], config.get("max_model_len", 512))
        runner.load()
        return LayerNode("windows", 14, 28, runner, lm_head=None)
    raise EngineUnavailable("Windows agent accepts mock or vllm-cuda")


def serve_windows(config: dict | None = None):
    config = config or windows_config()
    store = InMemoryKVStore()
    node = build_windows_node(config)
    kv_server = serve_in_thread(KVApp(store), config["host"], int(config["kv_port"]))
    node_server = serve_in_thread(NodeApp(node), config["host"], int(config["node_port"]))
    return kv_server, node_server, store, node
