"""Start the TokenHub Mac console or the Windows KV + layer agent."""

from __future__ import annotations

import argparse
import threading

from pool.config import load_config, mac_config, windows_config
from pool.protocol.constants import WINDOWS_LAYER_END, WINDOWS_LAYER_START


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tokenhub",
        description="TokenHub local agent。共享分层推理计算池。",
    )
    parser.add_argument("--role", choices=["mac", "windows"], required=True)
    parser.add_argument("--engine", choices=["mock", "vllm-metal", "vllm-cuda"])
    parser.add_argument("--config")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--kv-port", type=int)
    parser.add_argument("--node-port", type=int)
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=["auto", "single-mac", "dual"])
    parser.add_argument("--windows-kv")
    parser.add_argument("--windows-node")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--health-timeout", type=float)
    args = parser.parse_args(argv)

    if args.role == "mac":
        config = mac_config() if not args.config else load_config(args.config)
        if config["role"] != "mac":
            raise SystemExit("config role must be mac")
        _overlay_mac(config, args)
        from pool.agent.mac import serve_mac

        server = serve_mac(config)
        host, port = server.server_address[:2]
        print(f"TokenHub Mac console http://{host}:{port} engine={config['engine']} mode={config['mode']}")
        print("Hidden states travel in the clear. Front-half smoke quality does not count.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
        return

    config = windows_config() if not args.config else load_config(args.config)
    if config["role"] != "windows":
        raise SystemExit("config role must be windows")
    _overlay_windows(config, args)
    from pool.agent.windows import serve_windows

    kv_server, node_server, _store, _node = serve_windows(config)
    kv_host, kv_port = kv_server.server_address[:2]
    node_host, node_port = node_server.server_address[:2]
    print(f"TokenHub Windows KV  http://{kv_host}:{kv_port}  (cleartext)")
    print(
        f"TokenHub Windows Node http://{node_host}:{node_port}  "
        f"layers [{WINDOWS_LAYER_START}, {WINDOWS_LAYER_END})"
    )
    print("This process does not tokenize or emit plaintext.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        kv_server.shutdown()
        node_server.shutdown()


def _overlay_mac(config: dict, args: argparse.Namespace) -> None:
    if args.engine:
        config["engine"] = args.engine
    if args.host:
        config["host"] = args.host
    if args.port:
        config["port"] = args.port
    if args.model:
        config["model"] = args.model
    if args.mode:
        config["mode"] = args.mode
    if args.windows_kv is not None:
        config["windows_kv_url"] = args.windows_kv
    if args.windows_node is not None:
        config["windows_node_url"] = args.windows_node
    if args.max_new_tokens:
        config["max_new_tokens"] = args.max_new_tokens
    if args.max_model_len:
        config["max_model_len"] = args.max_model_len
    if args.health_timeout:
        config["health_timeout_s"] = args.health_timeout
    from pool.config import validate_config

    validate_config(config)


def _overlay_windows(config: dict, args: argparse.Namespace) -> None:
    if args.engine:
        config["engine"] = args.engine
    if args.host:
        config["host"] = args.host
    if args.kv_port:
        config["kv_port"] = args.kv_port
    if args.node_port:
        config["node_port"] = args.node_port
    if args.port and not args.kv_port:
        config["kv_port"] = args.port
    if args.model:
        config["model"] = args.model
    if args.max_model_len:
        config["max_model_len"] = args.max_model_len
    if args.health_timeout:
        config["health_timeout_s"] = args.health_timeout
    from pool.config import validate_config

    validate_config(config)
