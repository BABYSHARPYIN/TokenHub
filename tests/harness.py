"""Shared mock dual-stack for dispatcher tests."""

from __future__ import annotations

from pool.dispatch.dispatcher import Dispatcher
from pool.dispatch.tokenizer import ByteTokenizer
from pool.kv.memory import InMemoryKVStore
from pool.runtime.mock import MockEmbedder
from pool.runtime.node import build_mock_mac_node, build_mock_windows_node

SENTINEL = "SENTINEL_明文_do_not_leak_9f3a"


def make_dispatcher(
    *,
    windows_reachable: bool = True,
    kv_reachable: bool = True,
    fail_after: int | None = None,
    windows_worker=None,
    windows_kv=None,
    health_timeout_s: float = 3.0,
):
    mac = build_mock_mac_node()
    if windows_worker is None and windows_reachable:
        windows_worker = build_mock_windows_node(fail_after=fail_after)
    elif windows_worker is None:
        windows_worker = build_mock_windows_node(reachable=False, fail_after=fail_after)
    if windows_kv is None:
        windows_kv = InMemoryKVStore(reachable=kv_reachable)
    local_kv = InMemoryKVStore()
    dispatcher = Dispatcher(
        mac_worker=mac,
        embedder=MockEmbedder(),
        lm_head=mac.lm_head,
        tokenizer=ByteTokenizer(),
        local_kv=local_kv,
        windows_worker=windows_worker,
        windows_kv=windows_kv,
        health_timeout_s=health_timeout_s,
    )
    return dispatcher, mac, windows_worker, local_kv, windows_kv
