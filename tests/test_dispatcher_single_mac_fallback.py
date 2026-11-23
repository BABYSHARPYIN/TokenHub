import time

import pytest

from pool.errors import SessionFailed
from pool.kv.memory import InMemoryKVStore
from pool.protocol.types import Health
from pool.runtime.node import build_mock_windows_node
from tests.harness import SENTINEL, make_dispatcher


def _assert_front_half(result, mac, windows, local_kv, windows_kv):
    assert result.degraded == "mac_front_half"
    assert "does not count" in result.warning
    assert "质量不计" in result.warning
    assert result.mode_used == "single-mac"
    assert result.answer
    assert windows.seen == []
    assert mac.seen, "Mac did not run"
    assert any(item["phase"] == "prefill" and (item["layer_start"], item["layer_end"]) == (0, 14) for item in mac.seen)
    assert all((item["layer_start"], item["layer_end"]) == (0, 14) for item in mac.seen)
    assert all(item["return_logits"] is True for item in mac.seen)
    assert not any(item["layer_start"] == 0 and item["layer_end"] >= 28 for item in mac.seen)
    assert windows_kv.writes == []
    assert windows_kv.sessions == {}
    assert local_kv.writes
    assert all((item[1], item[2]) == (0, 14) for item in local_kv.writes)
    assert any(item[3] == 0 for item in local_kv.writes)


def test_auto_windows_down_uses_front_half():
    dispatcher, mac, windows, local_kv, windows_kv = make_dispatcher(
        windows_reachable=False,
        kv_reachable=False,
    )
    result = dispatcher.generate(SENTINEL, mode="auto", max_new_tokens=3)
    _assert_front_half(result, mac, windows, local_kv, windows_kv)


def test_single_mac_ignores_healthy_windows():
    dispatcher, mac, windows, local_kv, windows_kv = make_dispatcher()
    result = dispatcher.generate(SENTINEL, mode="single-mac", max_new_tokens=2)
    _assert_front_half(result, mac, windows, local_kv, windows_kv)


def test_dual_windows_down_fails():
    dispatcher, mac, windows, local_kv, windows_kv = make_dispatcher(
        windows_reachable=False,
        kv_reachable=False,
    )
    with pytest.raises(SessionFailed) as caught:
        dispatcher.generate(SENTINEL, mode="dual", max_new_tokens=4)
    assert caught.value.code == "windows_unreachable"
    assert windows.seen == []
    assert mac.seen == []
    assert local_kv.writes == []
    assert windows_kv.writes == []


def test_wrong_windows_range_is_not_a_dual_peer():
    class WrongRange:
        seen: list = []

        def health(self, timeout: float = 3.0):
            del timeout
            return Health(ok=True, role="windows", layer_start=0, layer_end=28)

        def forward(self, request):
            raise AssertionError(f"forwarded to a full-depth peer: {request}")

    dispatcher, mac, windows, local_kv, windows_kv = make_dispatcher(windows_worker=WrongRange())
    result = dispatcher.generate(SENTINEL, mode="auto", max_new_tokens=2)
    _assert_front_half(result, mac, windows, local_kv, windows_kv)
    with pytest.raises(SessionFailed) as caught:
        dispatcher.generate(SENTINEL, mode="dual", max_new_tokens=2)
    assert caught.value.code == "windows_unreachable"


def test_dual_mid_session_drop_fails_and_cleans_kv():
    dispatcher, mac, windows, local_kv, windows_kv = make_dispatcher(fail_after=1)
    with pytest.raises(SessionFailed) as caught:
        dispatcher.generate(SENTINEL, mode="dual", max_new_tokens=4)
    assert caught.value.code == "windows_dropped"
    assert "not switching to front-half" in str(caught.value)
    assert len(windows.seen) == 1
    assert windows.seen[0]["phase"] == "prefill"
    assert len(mac.seen) == 2
    assert all((item["layer_start"], item["layer_end"]) == (0, 14) for item in mac.seen)
    assert windows_kv.sessions == {}
    assert windows_kv.deleted
    assert local_kv.writes == []


def test_health_timeout_falls_back_without_hanging():
    class SlowNode:
        seen: list = []

        def health(self, timeout: float = 3.0):
            del timeout
            time.sleep(1.5)
            return Health(ok=True, role="windows", layer_start=14, layer_end=28)

        def forward(self, request):
            raise AssertionError("slow node should not receive a forward")

    dispatcher, mac, windows, local_kv, windows_kv = make_dispatcher(
        windows_worker=SlowNode(),
        windows_kv=InMemoryKVStore(),
        health_timeout_s=0.2,
    )
    started = time.monotonic()
    result = dispatcher.generate(SENTINEL, mode="auto", max_new_tokens=1)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    _assert_front_half(result, mac, windows, local_kv, windows_kv)


def test_kv_down_alone_is_not_dual():
    dispatcher, mac, windows, local_kv, windows_kv = make_dispatcher(
        kv_reachable=False,
        windows_worker=build_mock_windows_node(),
    )
    result = dispatcher.generate(SENTINEL, mode="auto", max_new_tokens=1)
    _assert_front_half(result, mac, windows, local_kv, windows_kv)
