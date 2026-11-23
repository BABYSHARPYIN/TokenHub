import json

from pool.protocol.guard import scrub_tensors
from tests.harness import SENTINEL, make_dispatcher


def _assert_windows_has_no_plaintext(payloads: list[dict]) -> None:
    assert payloads, "Windows node received no forward"
    for payload in payloads:
        scrubbed = json.dumps(scrub_tensors(payload), ensure_ascii=False)
        assert SENTINEL not in scrubbed
        assert payload["return_logits"] is False
        assert (payload["layer_start"], payload["layer_end"]) == (14, 28)
        assert payload["phase"] in {"prefill", "decode"}
        assert "prompt" not in payload
        assert "input_ids" not in payload
        assert "token_ids" not in payload
        assert "text" not in payload


def test_auto_dual_mock_loop_hides_plaintext():
    dispatcher, mac, windows, local_kv, windows_kv = make_dispatcher()
    result = dispatcher.generate(SENTINEL, mode="auto", max_new_tokens=3)

    assert result.degraded == ""
    assert result.warning == ""
    assert result.mode_used == "dual"
    assert result.answer
    assert result.prompt_tokens == len(SENTINEL.encode("utf-8"))
    assert result.completion_tokens == 3
    _assert_windows_has_no_plaintext(windows.seen)
    assert any(item["phase"] == "prefill" for item in windows.seen)
    assert any(item["phase"] == "decode" for item in windows.seen)
    assert all(item["layer_start"] == 0 and item["layer_end"] == 14 for item in mac.seen)
    assert all(item["return_logits"] is False for item in mac.seen)
    assert not any(item["layer_end"] >= 28 and item["layer_start"] == 0 for item in mac.seen)
    assert local_kv.writes == []
    assert any(item[1:] == (0, 14, 0) for item in windows_kv.writes)
    assert any(item[1:] == (14, 28, 0) for item in windows_kv.writes)
    assert windows_kv.sessions == {}
    assert result.session_id in windows_kv.deleted


def test_explicit_dual_matches_auto_when_windows_is_up():
    dispatcher, mac, windows, local_kv, windows_kv = make_dispatcher()
    result = dispatcher.generate(SENTINEL, mode="dual", max_new_tokens=2)
    assert result.degraded == ""
    assert result.mode_used == "dual"
    _assert_windows_has_no_plaintext(windows.seen)
    assert len(mac.seen) == len(windows.seen) == 2
    assert local_kv.writes == []
    assert windows_kv.writes
