import importlib.util
import json
from pathlib import Path

import pytest

from pool.config import load_config, validate_config
from pool.errors import EngineUnavailable, LayerRangeError
from pool.protocol.codec import as_blob
from pool.protocol.constants import MODEL_NAME, assert_locked_split
from pool.protocol.guard import assert_windows_request
from pool.runtime.node import LayerNode
from pool.runtime.runner import HiddenStateRunner
from pool.runtime.vllm_cuda import VllmCudaHiddenStateRunner
from pool.runtime.vllm_metal import VllmMetalHiddenStateRunner

ROOT = Path(__file__).resolve().parents[1]


def test_locked_split_and_configs():
    assert_locked_split()
    mac = load_config(ROOT / "configs" / "mac.example.json")
    windows = load_config(ROOT / "configs" / "windows.example.json")
    assert mac["layer_range"] == [0, 14]
    assert windows["layer_range"] == [14, 28]
    assert mac["model"] == windows["model"] == MODEL_NAME
    assert mac["encryption"] is False and windows["encryption"] is False
    with pytest.raises(ValueError):
        validate_config({**mac, "model": "Qwen/Qwen2.5-32B-Instruct"})
    with pytest.raises(ValueError):
        validate_config({**mac, "layer_range": [0, 28]})
    with pytest.raises(ValueError):
        validate_config({**mac, "encryption": True})
    with pytest.raises(ValueError):
        validate_config({**windows, "engine": "vllm-metal"})


def test_float32_roundtrip():
    blob = as_blob([[1.5, -2.0], [3.25, 4.0]])
    assert blob.dtype == "float32"
    assert blob.nested()[0][1] == pytest.approx(-2.0)
    assert blob.nested()[1][0] == pytest.approx(3.25)


def test_nodes_refuse_full_depth():
    with pytest.raises(LayerRangeError):
        LayerNode("full", 0, 28, runner=None, lm_head=None)
    mac = HiddenStateRunner("mac")
    with pytest.raises(LayerRangeError):
        mac.forward(as_blob([[1.0, 2.0]]), 0, 28, None, None, 0)
    windows = HiddenStateRunner("windows")
    with pytest.raises(LayerRangeError):
        windows.embed([1])
    with pytest.raises(LayerRangeError):
        windows.apply(as_blob([[1.0]]))


def test_real_engines_fail_closed_without_hardware():
    metal = VllmMetalHiddenStateRunner()
    cuda = VllmCudaHiddenStateRunner()
    assert (metal.layer_start, metal.layer_end) == (0, 14)
    assert (cuda.layer_start, cuda.layer_end) == (14, 28)
    metal_ready = importlib.util.find_spec("mlx") and importlib.util.find_spec("vllm")
    if metal_ready:
        pytest.skip("vllm-metal imports are present; default suite does not download weights")
    with pytest.raises(EngineUnavailable):
        metal.load()
    cuda_ready = importlib.util.find_spec("torch") and importlib.util.find_spec("vllm")
    if cuda_ready:
        import torch

        if torch.cuda.is_available():
            pytest.skip("CUDA vLLM is present; default suite does not download weights")
    with pytest.raises(EngineUnavailable):
        cuda.load()


def test_adapters_do_not_call_chat_api():
    banned = ("chat/completions", "llm.chat", "llm.generate", "ChatCompletion", "openai.OpenAI")
    for path in (ROOT / "pool").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for item in banned:
            assert item not in text, f"{path} contains {item}"


def test_windows_agent_has_no_text_codec():
    text = (ROOT / "pool" / "agent" / "windows.py").read_text(encoding="utf-8")
    assert "pool.dispatch" not in text
    assert "ByteTokenizer" not in text
    assert "encode(" not in text
    assert "detokenize" not in text


def test_windows_request_rejects_prompt_field():
    hidden = as_blob([[0.1, 0.2]])
    payload = {
        "protocol": "pool.v1",
        "session_id": "a" * 32,
        "phase": "prefill",
        "layer_start": 14,
        "layer_end": 28,
        "position_start": 0,
        "seq_len": 1,
        "hidden": hidden.to_json(),
        "past_keys": None,
        "past_values": None,
        "return_logits": False,
        "prompt": "leak",
    }
    with pytest.raises(Exception):
        assert_windows_request(payload, "leak")
    assert "leak" not in json.dumps({key: value for key, value in payload.items() if key != "prompt"})
