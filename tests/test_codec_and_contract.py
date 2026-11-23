import importlib.util
import json
from pathlib import Path

import pytest

from pool.config import load_config, validate_config
from pool.errors import EngineUnavailable, LayerRangeError
from pool.protocol.codec import as_blob
from pool.protocol.constants import (
    CHECKPOINT_FORMAT,
    EOS_TOKEN_ID,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MODEL_NAME,
    N_HEADS,
    N_KV_HEADS,
    NUM_LAYERS,
    VISION_DEEPSTACK_INDEXES,
    VISION_DEPTH,
    VISION_OWNER,
    VOCAB_SIZE,
    assert_locked_split,
)
from pool.protocol.guard import assert_windows_request
from pool.runtime.node import LayerNode
from pool.runtime.runner import HiddenStateRunner
from pool.runtime.vllm_cuda import VllmCudaHiddenStateRunner
from pool.runtime.vllm_metal import VllmMetalHiddenStateRunner

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_name_is_tokenhub():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "tokenhub"' in text
    assert "layered inference pool" not in text
    assert 'name = "pool"' not in text


def test_user_facing_launch_uses_python_m_pool():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m pool --role mac --engine mock" in readme
    assert "python -m pool --role windows --engine" in readme
    assert "tokenhub --role" not in readme
    assert "启动命令是 `python -m pool`" in readme
    for relative in (
        "pool/cli.py",
        "pool/__main__.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "prog=\"tokenhub\"" not in text
        assert "Start TokenHub with the `tokenhub`" not in text


def test_cli_help_uses_python_m_pool():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pool", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: python -m pool" in result.stdout
    assert "usage: tokenhub" not in result.stdout
    assert "TokenHub" in result.stdout


def test_locked_split_and_configs():
    assert_locked_split()
    mac = load_config(ROOT / "configs" / "mac.example.json")
    windows = load_config(ROOT / "configs" / "windows.example.json")
    assert MODEL_NAME == "Qwen/Qwen3-VL-8B-Instruct"
    assert "GGUF" not in MODEL_NAME
    assert CHECKPOINT_FORMAT == "safetensors"
    assert NUM_LAYERS == 36
    assert (HIDDEN_SIZE, N_HEADS, N_KV_HEADS, HEAD_DIM) == (4096, 32, 8, 128)
    assert (INTERMEDIATE_SIZE, VOCAB_SIZE, EOS_TOKEN_ID) == (12288, 151936, 151645)
    assert VISION_OWNER == "mac"
    assert VISION_DEPTH == 27
    assert VISION_DEEPSTACK_INDEXES == (8, 16, 24)
    assert mac["layer_range"] == [0, 18]
    assert windows["layer_range"] == [18, 36]
    assert mac["num_layers"] == windows["num_layers"] == 36
    assert mac["model"] == windows["model"] == MODEL_NAME
    assert mac["checkpoint_format"] == windows["checkpoint_format"] == "safetensors"
    assert mac["vision_owner"] == windows["vision_owner"] == "mac"
    assert mac["vision_depth"] == 27
    assert mac["vision_deepstack_indexes"] == [8, 16, 24]
    assert mac["encryption"] is False and windows["encryption"] is False
    with pytest.raises(ValueError):
        validate_config({**mac, "model": "Qwen/Qwen2.5-32B-Instruct"})
    with pytest.raises(ValueError):
        validate_config({**mac, "model": "Qwen/Qwen3-VL-8B-Instruct-GGUF"})
    with pytest.raises(ValueError):
        validate_config({**mac, "checkpoint_format": "gguf"})
    with pytest.raises(ValueError):
        validate_config({**mac, "layer_range": [0, 36]})
    with pytest.raises(ValueError):
        validate_config({**windows, "layer_range": [14, 14 + 14]})
    with pytest.raises(ValueError):
        validate_config({**mac, "num_layers": 14 * 2})
    with pytest.raises(ValueError):
        validate_config({**mac, "vision_owner": "windows"})
    with pytest.raises(ValueError):
        validate_config({**mac, "vision_depth": 32})
    with pytest.raises(ValueError):
        validate_config({**mac, "encryption": True})
    with pytest.raises(ValueError):
        validate_config({**windows, "engine": "vllm-metal"})
    with pytest.raises(ValueError):
        validate_config({**windows, "engine": "llama.cpp"})


def test_repository_has_no_stale_7b_split():
    stale = (
        "Qwen2.5-" + "7B",
        "[0, " + "14)",
        "[14, " + "28)",
        "[0, " + "14]",
        "[14, " + "28]",
        "28-" + "layer",
        "28 " + "层",
        "must be " + "28",
        "Qwen/" + "Qwen2.5-" + "7B-Instruct",
    )
    allowed_suffixes = {".py", ".md", ".json", ".html", ".js", ".toml", ".css"}
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".venv", "venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.suffix not in allowed_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for item in stale:
            if item in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {item}")
    assert offenders == []


def test_float32_roundtrip():
    blob = as_blob([[1.5, -2.0], [3.25, 4.0]])
    assert blob.dtype == "float32"
    assert blob.nested()[0][1] == pytest.approx(-2.0)
    assert blob.nested()[1][0] == pytest.approx(3.25)


def test_nodes_refuse_full_depth():
    with pytest.raises(LayerRangeError):
        LayerNode("full", 0, 36, runner=None, lm_head=None)
    mac = HiddenStateRunner("mac")
    with pytest.raises(LayerRangeError):
        mac.forward(as_blob([[1.0, 2.0]]), 0, 36, None, None, 0)
    windows = HiddenStateRunner("windows")
    with pytest.raises(LayerRangeError):
        windows.embed([1])
    with pytest.raises(LayerRangeError):
        windows.apply(as_blob([[1.0]]))


def test_real_engines_fail_closed_without_hardware():
    metal = VllmMetalHiddenStateRunner()
    cuda = VllmCudaHiddenStateRunner()
    assert (metal.layer_start, metal.layer_end) == (0, 18)
    assert (cuda.layer_start, cuda.layer_end) == (18, 36)
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
        "layer_start": 18,
        "layer_end": 36,
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
