"""Construct a vLLM engine object and seal its text entrypoints.

HiddenStateRunner calls decoder modules after this returns. It does not call
the engine's text generation entrypoints, including any OpenAI-compatible route.
"""

from __future__ import annotations

import os

from pool.errors import EngineUnavailable


def prepare_inprocess_engine() -> None:
    # Keep the worker in-process so layer modules can be hooked. A spawned
    # worker would hide the weights behind a process boundary.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def construct_engine(
    model_name: str,
    max_model_len: int,
    *,
    cpu_offload_gb: float = 0.0,
    swap_space: float | None = None,
):
    prepare_inprocess_engine()
    try:
        from vllm import LLM
    except Exception as exc:
        raise EngineUnavailable("vLLM is not installed") from exc
    offload = {}
    if cpu_offload_gb and cpu_offload_gb > 0:
        offload["cpu_offload_gb"] = cpu_offload_gb
    if swap_space and swap_space > 0:
        offload["swap_space"] = swap_space
    attempts = [
        {
            "model": model_name,
            "skip_tokenizer_init": True,
            "enforce_eager": True,
            "max_model_len": max_model_len,
            "trust_remote_code": True,
            **offload,
        },
        {
            "model": model_name,
            "skip_tokenizer_init": True,
            "enforce_eager": True,
            "trust_remote_code": True,
            **offload,
        },
        {"model": model_name, "enforce_eager": True, "trust_remote_code": True, **offload},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            engine = LLM(**kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        seal_text_entrypoints(engine)
        return engine
    if offload:
        raise EngineUnavailable(
            "vLLM rejected cpu_offload_gb or swap_space; refusing to load the shard without offload"
        )
    raise EngineUnavailable(f"unable to construct vLLM: {last_error}")


def seal_text_entrypoints(engine) -> None:
    def banned(*_args, **_kwargs):
        raise EngineUnavailable("pool does not call the engine text entrypoints")

    for name in ("generate", "chat", "encode"):
        try:
            setattr(engine, name, banned)
        except Exception:
            continue
