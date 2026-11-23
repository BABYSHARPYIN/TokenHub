"""Windows HiddenStateRunner for CUDA vLLM.

The runner loads the engine, keeps language layers [18, 36), and refuses
embeddings, lm_head, the vision tower, and tokenization. The expected host is
WSL2 with an NVIDIA GPU.
"""

from __future__ import annotations

from pool.errors import EngineUnavailable, LayerRangeError
from pool.protocol.constants import MODEL_NAME, WINDOWS_LAYER_END, WINDOWS_LAYER_START
from pool.runtime.engine_load import construct_engine
from pool.runtime.extract import drop_vision_module, find_vision_module
from pool.runtime.runner import HiddenStateRunner


class VllmCudaHiddenStateRunner(HiddenStateRunner):
    def __init__(self, model_name: str = MODEL_NAME, max_model_len: int = 512):
        super().__init__(role="windows")
        if (self.layer_start, self.layer_end) != (WINDOWS_LAYER_START, WINDOWS_LAYER_END):
            raise LayerRangeError(
                f"Windows CUDA runner is locked to layers [{WINDOWS_LAYER_START}, {WINDOWS_LAYER_END})"
            )
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.engine = None

    def load(self) -> "VllmCudaHiddenStateRunner":
        try:
            import torch
        except Exception as exc:
            raise EngineUnavailable("CUDA vLLM requires torch inside WSL2") from exc
        try:
            import vllm  # noqa: F401
        except Exception as exc:
            raise EngineUnavailable("CUDA vLLM requires the vllm package") from exc
        if not torch.cuda.is_available():
            raise EngineUnavailable("CUDA device is not available")
        self.engine = construct_engine(self.model_name, self.max_model_len)
        self.bind_module(self.engine)
        drop_vision_module(self.engine)
        if find_vision_module(self.engine) is not None:
            raise EngineUnavailable("Windows runner must not keep the vision encoder")
        if self.embed_table is not None or self.lm_weight is not None:
            raise EngineUnavailable("Windows runner must not keep embeddings or lm_head")
        half = WINDOWS_LAYER_END - WINDOWS_LAYER_START
        if self.layers is None or len(self.layers) != half:
            raise EngineUnavailable(
                f"CUDA vLLM did not expose Windows layers [{WINDOWS_LAYER_START}, {WINDOWS_LAYER_END})"
            )
        return self
