"""Windows HiddenStateRunner for CUDA vLLM.

The runner loads the engine, keeps layers [14, 28), and refuses embeddings,
lm_head, and tokenization. The expected host is WSL2 with an NVIDIA GPU.
"""

from __future__ import annotations

from pool.errors import EngineUnavailable, LayerRangeError
from pool.protocol.constants import MODEL_NAME
from pool.runtime.engine_load import construct_engine
from pool.runtime.runner import HiddenStateRunner


class VllmCudaHiddenStateRunner(HiddenStateRunner):
    def __init__(self, model_name: str = MODEL_NAME, max_model_len: int = 512):
        super().__init__(role="windows")
        if (self.layer_start, self.layer_end) != (14, 28):
            raise LayerRangeError("Windows CUDA runner is locked to layers [14, 28)")
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
        if self.embed_table is not None or self.lm_weight is not None:
            raise EngineUnavailable("Windows runner must not keep embeddings or lm_head")
        if self.layers is None or len(self.layers) != 14:
            raise EngineUnavailable("CUDA vLLM did not expose Windows layers [14, 28)")
        return self
