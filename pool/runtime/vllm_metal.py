"""Mac HiddenStateRunner for vllm-metal.

vLLM supplies the loader and scheduler objects. MLX/Metal supplies the tensors.
This adapter exports layers [0, 14), embeddings, the final norm, and lm_head,
then runs that range on hidden states. It never calls a chat-completions API.
"""

from __future__ import annotations

from pool.errors import EngineUnavailable, LayerRangeError
from pool.protocol.constants import MODEL_NAME
from pool.runtime.engine_load import construct_engine
from pool.runtime.runner import HiddenStateRunner


class VllmMetalHiddenStateRunner(HiddenStateRunner):
    def __init__(self, model_name: str = MODEL_NAME, max_model_len: int = 512):
        super().__init__(role="mac")
        if (self.layer_start, self.layer_end) != (0, 14):
            raise LayerRangeError("Mac vllm-metal runner is locked to layers [0, 14)")
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.engine = None

    def load(self) -> "VllmMetalHiddenStateRunner":
        try:
            import mlx.core  # noqa: F401
        except Exception as exc:
            raise EngineUnavailable(
                "vllm-metal requires the mlx package on macOS 15 or newer"
            ) from exc
        try:
            import vllm  # noqa: F401
        except Exception as exc:
            raise EngineUnavailable("vllm-metal requires the vllm package") from exc
        self.engine = construct_engine(self.model_name, self.max_model_len)
        self.bind_module(self.engine)
        if self.layers is None or len(self.layers) != 14:
            raise EngineUnavailable("vllm-metal did not expose the Mac layer range [0, 14)")
        return self
