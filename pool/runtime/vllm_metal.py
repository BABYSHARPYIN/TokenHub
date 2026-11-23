"""Mac HiddenStateRunner for vllm-metal.

vLLM supplies the loader and scheduler objects. MLX/Metal supplies the tensors.
This adapter exports language layers [0, 18), embeddings, the final norm, and
lm_head, then runs that range on hidden states. The vision tower stays on this
process. The adapter never calls a chat-completions API.
"""

from __future__ import annotations

from pool.errors import EngineUnavailable, LayerRangeError
from pool.protocol.constants import MAC_LAYER_END, MAC_LAYER_START, MODEL_NAME
from pool.runtime.engine_load import construct_engine
from pool.runtime.extract import find_vision_module
from pool.runtime.offload import skip_metal_offload
from pool.runtime.runner import HiddenStateRunner


class VllmMetalHiddenStateRunner(HiddenStateRunner):
    def __init__(self, model_name: str = MODEL_NAME, max_model_len: int = 512):
        super().__init__(role="mac")
        if (self.layer_start, self.layer_end) != (MAC_LAYER_START, MAC_LAYER_END):
            raise LayerRangeError(
                f"Mac vllm-metal runner is locked to layers [{MAC_LAYER_START}, {MAC_LAYER_END})"
            )
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.engine = None
        self.vision = None
        self.offload_plan = None

    def load(self) -> "VllmMetalHiddenStateRunner":
        self.offload_plan = skip_metal_offload()
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
        self.vision = find_vision_module(self.engine)
        half = MAC_LAYER_END - MAC_LAYER_START
        if self.layers is None or len(self.layers) != half:
            raise EngineUnavailable(
                f"vllm-metal did not expose the Mac layer range [{MAC_LAYER_START}, {MAC_LAYER_END})"
            )
        return self
