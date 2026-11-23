"""Windows HiddenStateRunner for CUDA vLLM.

The runner loads the engine, keeps language layers [18, 36), and refuses
embeddings, lm_head, the vision tower, and tokenization. The expected host is
WSL2 with an NVIDIA GPU.
"""

from __future__ import annotations

from pool.errors import EngineUnavailable, LayerRangeError
from pool.protocol.constants import MODEL_NAME, NUM_LAYERS, WINDOWS_LAYER_END, WINDOWS_LAYER_START
from pool.runtime.engine_load import construct_engine
from pool.runtime.extract import drop_vision_module, find_vision_module
from pool.runtime.offload import choose_cuda_offload, probe_cuda_free_bytes
from pool.runtime.runner import HiddenStateRunner


class VllmCudaHiddenStateRunner(HiddenStateRunner):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_model_len: int = 512,
        *,
        offload: str = "auto",
        offload_cap_gb: float | None = None,
        vram_probe=None,
    ):
        super().__init__(role="windows")
        if (self.layer_start, self.layer_end) != (WINDOWS_LAYER_START, WINDOWS_LAYER_END):
            raise LayerRangeError(
                "CUDA runner default shard is the reference even split; "
                "call assign_range before load to bind a scheduled hop"
            )
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.offload_policy = offload
        self.offload_cap_gb = offload_cap_gb
        self.vram_probe = vram_probe or probe_cuda_free_bytes
        self.offload_plan = None
        self.engine = None

    def assign_range(self, layer_start: int, layer_end: int) -> None:
        if layer_end <= layer_start or layer_start < 0 or layer_end > NUM_LAYERS:
            raise LayerRangeError("CUDA shard is outside the language tower")
        if layer_start == 0 and layer_end >= NUM_LAYERS:
            raise LayerRangeError("CUDA runner refuses a full-tower engine call")
        self.layer_start = layer_start
        self.layer_end = layer_end

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
        free = int(self.vram_probe())
        shard = self.layer_end - self.layer_start
        self.offload_plan = choose_cuda_offload(
            free,
            shard,
            policy=self.offload_policy,
            offload_cap_gb=self.offload_cap_gb,
        )
        self.engine = construct_engine(
            self.model_name,
            self.max_model_len,
            cpu_offload_gb=self.offload_plan.cpu_offload_gb,
            swap_space=self.offload_plan.swap_space_gb,
        )
        self.bind_module(self.engine)
        drop_vision_module(self.engine)
        if find_vision_module(self.engine) is not None:
            raise EngineUnavailable("Windows runner must not keep the vision encoder")
        if self.embed_table is not None or self.lm_weight is not None:
            raise EngineUnavailable("Windows runner must not keep embeddings or lm_head")
        if self.layers is None or len(self.layers) != shard:
            raise EngineUnavailable(
                f"CUDA vLLM did not expose the scheduled shard [{self.layer_start}, {self.layer_end})"
            )
        return self
