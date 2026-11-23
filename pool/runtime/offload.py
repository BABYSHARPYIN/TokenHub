"""Decide when a CUDA shard must spill weights and KV to CPU memory.

The locked checkpoint stays Qwen/Qwen3-VL-8B-Instruct. Offload never quantizes
the model, swaps the repo, or drops language layers. vLLM receives the decision
as ``cpu_offload_gb`` (weights) and ``swap_space`` (KV). Apple Silicon has no
discrete VRAM split, so that path records a no-op.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pool.errors import EngineUnavailable
from pool.protocol.constants import SPEC

LOG = logging.getLogger("pool.offload")
GIB = 1024 ** 3
# Scratch that must stay on the GPU beside the resident layer: activations,
# allocator slack, and the CUDA context. One language layer of weights is added
# on top of this when checking that a shard can run at all.
WORKING_SET_BYTES = 2 * GIB


def language_layer_weight_bytes(spec=SPEC, dtype_bytes: int = 2) -> int:
    """bf16 Q/K/V/O + SwiGLU bytes for one language-tower block. Not the vision tower."""
    hidden = spec.hidden_size
    query = spec.n_heads * spec.head_dim
    key = spec.n_kv_heads * spec.head_dim
    params = (
        hidden * query
        + 2 * hidden * key
        + query * hidden
        + 3 * spec.intermediate * hidden
        + 2 * hidden
    )
    return params * dtype_bytes


def kv_bytes_for_layers(num_layers: int, *, max_model_len: int = 512, spec=SPEC, dtype_bytes: int = 2) -> int:
    per_token = 2 * spec.n_kv_heads * spec.head_dim * dtype_bytes
    return num_layers * max_model_len * per_token


LAYER_WEIGHT_BYTES = language_layer_weight_bytes()


@dataclass(frozen=True)
class OffloadPlan:
    cpu_offload_gb: float
    swap_space_gb: float
    mode: str
    detail: str

    @property
    def enabled(self) -> bool:
        return self.mode == "memory-offload"


def _gb(num_bytes: int) -> float:
    if num_bytes <= 0:
        return 0.0
    return num_bytes / GIB


def plan_offload(
    free_vram_bytes: int,
    shard_layers: int,
    *,
    policy: str = "auto",
    offload_cap_bytes: int | None = None,
    swap_space_bytes: int | None = None,
    layer_weight_bytes: int | None = None,
    working_set_bytes: int | None = None,
    kv_bytes: int | None = None,
) -> OffloadPlan:
    """Choose GPU-resident bytes versus CPU offload for one language shard.

    ``policy`` is ``auto`` (default), ``force``, or ``disable``. ``auto`` spills
    only the bytes that do not fit. ``force`` keeps a single layer on the GPU
    and spills the rest. ``disable`` fails closed instead of spilling.
    """
    if policy not in {"auto", "force", "disable"}:
        raise ValueError("offload policy must be auto, force, or disable")
    if shard_layers < 1:
        raise EngineUnavailable("shard has no language layers to run")
    layer = LAYER_WEIGHT_BYTES if layer_weight_bytes is None else layer_weight_bytes
    working = WORKING_SET_BYTES if working_set_bytes is None else working_set_bytes
    kv = kv_bytes_for_layers(shard_layers) if kv_bytes is None else kv_bytes
    if layer < 1 or working < 1:
        raise EngineUnavailable("offload planner was given an empty working set")
    minimum = layer + working
    if free_vram_bytes < minimum:
        raise EngineUnavailable(
            "cannot fit one language layer plus a minimal working set in GPU VRAM; "
            "CPU offload cannot run this shard"
        )
    weights = shard_layers * layer
    weight_budget = max(0, free_vram_bytes - working)
    overflow = max(0, weights - weight_budget)
    # At least one layer stays on the GPU so the forward has something to compute.
    max_spill = max(0, weights - layer)
    spill = min(overflow, max_spill)
    if policy == "force":
        spill = max_spill
    resident = weights - spill
    if resident + working > free_vram_bytes:
        raise EngineUnavailable(
            "cannot fit one language layer plus a minimal working set in GPU VRAM; "
            "CPU offload cannot run this shard"
        )
    remaining = free_vram_bytes - resident - working
    if kv <= remaining:
        swap = 0 if swap_space_bytes is None else swap_space_bytes
    else:
        swap = kv if swap_space_bytes is None else swap_space_bytes
        if swap < kv:
            raise EngineUnavailable(
                "KV does not fit in the remaining GPU memory and swap_space is below the KV size"
            )
    if policy == "disable" and (spill > 0 or kv > remaining):
        raise EngineUnavailable(
            "language shard does not fit in GPU VRAM and auto-offload is disabled"
        )
    if offload_cap_bytes is not None and spill > offload_cap_bytes:
        raise EngineUnavailable(
            "cpu_offload_gb cap is smaller than the weight bytes that do not fit in GPU VRAM"
        )
    mode = "memory-offload" if spill > 0 or (swap > 0 and kv > remaining) else "gpu"
    if mode == "memory-offload":
        detail = (
            "memory-offload mode: weights or KV did not fit in GPU VRAM; "
            "this node will be slower"
        )
    else:
        detail = "language shard fits in GPU VRAM"
    return OffloadPlan(
        cpu_offload_gb=_gb(spill),
        swap_space_gb=_gb(swap),
        mode=mode,
        detail=detail,
    )


def log_offload_plan(plan: OffloadPlan, *, engine: str) -> None:
    if plan.mode == "unified-memory":
        LOG.info("%s", plan.detail)
        return
    if plan.mode == "memory-offload":
        LOG.warning(
            "memory-offload mode on %s: cpu_offload_gb=%.3f swap_space_gb=%.3f; this node will be slower",
            engine,
            plan.cpu_offload_gb,
            plan.swap_space_gb,
        )
        return
    LOG.info("%s on %s (cpu_offload_gb=0)", plan.detail, engine)


def skip_metal_offload() -> OffloadPlan:
    """vllm-metal shares CPU and GPU memory. There is no VRAM-to-RAM downgrade."""
    plan = OffloadPlan(
        cpu_offload_gb=0.0,
        swap_space_gb=0.0,
        mode="unified-memory",
        detail="vllm-metal uses unified memory; skipping discrete VRAM offload",
    )
    log_offload_plan(plan, engine="vllm-metal")
    return plan


def choose_cuda_offload(
    free_vram_bytes: int,
    shard_layers: int,
    *,
    policy: str = "auto",
    offload_cap_gb: float | None = None,
    swap_space_gb: float | None = None,
) -> OffloadPlan:
    """Planner entry used by the CUDA adapter. Tests pass a mocked VRAM probe."""
    cap = None if offload_cap_gb is None else int(offload_cap_gb * GIB)
    swap = None if swap_space_gb is None else int(swap_space_gb * GIB)
    plan = plan_offload(
        free_vram_bytes,
        shard_layers,
        policy=policy,
        offload_cap_bytes=cap,
        swap_space_bytes=swap,
    )
    log_offload_plan(plan, engine="vllm-cuda")
    return plan


def probe_cuda_free_bytes() -> int:
    """Read free VRAM from the current CUDA device. Not used by the mock suite."""
    try:
        import torch
    except Exception as exc:
        raise EngineUnavailable("CUDA VRAM probe requires torch") from exc
    if not torch.cuda.is_available():
        raise EngineUnavailable("CUDA device is not available")
    free, _total = torch.cuda.mem_get_info()
    return int(free)
