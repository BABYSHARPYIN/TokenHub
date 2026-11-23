"""Locked v1 model and layer split.

Do not switch this file to an 8B or 32B model, a 32-layer stack, or an uneven cut.
Mac always owns [0, 14). Windows always owns [14, 28).
"""

from dataclasses import dataclass

PROTOCOL = "pool.v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_LAYERS = 28
HIDDEN_SIZE = 3584
N_HEADS = 28
N_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE_SIZE = 18944
VOCAB_SIZE = 152064
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1_000_000.0
MAX_POSITION_EMBEDDINGS = 32768
EOS_TOKEN_ID = 151645
BOS_TOKEN_ID = 151643

MAC_LAYER_START = 0
MAC_LAYER_END = 14
WINDOWS_LAYER_START = 14
WINDOWS_LAYER_END = 28

DEGRADED_FRONT_HALF = "mac_front_half"
HEALTH_TIMEOUT_S = 3.0

FRONT_HALF_WARNING = (
    "Front-half smoke (mac_front_half): Mac ran only layers [0, 14) for prefill and decode, "
    "then sampled with lm_head on those front-half hidden states. "
    "Layers [14, 28) did not run. This is not a full-model chat completion. "
    "Output quality does not count. "
    "前半层冒烟：只跑了 Mac 层 [0, 14)，没有跑 Windows 后半层，输出质量不计。"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str = MODEL_NAME
    num_layers: int = NUM_LAYERS
    hidden_size: int = HIDDEN_SIZE
    n_heads: int = N_HEADS
    n_kv_heads: int = N_KV_HEADS
    head_dim: int = HEAD_DIM
    intermediate: int = INTERMEDIATE_SIZE
    vocab_size: int = VOCAB_SIZE
    rms_norm_eps: float = RMS_NORM_EPS
    rope_theta: float = ROPE_THETA
    eos_token_id: int = EOS_TOKEN_ID


SPEC = ModelSpec()


def assert_locked_split() -> None:
    if SPEC.name != "Qwen/Qwen2.5-7B-Instruct":
        raise RuntimeError("v1 model is frozen to Qwen2.5-7B-Instruct")
    if SPEC.num_layers != 28:
        raise RuntimeError("v1 is a 28-layer model")
    if (MAC_LAYER_START, MAC_LAYER_END) != (0, 14):
        raise RuntimeError("Mac layers are frozen to [0, 14)")
    if (WINDOWS_LAYER_START, WINDOWS_LAYER_END) != (14, 28):
        raise RuntimeError("Windows layers are frozen to [14, 28)")
    if MAC_LAYER_END != WINDOWS_LAYER_START:
        raise RuntimeError("the v1 cut is contiguous")
    if WINDOWS_LAYER_END != SPEC.num_layers:
        raise RuntimeError("Windows must end at the last layer")
