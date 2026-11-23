"""Locked v1 model and layer split.

The checkpoint is the Hugging Face safetensors repo
Qwen/Qwen3-VL-8B-Instruct (Qwen3VLForConditionalGeneration).
It is not Qwen/Qwen3-VL-8B-Instruct-GGUF and it is not a llama.cpp build.

The language tower has 36 layers. [0, 18) / [18, 36) is the reference
even split for two similar nodes, not the only legal assignment.
The vision tower is a 27-depth ViT with DeepStack indexes 8/16/24.
Vision stays on Mac with the tokenizer, embeddings, lm_head, and plaintext.
"""

from dataclasses import dataclass

PROTOCOL = "pool.v1"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
CHECKPOINT_FORMAT = "safetensors"
GGUF_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct-GGUF"

# Language tower (text_config). This is not the vision depth.
NUM_LAYERS = 36
HIDDEN_SIZE = 4096
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 12288
VOCAB_SIZE = 151936
RMS_NORM_EPS = 1e-6
ROPE_THETA = 5_000_000.0
MAX_POSITION_EMBEDDINGS = 262144
EOS_TOKEN_ID = 151645
BOS_TOKEN_ID = 151643

MAC_LAYER_START = 0
MAC_LAYER_END = NUM_LAYERS // 2
WINDOWS_LAYER_START = NUM_LAYERS // 2
WINDOWS_LAYER_END = NUM_LAYERS

# Vision tower. Execution and weights stay on Mac in every mode.
VISION_OWNER = "mac"
VISION_DEPTH = 27
VISION_DEEPSTACK_INDEXES = (8, 16, 24)
VISION_HIDDEN_SIZE = 1152
VISION_NUM_HEADS = 16
VISION_INTERMEDIATE_SIZE = 4304
VISION_OUT_HIDDEN_SIZE = 4096
VISION_PATCH_SIZE = 16
VISION_SPATIAL_MERGE_SIZE = 2
VISION_TEMPORAL_PATCH_SIZE = 2
VISION_MODULE_ATTRS = ("visual", "vision_tower", "vision_model")

DEGRADED_FRONT_HALF = "mac_front_half"
HEALTH_TIMEOUT_S = 3.0

FRONT_HALF_WARNING = (
    "Front-half smoke (mac_front_half): Mac ran only language-tower layers [0, 18) "
    "for prefill and decode, then sampled with lm_head on those front-half hidden states. "
    "Layers [18, 36) did not run. This is not a full-model chat completion. "
    "Output quality does not count. "
    "前半层冒烟：只跑了 Mac 语言塔层 [0, 18)，没有跑 Windows 后半层 [18, 36)，输出质量不计。"
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
    if SPEC.name != "Qwen/Qwen3-VL-8B-Instruct":
        raise RuntimeError("v1 model is frozen to Qwen/Qwen3-VL-8B-Instruct")
    if "GGUF" in SPEC.name or SPEC.name == GGUF_MODEL_NAME:
        raise RuntimeError("v1 refuses the GGUF checkpoint")
    if CHECKPOINT_FORMAT != "safetensors":
        raise RuntimeError("v1 checkpoint is Hugging Face safetensors")
    if GGUF_MODEL_NAME == MODEL_NAME:
        raise RuntimeError("the GGUF repo must stay distinct from the locked checkpoint")
    if SPEC.num_layers != 36:
        raise RuntimeError("v1 language tower is 36 layers")
    if SPEC.hidden_size != 4096 or HIDDEN_SIZE != 4096:
        raise RuntimeError("language hidden_size is frozen to 4096")
    if SPEC.n_heads != 32 or N_HEADS != 32:
        raise RuntimeError("language num_attention_heads is frozen to 32")
    if SPEC.n_kv_heads != 8 or N_KV_HEADS != 8:
        raise RuntimeError("language num_key_value_heads is frozen to 8")
    if SPEC.head_dim != 128 or HEAD_DIM != 128:
        raise RuntimeError("language head_dim is frozen to 128")
    if SPEC.intermediate != 12288 or INTERMEDIATE_SIZE != 12288:
        raise RuntimeError("language intermediate_size is frozen to 12288")
    if SPEC.vocab_size != 151936 or VOCAB_SIZE != 151936:
        raise RuntimeError("language vocab_size is frozen to 151936")
    if SPEC.eos_token_id != 151645 or EOS_TOKEN_ID != 151645:
        raise RuntimeError("eos_token_id is frozen to 151645")
    if SPEC.rope_theta != 5_000_000.0 or ROPE_THETA != 5_000_000.0:
        raise RuntimeError("language rope_theta is frozen to 5000000")
    if MAX_POSITION_EMBEDDINGS != 262144:
        raise RuntimeError("language max_position_embeddings is frozen to 262144")
    if (MAC_LAYER_START, MAC_LAYER_END) != (0, 18):
        raise RuntimeError("Mac language layers are frozen to [0, 18)")
    if (WINDOWS_LAYER_START, WINDOWS_LAYER_END) != (18, 36):
        raise RuntimeError("Windows language layers are frozen to [18, 36)")
    if MAC_LAYER_END != WINDOWS_LAYER_START:
        raise RuntimeError("the v1 cut is contiguous")
    if WINDOWS_LAYER_END != SPEC.num_layers:
        raise RuntimeError("Windows must end at the last language layer")
    if MAC_LAYER_END - MAC_LAYER_START != NUM_LAYERS // 2:
        raise RuntimeError("Mac must own an even front half of the language tower")
    if WINDOWS_LAYER_END - WINDOWS_LAYER_START != NUM_LAYERS // 2:
        raise RuntimeError("Windows must own an even back half of the language tower")
    if NUM_LAYERS % 2 != 0:
        raise RuntimeError("the language tower must split evenly")
    if VISION_OWNER != "mac":
        raise RuntimeError("vision encoder stays on Mac")
    if VISION_DEPTH != 27:
        raise RuntimeError("vision tower depth is frozen to 27")
    if VISION_DEEPSTACK_INDEXES != (8, 16, 24):
        raise RuntimeError("vision DeepStack indexes are frozen to 8/16/24")
    if any(index < 0 or index >= VISION_DEPTH for index in VISION_DEEPSTACK_INDEXES):
        raise RuntimeError("DeepStack indexes must point into the vision tower")
    if VISION_DEPTH == NUM_LAYERS:
        raise RuntimeError("vision depth must not be confused with the language tower")
    if VISION_HIDDEN_SIZE != 1152 or VISION_NUM_HEADS != 16:
        raise RuntimeError("vision width is frozen to the official ViT config")
    if VISION_INTERMEDIATE_SIZE != 4304 or VISION_OUT_HIDDEN_SIZE != 4096:
        raise RuntimeError("vision projection widths are frozen to the official config")
