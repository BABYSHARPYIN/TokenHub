"""TokenHub. The import package stays `pool`; the distribution name is `tokenhub`.

This process is a 共享分层推理计算池.

v1 splits Qwen/Qwen3-VL-8B-Instruct language layers into Mac [0, 18) and Windows [18, 36).
The vision tower stays on Mac. The checkpoint is Hugging Face safetensors.
Hidden states cross the public internet in the clear. There is no encryption,
billing, or multi-tenant auth.
"""

from pool.protocol.constants import MODEL_NAME, PROTOCOL

__version__ = "0.1.0"
__all__ = ["MODEL_NAME", "PROTOCOL", "__version__"]
