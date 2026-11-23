"""共享分层推理计算池 (package name: pool).

v1 splits Qwen2.5-7B-Instruct into Mac layers [0, 14) and Windows layers [14, 28).
Hidden states cross the public internet in the clear. There is no encryption,
billing, or multi-tenant auth.
"""

from pool.protocol.constants import MODEL_NAME, PROTOCOL

__version__ = "0.1.0"
__all__ = ["MODEL_NAME", "PROTOCOL", "__version__"]
