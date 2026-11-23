"""Frozen v1 activation protocol. Payloads are cleartext JSON plus base64 float32.

The locked checkpoint is Qwen/Qwen3-VL-8B-Instruct safetensors.
Language layers split Mac [0, 18) / Windows [18, 36). Vision stays on Mac.
"""

from pool.protocol.constants import MODEL_NAME, PROTOCOL, SPEC

__all__ = ["MODEL_NAME", "PROTOCOL", "SPEC"]
