"""Tokenizers stay on Mac. Windows never imports this module."""

from __future__ import annotations

from pool.errors import EngineUnavailable


class ByteTokenizer:
    """UTF-8 stand-in used by the mock engine. Not the Qwen vocabulary."""

    eos_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        body = bytes(token for token in token_ids if 0 <= token < 256)
        return body.decode("utf-8", errors="replace")


class HuggingFaceTokenizer:
    """Real Qwen tokenizer. Loaded only for the vllm-metal Mac path."""

    def __init__(self, model_name: str):
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise EngineUnavailable(
                "the Mac vllm-metal path needs transformers to tokenize on Mac"
            ) from exc
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.eos_token_id is None:
            raise EngineUnavailable("tokenizer has no eos id")
        self.eos_id = int(self.tok.eos_token_id)

    def encode(self, text: str) -> list[int]:
        return [int(token) for token in self.tok.encode(text, add_special_tokens=True)]

    def decode(self, token_ids: list[int]) -> str:
        return self.tok.decode(token_ids, skip_special_tokens=True)
