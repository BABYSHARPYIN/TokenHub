"""Deterministic stand-ins for embed, layer ranges, and lm_head.

Mock math keeps the protocol honest in CI. It does not imitate a full-model reply
by calling a chat API, and it does not execute layers outside the node's range.
"""

from __future__ import annotations

from pool.protocol.codec import TensorBlob, as_blob


class MockEmbedder:
    """Mac-only. Token ids never leave this object."""

    def __init__(self, hidden_size: int = 8):
        self.hidden_size = hidden_size

    def embed(self, token_ids: list[int]) -> TensorBlob:
        rows = []
        for token in token_ids:
            rows.append([((token + 1) * (index + 3) % 97) / 97.0 for index in range(self.hidden_size)])
        return as_blob(rows)


class MockRunner:
    """Adds a range-dependent shift and emits one KV vector per layer and token."""

    def __init__(self, hidden_size: int = 8, kv_dim: int = 4):
        self.hidden_size = hidden_size
        self.kv_dim = kv_dim
        self.calls: list[tuple[int, int]] = []

    def forward(
        self,
        hidden: TensorBlob,
        layer_start: int,
        layer_end: int,
        past_keys: TensorBlob | None,
        past_values: TensorBlob | None,
        position_start: int,
    ) -> tuple[TensorBlob, TensorBlob, TensorBlob]:
        del past_keys, past_values
        self.calls.append((layer_start, layer_end))
        if layer_start == 0 and layer_end >= 36:
            raise RuntimeError("mock engine refuses a full-tower forward")
        rows = hidden.nested()
        shift = (layer_end - layer_start) / 1000.0 + layer_start / 10000.0
        out = [[value + shift for value in row] for row in rows]
        n_layers = layer_end - layer_start
        seq = len(rows)
        keys = [
            [[shift + position_start + seq_index + layer_offset for _ in range(self.kv_dim)] for seq_index in range(seq)]
            for layer_offset in range(n_layers)
        ]
        values = [
            [[row[0] * 0.5 for _ in range(self.kv_dim)] for row in rows]
            for _ in range(n_layers)
        ]
        return as_blob(out), as_blob(keys), as_blob(values)


class MockLmHead:
    """Projects whatever hidden state it is given. Vocab is the mock byte vocab, not Qwen's."""

    def __init__(self, vocab_size: int = 257):
        self.vocab_size = vocab_size
        self.inputs: list[TensorBlob] = []

    def apply(self, hidden: TensorBlob) -> TensorBlob:
        self.inputs.append(hidden)
        flat = hidden.flat()
        checksum = sum(flat)
        letter = (int(abs(checksum) * 1000) + len(self.inputs)) % 26
        token = 97 + letter
        logits = [0.0] * self.vocab_size
        logits[token] = 10.0
        return as_blob(logits)
