"""HiddenStateRunner: execute a locked layer range and, on Mac, embed and project."""

from __future__ import annotations

from pool.errors import EngineUnavailable, LayerRangeError
from pool.protocol.codec import TensorBlob, as_blob
from pool.protocol.constants import (
    MAC_LAYER_END,
    MAC_LAYER_START,
    WINDOWS_LAYER_END,
    WINDOWS_LAYER_START,
    ModelSpec,
)
from pool.runtime.block import apply_lm_head_list, forward_layers_list
from pool.runtime.extract import export_head, export_range
from pool.runtime.native import apply_lm_head_mlx, apply_lm_head_torch, forward_layers_mlx, forward_layers_torch


class HiddenStateRunner:
    """Hooks model internals for one half of the 28-layer stack.

    `role="mac"` is layers [0, 14) plus embeddings and lm_head.
    `role="windows"` is layers [14, 28) and refuses embed, logits, and text.
    """

    def __init__(self, role: str, spec: ModelSpec | None = None):
        if role not in {"mac", "windows"}:
            raise LayerRangeError("role must be mac or windows")
        self.role = role
        self.spec = spec or ModelSpec()
        if role == "mac":
            self.layer_start, self.layer_end = MAC_LAYER_START, MAC_LAYER_END
        else:
            self.layer_start, self.layer_end = WINDOWS_LAYER_START, WINDOWS_LAYER_END
        self.layers = None
        self.embed_table = None
        self.final_norm = None
        self.lm_weight = None

    def bind_module(self, module) -> None:
        self.layers = export_range(module, self.layer_start, self.layer_end, self.spec)
        if self.role == "mac":
            self.embed_table, self.final_norm, self.lm_weight = export_head(module, self.spec)
        elif any(hasattr(module, name) and False for name in ()):
            raise EngineUnavailable("Windows runner does not bind a tokenizer")

    def bind_weights(self, layers, embed=None, final_norm=None, lm_weight=None) -> None:
        if len(layers) != self.layer_end - self.layer_start:
            raise LayerRangeError("weight list does not match the locked layer range")
        self.layers = list(layers)
        if self.role == "windows":
            if embed is not None or lm_weight is not None:
                raise LayerRangeError("Windows runner cannot hold embeddings or lm_head")
            return
        if embed is None or final_norm is None or lm_weight is None:
            raise EngineUnavailable("Mac runner requires embeddings, final norm, and lm_head")
        self.embed_table = embed
        self.final_norm = final_norm
        self.lm_weight = lm_weight

    def embed(self, token_ids: list[int]) -> TensorBlob:
        if self.role != "mac":
            raise LayerRangeError("Windows runner cannot embed token ids")
        table = self.embed_table
        if table is None:
            raise EngineUnavailable("embeddings are not loaded")
        if isinstance(table, list):
            rows = []
            for token in token_ids:
                if token < 0 or token >= len(table):
                    raise EngineUnavailable(f"token id {token} is outside the embedding table")
                rows.append(list(table[token]))
            return as_blob(rows)
        module = type(table).__module__
        if module.startswith("torch"):
            import torch

            ids = torch.tensor(token_ids, dtype=torch.long, device=table.device)
            rows = torch.nn.functional.embedding(ids, table).float().detach().cpu().tolist()
            return as_blob(rows)
        if module.startswith("mlx"):
            import mlx.core as mx

            picked = table[mx.array(token_ids, dtype=mx.int32)]
            mx.eval(picked)
            return as_blob(picked.tolist())
        raise EngineUnavailable("embedding table is not a list, torch tensor, or mlx array")

    def forward(
        self,
        hidden: TensorBlob,
        layer_start: int,
        layer_end: int,
        past_keys: TensorBlob | None,
        past_values: TensorBlob | None,
        position_start: int,
    ) -> tuple[TensorBlob, TensorBlob, TensorBlob]:
        if (layer_start, layer_end) != (self.layer_start, self.layer_end):
            raise LayerRangeError(
                f"{self.role} runner refuses [{layer_start}, {layer_end}); "
                f"locked range is [{self.layer_start}, {self.layer_end})"
            )
        if self.layers is None:
            raise EngineUnavailable("runner has no weights; call load() or bind_weights()")
        rows = hidden.nested()
        past_k = None if past_keys is None else past_keys.nested()
        past_v = None if past_values is None else past_values.nested()
        kind = self.layers[0].kind
        if kind == "list":
            out, keys, values = forward_layers_list(self.layers, rows, past_k, past_v, position_start)
        elif kind == "torch":
            out, keys, values = forward_layers_torch(self.layers, rows, past_k, past_v, position_start)
        elif kind == "mlx":
            out, keys, values = forward_layers_mlx(self.layers, rows, past_k, past_v, position_start)
        else:
            raise EngineUnavailable(f"unknown weight kind {kind}")
        return as_blob(out), as_blob(keys), as_blob(values)

    def apply(self, hidden: TensorBlob) -> TensorBlob:
        if self.role != "mac":
            raise LayerRangeError("Windows runner cannot apply lm_head")
        if self.lm_weight is None or self.final_norm is None:
            raise EngineUnavailable("lm_head is not loaded")
        rows = hidden.nested()
        kind = "list" if isinstance(self.lm_weight, list) else ("mlx" if type(self.lm_weight).__module__.startswith("mlx") else "torch")
        if kind == "list":
            logits = apply_lm_head_list(rows, self.final_norm, self.lm_weight, self.spec.rms_norm_eps)
        elif kind == "torch":
            logits = apply_lm_head_torch(rows, self.final_norm, self.lm_weight, self.spec.rms_norm_eps)
        else:
            logits = apply_lm_head_mlx(rows, self.final_norm, self.lm_weight, self.spec.rms_norm_eps)
        return as_blob(logits)
