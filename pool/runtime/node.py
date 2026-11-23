"""A layer node executes the contiguous range assigned for the current hop."""

from __future__ import annotations

from pool.errors import LayerRangeError
from pool.protocol.constants import (
    MAC_LAYER_END,
    MAC_LAYER_START,
    NUM_LAYERS,
    WINDOWS_LAYER_END,
    WINDOWS_LAYER_START,
)
from pool.protocol.types import ForwardRequest, ForwardResponse, Health


def _check_shard(layer_start: int, layer_end: int) -> None:
    if layer_end <= layer_start:
        raise LayerRangeError("node range is empty")
    if layer_start < 0 or layer_end > NUM_LAYERS:
        raise LayerRangeError(
            f"node range [{layer_start}, {layer_end}) is outside the {NUM_LAYERS}-layer language tower"
        )
    if layer_start == 0 and layer_end >= NUM_LAYERS:
        raise LayerRangeError(
            f"a node cannot execute the full [0, {NUM_LAYERS}) tower in one hop"
        )


class LayerNode:
    def __init__(
        self,
        name: str,
        layer_start: int,
        layer_end: int,
        runner,
        lm_head=None,
        *,
        reachable: bool = True,
        fail_after: int | None = None,
        flexible: bool = False,
        vram_gb: float = 16.0,
        max_layers: int | None = None,
        offload: bool = False,
    ):
        _check_shard(layer_start, layer_end)
        self.name = name
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.runner = runner
        self.lm_head = lm_head
        self.reachable = reachable
        self.fail_after = fail_after
        self.flexible = flexible
        self.vram_gb = vram_gb
        self.max_layers = NUM_LAYERS if max_layers is None else max_layers
        self.offload = offload
        self.seen: list[dict] = []
        self._ok_forwards = 0

    def assign(self, layer_start: int, layer_end: int) -> None:
        """Bind the range for the next hop. The same process may be bound again later."""
        _check_shard(layer_start, layer_end)
        if not self.flexible and (layer_start, layer_end) != (self.layer_start, self.layer_end):
            raise LayerRangeError(
                f"{self.name} is fixed to [{self.layer_start}, {self.layer_end}), "
                f"refusing [{layer_start}, {layer_end})"
            )
        self.layer_start = layer_start
        self.layer_end = layer_end

    def health(self, timeout: float = 3.0) -> Health:
        del timeout
        if not self.reachable:
            raise TimeoutError(f"{self.name} unreachable")
        if self.flexible:
            return Health(ok=True, role=self.name, layer_start=None, layer_end=None, detail="assignable")
        return Health(
            ok=True,
            role=self.name,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            detail="layer-node",
        )

    def forward(self, request: ForwardRequest) -> ForwardResponse:
        if not self.reachable:
            raise ConnectionError(f"{self.name} unreachable")
        if self.fail_after is not None and self._ok_forwards >= self.fail_after:
            raise ConnectionError(f"{self.name} dropped")
        if self.flexible:
            self.assign(request.layer_start, request.layer_end)
        if (request.layer_start, request.layer_end) != (self.layer_start, self.layer_end):
            raise LayerRangeError(
                f"{self.name} owns [{self.layer_start}, {self.layer_end}), "
                f"refusing [{request.layer_start}, {request.layer_end})"
            )
        if request.return_logits and self.lm_head is None:
            raise LayerRangeError(f"{self.name} has no lm_head and cannot return logits")
        self.seen.append(request.public_dict())
        hidden, keys, values = self.runner.forward(
            request.hidden,
            request.layer_start,
            request.layer_end,
            request.past_keys,
            request.past_values,
            request.position_start,
        )
        logits = self.lm_head.apply(hidden) if request.return_logits else None
        self._ok_forwards += 1
        return ForwardResponse(
            hidden=hidden,
            keys=keys,
            values=values,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            logits=logits,
        )


def build_mock_mac_node(**kwargs) -> LayerNode:
    from pool.runtime.mock import MockLmHead, MockRunner

    head = kwargs.pop("lm_head", None) or MockLmHead()
    runner = kwargs.pop("runner", None) or MockRunner()
    return LayerNode("mac", MAC_LAYER_START, MAC_LAYER_END, runner, lm_head=head, **kwargs)


def build_mock_windows_node(**kwargs) -> LayerNode:
    from pool.runtime.mock import MockRunner

    if kwargs.get("lm_head") is not None:
        raise LayerRangeError("Windows mock node cannot own an lm_head")
    runner = kwargs.pop("runner", None) or MockRunner()
    kwargs.pop("lm_head", None)
    return LayerNode("windows", WINDOWS_LAYER_START, WINDOWS_LAYER_END, runner, lm_head=None, **kwargs)
