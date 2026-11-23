"""A layer node executes exactly one frozen range and returns hidden states plus new KV."""

from __future__ import annotations

from pool.errors import LayerRangeError
from pool.protocol.types import ForwardRequest, ForwardResponse, Health


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
    ):
        if layer_end <= layer_start:
            raise LayerRangeError("node range is empty")
        if layer_start == 0 and layer_end >= 28:
            raise LayerRangeError("a node cannot own the full [0, 28) stack")
        self.name = name
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.runner = runner
        self.lm_head = lm_head
        self.reachable = reachable
        self.fail_after = fail_after
        self.seen: list[dict] = []
        self._ok_forwards = 0

    def health(self, timeout: float = 3.0) -> Health:
        del timeout
        if not self.reachable:
            raise TimeoutError(f"{self.name} unreachable")
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
    return LayerNode("mac", 0, 14, runner, lm_head=head, **kwargs)


def build_mock_windows_node(**kwargs) -> LayerNode:
    from pool.runtime.mock import MockRunner

    if kwargs.get("lm_head") is not None:
        raise LayerRangeError("Windows mock node cannot own an lm_head")
    runner = kwargs.pop("runner", None) or MockRunner()
    kwargs.pop("lm_head", None)
    return LayerNode("windows", 14, 28, runner, lm_head=None, **kwargs)
