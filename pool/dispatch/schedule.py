"""Assign contiguous language-layer hops at runtime.

Config files do not own the cut. A two-node pool of similar capacity still
lands on the reference even split, but that is a scheduling result. A single
node never receives the whole tower in one hop: self-check uses two hops on
that same node.
"""

from __future__ import annotations

from dataclasses import dataclass

from pool.errors import PoolError
from pool.protocol.constants import NUM_LAYERS
from pool.runtime.offload import LAYER_WEIGHT_BYTES, GIB


class ScheduleError(PoolError):
    """The live nodes cannot cover the requested language-layer span."""


@dataclass(frozen=True)
class Hop:
    node_id: str
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class NodeOffer:
    node_id: str
    vram_gb: float
    max_layers: int
    offload: bool
    holds_private: bool
    fixed_start: int | None = None
    fixed_end: int | None = None

    @property
    def flexible(self) -> bool:
        return self.fixed_start is None or self.fixed_end is None


def _layer_gib() -> float:
    return LAYER_WEIGHT_BYTES / GIB


def layers_without_offload(offer: NodeOffer) -> int:
    """How many language layers fit in reported VRAM before CPU offload.

    The scheduler prefers this count so a smaller GPU receives a lighter shard
    instead of an even split that immediately spills.
    """
    fit = int(offer.vram_gb / _layer_gib()) if offer.vram_gb > 0 else offer.max_layers
    fit = max(1, fit)
    return max(1, min(offer.max_layers, fit))


def _hamilton(weights: list[float], total: int) -> list[int]:
    if total < len(weights):
        raise ScheduleError("more nodes than language layers")
    if not weights or sum(weights) <= 0:
        raise ScheduleError("nodes reported no schedulable capacity")
    quotas = [total * weight / sum(weights) for weight in weights]
    counts = [int(quota) for quota in quotas]
    leftover = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (quotas[index] - counts[index], weights[index], -index),
        reverse=True,
    )
    for index in order:
        if leftover <= 0:
            break
        counts[index] += 1
        leftover -= 1
    guard = 0
    while any(count <= 0 for count in counts):
        empty = counts.index(0)
        donor = max(range(len(counts)), key=lambda index: counts[index])
        if counts[donor] <= 1:
            raise ScheduleError("cannot give every live node a non-empty shard")
        counts[donor] -= 1
        counts[empty] += 1
        guard += 1
        if guard > total:
            raise ScheduleError("layer allocation did not converge")
    if sum(counts) != total:
        raise ScheduleError("layer allocation does not cover the requested span")
    if any(count >= total for count in counts):
        raise ScheduleError("refusing one hop across the full language tower")
    return counts


def fixed_cover(offers: list[NodeOffer], num_layers: int = NUM_LAYERS) -> list[Hop] | None:
    """Use pre-bound ranges when they already tile ``[0, num_layers)``."""
    if any(offer.flexible for offer in offers):
        return None
    ordered = sorted(offers, key=lambda offer: int(offer.fixed_start))
    cursor = 0
    hops: list[Hop] = []
    for offer in ordered:
        start = int(offer.fixed_start)
        end = int(offer.fixed_end)
        if start != cursor or end <= start:
            return None
        if start == 0 and end >= num_layers:
            return None
        hops.append(Hop(offer.node_id, start, end))
        cursor = end
    if cursor != num_layers:
        return None
    return hops


def allocate_flexible(offers: list[NodeOffer], num_layers: int = NUM_LAYERS) -> list[Hop]:
    """Weight shards by VRAM that does not need offload. The private node is first."""
    private = [offer for offer in offers if offer.holds_private]
    if len(private) != 1:
        raise ScheduleError("a full language run needs exactly one Mac private node")
    others = sorted(
        (offer for offer in offers if not offer.holds_private),
        key=lambda offer: offer.node_id,
    )
    ordered = [private[0], *others]
    if len(ordered) == 1:
        raise ScheduleError("one node cannot take the language tower in a single hop")
    weights = [float(layers_without_offload(offer)) for offer in ordered]
    counts = _hamilton(weights, num_layers)
    for offer, count in zip(ordered, counts):
        if count > offer.max_layers:
            raise ScheduleError(
                f"{offer.node_id} can host {offer.max_layers} language layers, "
                f"schedule asked for {count}"
            )
    hops: list[Hop] = []
    cursor = 0
    for offer, count in zip(ordered, counts):
        hops.append(Hop(offer.node_id, cursor, cursor + count))
        cursor += count
    return hops


def same_node_hops(node_id: str, num_layers: int = NUM_LAYERS) -> list[Hop]:
    """Two hops on one process. This is not one full-tower engine call."""
    if num_layers < 2:
        raise ScheduleError("language tower is too small to split")
    mid = num_layers // 2
    return [Hop(node_id, 0, mid), Hop(node_id, mid, num_layers)]


def front_portion(node_id: str, *, limit: int, num_layers: int = NUM_LAYERS) -> list[Hop]:
    """Mac-only smoke. Never the full tower, and never past ``limit`` (the front half)."""
    width = min(limit, num_layers // 2)
    if width < 1 or width >= num_layers:
        raise ScheduleError("front portion must be a proper prefix of the language tower")
    return [Hop(node_id, 0, width)]


def cover_or_none(offers: list[NodeOffer], num_layers: int = NUM_LAYERS) -> list[Hop] | None:
    if len(offers) < 2:
        return None
    fixed = fixed_cover(offers, num_layers)
    if fixed is not None:
        return fixed
    if any(not offer.flexible for offer in offers):
        return None
    try:
        return allocate_flexible(offers, num_layers)
    except ScheduleError:
        return None
