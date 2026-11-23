"""One language step at a time.

After a node finishes the range it was given, the machine asks which node and
range run next. The next hop may be a different node or the same node again.
Membership may be replanned only on that boundary. A failure inside the hop
does not reach ``commit``.
"""

from __future__ import annotations

from pool.dispatch.schedule import Hop, NodeOffer, ScheduleError, allocate_flexible
from pool.errors import PoolError


class CoverageError(PoolError):
    """Remaining layers cannot be covered without shrinking the run mid-flight."""


class HopMachine:
    def __init__(self, hops: list[Hop]):
        if not hops:
            raise CoverageError("hop list is empty")
        self.hops = list(hops)
        self.index = 0

    @property
    def current(self) -> Hop:
        if self.index >= len(self.hops):
            raise CoverageError("no hop is in progress")
        return self.hops[self.index]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.hops)

    def commit(self, live_ids: set[str], offers: dict[str, NodeOffer]) -> Hop | None:
        """Record a finished hop and return the next one, replanning the suffix if needed."""
        if self.finished:
            return None
        if self.index + 1 >= len(self.hops):
            self.index += 1
            return None
        remaining = self.hops[self.index + 1 :]
        if all(hop.node_id in live_ids for hop in remaining):
            self.index += 1
            return self.current
        start = remaining[0].start
        end = remaining[-1].end
        survivors = [offers[node_id] for node_id in sorted(live_ids) if node_id in offers]
        # A full-cover request must not collapse to a Mac-only prefix between hops.
        if len(survivors) < 2 and not _same_node_suffix(remaining, survivors):
            raise CoverageError(
                "a node left between hops; remaining layers are not covered; "
                "not switching to a front-half run"
            )
        try:
            replanned = _replan_suffix(survivors, start, end)
        except ScheduleError as exc:
            raise CoverageError(str(exc)) from exc
        self.hops = self.hops[: self.index + 1] + replanned
        self.index += 1
        return self.current


def _same_node_suffix(remaining: list[Hop], survivors: list[NodeOffer]) -> bool:
    if len(survivors) != 1 or not remaining:
        return False
    return all(hop.node_id == survivors[0].node_id for hop in remaining)


def _replan_suffix(offers: list[NodeOffer], start: int, end: int) -> list[Hop]:
    span = end - start
    if span < 1:
        raise ScheduleError("empty suffix")
    # Reuse the allocator on a synthetic tower of ``span`` layers, then shift.
    shifted = allocate_flexible(offers, span)
    return [Hop(hop.node_id, hop.start + start, hop.end + start) for hop in shifted]
