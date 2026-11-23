"""Schedule language hops, then run one assigned range at a time.

The dispatcher does not bake in a Mac-then-Windows pipeline. Each hop returns
hidden states here, and the next hop may be another node or the same node
again. Vision, embeddings, tokenization, and lm_head stay on the Mac assembler.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pool.dispatch.assembler import assemble
from pool.dispatch.machine import CoverageError, HopMachine
from pool.dispatch.sampling import greedy
from pool.dispatch.schedule import (
    Hop,
    NodeOffer,
    cover_or_none,
    front_portion,
    same_node_hops,
)
from pool.errors import LayerRangeError, PlaintextLeak, SessionFailed
from pool.protocol.constants import (
    DEGRADED_FRONT_HALF,
    FRONT_HALF_WARNING,
    HEALTH_TIMEOUT_S,
    MAC_LAYER_END,
    NUM_LAYERS,
    assert_locked_split,
)
from pool.protocol.guard import assert_windows_request
from pool.protocol.types import ForwardRequest, GenerateResult, Health

SELF_VERIFY_DEGRADED = "mac_self_verify"
SELF_VERIFY_WARNING = (
    "Mac self-check ran the language tower as separate hops on one process, "
    "not as one full-tower engine call. Output quality does not count. "
    "同一台 Mac 分两跳核对语言塔，不是一次跑满 36 层。输出质量不计。"
)


class RemoteFailure(Exception):
    """A remote hop failed after a full-cover session had started."""


@dataclass(frozen=True)
class _Decision:
    kind: str
    kv: object
    degraded: str
    hops: tuple


class Dispatcher:
    def __init__(
        self,
        mac_worker,
        embedder,
        lm_head,
        tokenizer,
        local_kv,
        windows_worker=None,
        windows_kv=None,
        peers=None,
        *,
        health_timeout_s: float = HEALTH_TIMEOUT_S,
        self_verify: bool = False,
    ):
        assert_locked_split()
        self.mac_worker = mac_worker
        self.embedder = embedder
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.local_kv = local_kv
        self.windows_kv = windows_kv
        self.health_timeout_s = health_timeout_s
        self.self_verify = self_verify
        self.remotes: list = []
        if windows_worker is not None:
            self.remotes.append(windows_worker)
        for peer in peers or []:
            if peer is not windows_worker:
                self.remotes.append(peer)
        self._workers = {_node_id(mac_worker): mac_worker}
        for remote in self.remotes:
            self._workers[_node_id(remote)] = remote
        if not getattr(mac_worker, "flexible", False):
            if mac_worker.layer_start != 0 or mac_worker.layer_end > MAC_LAYER_END:
                raise LayerRangeError(
                    "a fixed Mac node must own a front prefix no wider than the reference front half"
                )

    def generate(
        self,
        prompt: str,
        *,
        mode: str = "auto",
        max_new_tokens: int = 32,
        self_verify: bool | None = None,
    ) -> GenerateResult:
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("prompt is empty")
        if mode not in {"auto", "single-mac", "dual"}:
            raise ValueError("mode must be auto, single-mac, or dual")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        verify = self.self_verify if self_verify is None else self_verify
        decision = self._decide(mode, verify)
        session_id = uuid.uuid4().hex
        decision.kv.create(session_id)
        try:
            token_ids = self.tokenizer.encode(prompt)
            if not token_ids:
                raise ValueError("tokenizer produced no tokens")
            hidden = self.embedder.embed(token_ids)
            _hidden, logits = self._step(
                decision,
                session_id,
                "prefill",
                hidden,
                position_start=0,
                seq_len=len(token_ids),
                prompt=prompt,
            )
            produced: list[int] = []
            while len(produced) < max_new_tokens:
                next_id = greedy(logits)
                if next_id == self.tokenizer.eos_id:
                    break
                produced.append(next_id)
                if len(produced) >= max_new_tokens:
                    break
                step_hidden = self.embedder.embed([next_id])
                _hidden, logits = self._step(
                    decision,
                    session_id,
                    "decode",
                    step_hidden,
                    position_start=len(token_ids) + len(produced) - 1,
                    seq_len=1,
                    prompt=prompt,
                )
            text = self.tokenizer.decode(produced)
            thinking, answer = assemble(text)
            warning = _warning(decision.degraded)
            return GenerateResult(
                answer=answer,
                thinking=thinking,
                degraded=decision.degraded,
                mode_used=decision.kind,
                prompt_tokens=len(token_ids),
                completion_tokens=len(produced),
                session_id=session_id,
                warning=warning,
            )
        except RemoteFailure as exc:
            if decision.kind != "single-mac":
                raise SessionFailed(
                    "Windows dropped mid-session; KV cleaned; not switching to front-half",
                    code="windows_dropped",
                ) from exc
            raise SessionFailed(str(exc), code="local_failure") from exc
        finally:
            try:
                decision.kv.delete(session_id)
            except Exception:
                pass

    def peer_status(self) -> dict:
        kv_ok, live = self._probe()
        full = self._full_hops(live) if kv_ok else None
        healthy = full is not None
        return {
            "windows_reachable": healthy,
            "windows_kv": "up" if kv_ok and self.windows_kv is not None else "down",
            "windows_node": "up" if live else "down",
            "resolved_mode": "dual" if healthy else "single-mac",
            "degraded_if_auto": "" if healthy else DEGRADED_FRONT_HALF,
            "health_timeout_s": self.health_timeout_s,
            "mac_layers": [0, MAC_LAYER_END] if not healthy else [full[0].start, full[0].end],
            "remote_hops": []
            if not full
            else [[hop.node_id, hop.start, hop.end] for hop in full if hop.node_id != _node_id(self.mac_worker)],
        }

    def _decide(self, mode: str, self_verify: bool) -> _Decision:
        if mode == "single-mac":
            hops = front_portion(_node_id(self.mac_worker), limit=self._front_limit())
            return _Decision("single-mac", self.local_kv, DEGRADED_FRONT_HALF, tuple(hops))
        kv_ok, live = self._probe()
        full = self._full_hops(live) if kv_ok else None
        if mode == "dual":
            if full is None:
                raise SessionFailed(
                    "mode=dual but Windows KV or Windows Node is unreachable",
                    code="windows_unreachable",
                )
            return _Decision("dual", self.windows_kv, "", tuple(full))
        if full is not None:
            return _Decision("dual", self.windows_kv, "", tuple(full))
        if self_verify:
            hops = same_node_hops(_node_id(self.mac_worker), NUM_LAYERS)
            return _Decision("single-mac", self.local_kv, SELF_VERIFY_DEGRADED, tuple(hops))
        hops = front_portion(_node_id(self.mac_worker), limit=self._front_limit())
        return _Decision("single-mac", self.local_kv, DEGRADED_FRONT_HALF, tuple(hops))

    def _front_limit(self) -> int:
        if getattr(self.mac_worker, "flexible", False):
            return min(int(getattr(self.mac_worker, "max_layers", MAC_LAYER_END)), MAC_LAYER_END)
        return min(int(self.mac_worker.layer_end), MAC_LAYER_END)

    def _full_hops(self, live: list) -> list[Hop] | None:
        if not live or self.windows_kv is None:
            return None
        offers = [self._offer(self.mac_worker, None)]
        for worker, health in live:
            offers.append(self._offer(worker, health))
        return cover_or_none(offers, NUM_LAYERS)

    def _offer(self, worker, health) -> NodeOffer:
        flexible = bool(getattr(worker, "flexible", False))
        if flexible:
            fixed_start = fixed_end = None
        elif health is not None and health.layer_start is not None and health.layer_end is not None:
            fixed_start, fixed_end = int(health.layer_start), int(health.layer_end)
        else:
            fixed_start = getattr(worker, "layer_start", None)
            fixed_end = getattr(worker, "layer_end", None)
        return NodeOffer(
            node_id=_node_id(worker),
            vram_gb=float(getattr(worker, "vram_gb", 16.0)),
            max_layers=int(getattr(worker, "max_layers", NUM_LAYERS)),
            offload=bool(getattr(worker, "offload", False)),
            holds_private=getattr(worker, "lm_head", None) is not None,
            fixed_start=fixed_start,
            fixed_end=fixed_end,
        )

    def _probe(self) -> tuple[bool, list]:
        if not self.remotes or self.windows_kv is None:
            return False, []
        deadline = time.monotonic() + self.health_timeout_s

        def remain() -> float:
            return max(0.01, deadline - time.monotonic())

        pool = ThreadPoolExecutor(max_workers=max(2, len(self.remotes) + 1))
        try:
            kv_future = pool.submit(self.windows_kv.health, self.health_timeout_s)
            node_futures = [
                (worker, pool.submit(worker.health, self.health_timeout_s)) for worker in self.remotes
            ]
            kv_health = self._wait(kv_future, remain(), "windows-kv")
            live = []
            for worker, future in node_futures:
                health = self._wait(future, remain(), _node_id(worker))
                if health.ok and _health_can_join(health):
                    live.append((worker, health))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return bool(kv_health.ok), live

    @staticmethod
    def _wait(future, timeout: float, role: str) -> Health:
        try:
            result = future.result(timeout=timeout)
        except Exception:
            return Health(ok=False, role=role, detail="unreachable")
        if not isinstance(result, Health):
            return Health(ok=False, role=role, detail="bad health payload")
        return result

    def _step(self, decision: _Decision, session_id, phase, hidden, position_start, seq_len, prompt):
        machine = HopMachine(list(decision.hops))
        current = hidden
        logits = None
        offers = {offer.node_id: offer for offer in self._offers()}
        while not machine.finished:
            hop = machine.current
            node = self._workers[hop.node_id]
            remote = _node_id(node) != _node_id(self.mac_worker)
            if hasattr(node, "assign"):
                node.assign(hop.start, hop.end)
            past_keys, past_values = decision.kv.read(session_id, hop.start, hop.end)
            return_logits = decision.kind == "single-mac" and decision.degraded == DEGRADED_FRONT_HALF
            request = ForwardRequest(
                session_id=session_id,
                phase=phase,
                layer_start=hop.start,
                layer_end=hop.end,
                position_start=position_start,
                seq_len=seq_len,
                hidden=current,
                return_logits=return_logits,
                past_keys=past_keys,
                past_values=past_values,
            )
            if remote:
                assert_windows_request(request.public_dict(), prompt)
            try:
                response = node.forward(request)
            except (PlaintextLeak, LayerRangeError):
                raise
            except Exception as exc:
                if decision.kind != "single-mac":
                    raise RemoteFailure(str(exc)) from exc
                raise
            if (response.layer_start, response.layer_end) != (hop.start, hop.end):
                raise LayerRangeError(
                    f"hop response was not [{hop.start}, {hop.end})"
                )
            self._append(decision, session_id, response, position_start, prompt)
            current = response.hidden
            if return_logits:
                if response.logits is None:
                    raise SessionFailed(
                        f"front path did not return logits from [{hop.start}, {hop.end})",
                        code="no_logits",
                    )
                logits = response.logits
            if machine.index + 1 >= len(machine.hops):
                machine.index += 1
                break
            try:
                nxt = machine.commit(self._snapshot_live_ids(), offers)
            except CoverageError as exc:
                raise RemoteFailure(str(exc)) from exc
            if nxt is None:
                break
        if decision.kind == "single-mac" and decision.degraded == DEGRADED_FRONT_HALF:
            return current, logits
        return current, self.lm_head.apply(current)

    def _offers(self) -> list[NodeOffer]:
        offers = [self._offer(self.mac_worker, None)]
        for worker in self.remotes:
            if getattr(worker, "reachable", True):
                offers.append(self._offer(worker, None))
        return offers

    def _snapshot_live_ids(self) -> set[str]:
        """Liveness at a hop boundary. Does not rebuild a front-half plan."""
        live = {_node_id(self.mac_worker)}
        _kv_ok, probed = self._probe()
        for worker, health in probed:
            if health.ok:
                live.add(_node_id(worker))
        return live

    def _append(self, decision, session_id, response, position_start, prompt) -> None:
        try:
            decision.kv.append(
                session_id,
                response.layer_start,
                response.layer_end,
                position_start,
                response.keys,
                response.values,
                prompt=prompt,
            )
        except Exception as exc:
            if decision.kind != "single-mac":
                raise RemoteFailure(str(exc)) from exc
            raise


def _warning(degraded: str) -> str:
    if degraded == DEGRADED_FRONT_HALF:
        return FRONT_HALF_WARNING
    if degraded == SELF_VERIFY_DEGRADED:
        return SELF_VERIFY_WARNING
    return ""


def _health_can_join(health: Health) -> bool:
    if health.layer_start is None or health.layer_end is None:
        return True
    if health.layer_start == 0 and health.layer_end >= NUM_LAYERS:
        return False
    return health.layer_end > health.layer_start


def _node_id(worker) -> str:
    named = getattr(worker, "name", None)
    if named:
        return named
    return f"remote-{id(worker)}"
