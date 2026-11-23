"""Schedule a session across Mac [0, 14) and, when reachable, Windows [14, 28)."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pool.dispatch.assembler import assemble
from pool.dispatch.sampling import greedy
from pool.errors import LayerRangeError, PlaintextLeak, SessionFailed
from pool.protocol.constants import (
    DEGRADED_FRONT_HALF,
    FRONT_HALF_WARNING,
    HEALTH_TIMEOUT_S,
    WINDOWS_LAYER_END,
    WINDOWS_LAYER_START,
    assert_locked_split,
)
from pool.protocol.guard import assert_windows_request
from pool.protocol.types import ForwardRequest, GenerateResult, Health


class RemoteFailure(Exception):
    """Windows KV or the Windows node failed after a dual session had started."""


@dataclass(frozen=True)
class _Decision:
    kind: str
    kv: object
    degraded: str


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
        *,
        health_timeout_s: float = HEALTH_TIMEOUT_S,
    ):
        assert_locked_split()
        self.mac_worker = mac_worker
        self.embedder = embedder
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.local_kv = local_kv
        self.windows_worker = windows_worker
        self.windows_kv = windows_kv
        self.health_timeout_s = health_timeout_s
        if (mac_worker.layer_start, mac_worker.layer_end) != (0, 14):
            raise LayerRangeError("dispatcher requires a Mac node locked to [0, 14)")

    def generate(self, prompt: str, *, mode: str = "auto", max_new_tokens: int = 32) -> GenerateResult:
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("prompt is empty")
        if mode not in {"auto", "single-mac", "dual"}:
            raise ValueError("mode must be auto, single-mac, or dual")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        decision = self._decide(mode)
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
            warning = FRONT_HALF_WARNING if decision.degraded == DEGRADED_FRONT_HALF else ""
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
            if decision.kind == "dual":
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
        kv_health, node_health = self._probe_windows()
        kv_up = bool(kv_health.ok)
        node_up = bool(node_health.ok) and (node_health.layer_start, node_health.layer_end) == (
            WINDOWS_LAYER_START,
            WINDOWS_LAYER_END,
        )
        healthy = kv_up and node_up
        return {
            "windows_reachable": healthy,
            "windows_kv": "up" if kv_up else "down",
            "windows_node": "up" if node_up else "down",
            "resolved_mode": "dual" if healthy else "single-mac",
            "degraded_if_auto": "" if healthy else DEGRADED_FRONT_HALF,
            "health_timeout_s": self.health_timeout_s,
        }

    def _decide(self, mode: str) -> _Decision:
        if mode == "single-mac":
            return _Decision("single-mac", self.local_kv, DEGRADED_FRONT_HALF)
        healthy = self._windows_healthy()
        if mode == "dual":
            if not healthy:
                raise SessionFailed(
                    "mode=dual but Windows KV or Windows Node is unreachable",
                    code="windows_unreachable",
                )
            return _Decision("dual", self.windows_kv, "")
        if healthy:
            return _Decision("dual", self.windows_kv, "")
        return _Decision("single-mac", self.local_kv, DEGRADED_FRONT_HALF)

    def _windows_healthy(self) -> bool:
        return self.peer_status()["windows_reachable"]

    def _probe_windows(self) -> tuple[Health, Health]:
        if self.windows_kv is None or self.windows_worker is None:
            return (
                Health(ok=False, role="windows-kv", detail="not configured"),
                Health(ok=False, role="windows-node", detail="not configured"),
            )
        deadline = time.monotonic() + self.health_timeout_s

        def remain() -> float:
            return max(0.01, deadline - time.monotonic())

        pool = ThreadPoolExecutor(max_workers=2)
        try:
            kv_future = pool.submit(self.windows_kv.health, self.health_timeout_s)
            node_future = pool.submit(self.windows_worker.health, self.health_timeout_s)
            kv_health = self._wait(kv_future, remain(), "windows-kv")
            node_health = self._wait(node_future, remain(), "windows-node")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return kv_health, node_health

    @staticmethod
    def _wait(future, timeout: float, role: str) -> Health:
        try:
            result = future.result(timeout=timeout)
        except Exception:
            return Health(ok=False, role=role, detail="unreachable")
        if not isinstance(result, Health):
            return Health(ok=False, role=role, detail="bad health payload")
        return result

    def _step(self, decision, session_id, phase, hidden, position_start, seq_len, prompt):
        past_keys, past_values = decision.kv.read(session_id, 0, 14)
        mac_request = ForwardRequest(
            session_id=session_id,
            phase=phase,
            layer_start=0,
            layer_end=14,
            position_start=position_start,
            seq_len=seq_len,
            hidden=hidden,
            return_logits=decision.kind == "single-mac",
            past_keys=past_keys,
            past_values=past_values,
        )
        mac_response = self.mac_worker.forward(mac_request)
        if (mac_response.layer_start, mac_response.layer_end) != (0, 14):
            raise LayerRangeError("Mac response was not [0, 14)")
        self._append(decision, session_id, mac_response, position_start, prompt)
        if decision.kind == "single-mac":
            if mac_response.logits is None:
                raise SessionFailed(
                    "front-half path did not return logits from Mac [0, 14)",
                    code="no_logits",
                )
            return mac_response.hidden, mac_response.logits
        past_keys, past_values = decision.kv.read(session_id, 14, 28)
        windows_request = ForwardRequest(
            session_id=session_id,
            phase=phase,
            layer_start=14,
            layer_end=28,
            position_start=position_start,
            seq_len=seq_len,
            hidden=mac_response.hidden,
            return_logits=False,
            past_keys=past_keys,
            past_values=past_values,
        )
        assert_windows_request(windows_request.public_dict(), prompt)
        try:
            windows_response = self.windows_worker.forward(windows_request)
        except (PlaintextLeak, LayerRangeError):
            raise
        except Exception as exc:
            raise RemoteFailure(str(exc)) from exc
        if (windows_response.layer_start, windows_response.layer_end) != (14, 28):
            raise LayerRangeError("Windows response was not [14, 28)")
        self._append(decision, session_id, windows_response, position_start, prompt)
        return windows_response.hidden, self.lm_head.apply(windows_response.hidden)

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
            if decision.kind == "dual":
                raise RemoteFailure(str(exc)) from exc
            raise
