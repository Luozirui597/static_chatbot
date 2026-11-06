"""Single-use execution gate for the isolated Option A web test instance.

The gate advances through each frozen script in order.  A script is bound to
exactly one session, a turn is armed only after the previous turn has been
fully checked and completed, and a consumed turn can never be armed again.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Mapping, Sequence


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class GateError(RuntimeError):
    """A safe, non-secret error raised by the arm gate."""


@dataclass(frozen=True)
class TurnSpec:
    script_id: str
    turn_id: str
    content: str
    content_sha256: str


@dataclass(frozen=True)
class ArmedTurn:
    instance_id: str
    session_id: int
    script_id: str
    turn_id: str
    content_sha256: str


@dataclass(frozen=True)
class TurnCompletion:
    instance_id: str
    session_id: int
    script_id: str
    turn_id: str
    user_message_id: int
    assistant_message_id: int
    content_sha256: str


class RunState:
    """Batch state shared by the gate and the request wrapper.

    A failed batch is not recoverable in the same process.  The caller must
    start a new isolated run rather than retrying a turn.
    """

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self._reason: str | None = None
        self._lock = threading.Lock()

    @property
    def failed(self) -> bool:
        return self._reason is not None

    @property
    def failure_reason(self) -> str | None:
        return self._reason

    def fail(self, reason: str) -> None:
        safe_reason = reason.strip() or "unspecified failure"
        with self._lock:
            if self._reason is None:
                self._reason = safe_reason[:500]

    def ensure_ok(self) -> None:
        if self._reason is not None:
            raise GateError(f"run stopped: {self._reason}")


class TurnGate:
    """Enforce one ordered, single-use execution of the frozen batch."""

    def __init__(
        self,
        *,
        instance_id: str,
        scripts: Mapping[str, Sequence[TurnSpec]] | None = None,
        turns: Mapping[tuple[str, str], TurnSpec] | None = None,
        run_state: RunState | None = None,
    ) -> None:
        if (scripts is None) == (turns is None):
            raise ValueError("provide exactly one of scripts or turns")

        if turns is not None:
            grouped: dict[str, list[TurnSpec]] = {}
            for spec in turns.values():
                grouped.setdefault(spec.script_id, []).append(spec)
            scripts = {script_id: tuple(items) for script_id, items in grouped.items()}

        assert scripts is not None
        self.instance_id = instance_id
        self._scripts: dict[str, tuple[TurnSpec, ...]] = {}
        self._script_order: list[str] = []
        self._turns: dict[tuple[str, str], TurnSpec] = {}
        for script_id, items in scripts.items():
            if not isinstance(script_id, str) or not items:
                raise ValueError("every script needs a non-empty turn sequence")
            ordered = tuple(items)
            seen_ids: set[str] = set()
            for spec in ordered:
                if spec.script_id != script_id:
                    raise ValueError("turn script_id does not match its script")
                if spec.turn_id in seen_ids:
                    raise ValueError("duplicate turn id in a script")
                if spec.content_sha256 != content_sha256(spec.content):
                    raise ValueError("turn content hash does not match content")
                seen_ids.add(spec.turn_id)
                self._turns[(script_id, spec.turn_id)] = spec
            self._scripts[script_id] = ordered
            self._script_order.append(script_id)

        if not self._script_order:
            raise ValueError("the frozen batch has no scripts")

        self._run_state = run_state or RunState(instance_id)
        self._lock = threading.Lock()
        self._armed: ArmedTurn | None = None
        self._in_flight: ArmedTurn | None = None
        self._session_by_script: dict[str, int] = {}
        self._script_by_session: dict[int, str] = {}
        self._next_index: dict[str, int] = {
            script_id: 0 for script_id in self._script_order
        }
        self._completed: dict[tuple[str, str], TurnCompletion] = {}
        self._total_consumed = 0
        self.total_expected = sum(len(items) for items in self._scripts.values())

    @property
    def run_state(self) -> RunState:
        return self._run_state

    @property
    def turns(self) -> dict[tuple[str, str], TurnSpec]:
        return dict(self._turns)

    @property
    def total_consumed(self) -> int:
        with self._lock:
            return self._total_consumed

    @property
    def in_flight(self) -> ArmedTurn | None:
        with self._lock:
            return self._in_flight

    @property
    def armed(self) -> ArmedTurn | None:
        with self._lock:
            return self._armed

    @property
    def next_turn(self) -> TurnSpec | None:
        with self._lock:
            for script_id in self._script_order:
                index = self._next_index[script_id]
                specs = self._scripts[script_id]
                if index < len(specs):
                    return specs[index]
            return None

    def session_for_script(self, script_id: str) -> int | None:
        with self._lock:
            return self._session_by_script.get(script_id)

    def completed_turn(
        self, script_id: str, turn_id: str
    ) -> TurnCompletion | None:
        with self._lock:
            return self._completed.get((script_id, turn_id))

    def arm(
        self,
        *,
        instance_id: str,
        session_id: int,
        script_id: str,
        turn_id: str,
    ) -> ArmedTurn:
        """Arm the next frozen turn; no model call is made here."""

        with self._lock:
            self._run_state.ensure_ok()
            if instance_id != self.instance_id:
                raise GateError("arm belongs to a different test instance")
            if not isinstance(session_id, int) or isinstance(session_id, bool) or session_id <= 0:
                raise GateError("session id must be a positive integer")
            if self._in_flight is not None:
                raise GateError("a request is still in flight")
            if self._armed is not None:
                raise GateError("a previous turn is armed but not consumed")
            if self._total_consumed >= self.total_expected:
                raise GateError("the frozen batch has no remaining calls")

            if script_id not in self._scripts:
                raise GateError("unknown script")
            if turn_id not in {spec.turn_id for spec in self._scripts[script_id]}:
                raise GateError("unknown turn")

            next_script = None
            for candidate in self._script_order:
                if self._next_index[candidate] < len(self._scripts[candidate]):
                    next_script = candidate
                    break
            if script_id != next_script:
                raise GateError("script is not next in the frozen batch order")

            index = self._next_index[script_id]
            expected = self._scripts[script_id][index]
            if turn_id != expected.turn_id:
                raise GateError("turn is not next in the frozen script order")
            if (script_id, turn_id) in self._completed:
                raise GateError("turn was already completed")

            bound_session = self._session_by_script.get(script_id)
            if bound_session is not None and bound_session != session_id:
                raise GateError("script is bound to a different session")
            session_script = self._script_by_session.get(session_id)
            if session_script is not None and session_script != script_id:
                raise GateError("session is already bound to another script")
            if bound_session is None:
                self._session_by_script[script_id] = session_id
                self._script_by_session[session_id] = script_id

            armed = ArmedTurn(
                instance_id=self.instance_id,
                session_id=session_id,
                script_id=script_id,
                turn_id=turn_id,
                content_sha256=spec_content_hash(expected),
            )
            self._armed = armed
            return armed

    def consume(self, *, session_id: int, content: str) -> ArmedTurn:
        """Atomically consume the current arm.

        The request is now in flight.  The turn cannot advance until the HTTP
        layer has verified the response, call count, snapshot and evidence.
        """

        with self._lock:
            self._run_state.ensure_ok()
            if self._in_flight is not None:
                raise GateError("a request is already in flight")
            if self._armed is None:
                raise GateError("no armed turn to consume")
            if session_id != self._armed.session_id:
                raise GateError("request session does not match armed session")
            if content_sha256(content) != self._armed.content_sha256:
                raise GateError("request content does not match frozen turn")
            self._in_flight = self._armed
            self._armed = None
            self._total_consumed += 1
            return self._in_flight

    def complete(
        self,
        *,
        armed: ArmedTurn,
        user_message_id: int,
        assistant_message_id: int,
    ) -> TurnCompletion:
        """Advance exactly one turn after all request checks have passed."""

        with self._lock:
            self._run_state.ensure_ok()
            if self._in_flight is None or armed != self._in_flight:
                raise GateError("no matching in-flight turn to complete")
            if (
                not isinstance(user_message_id, int)
                or isinstance(user_message_id, bool)
                or user_message_id <= 0
                or not isinstance(assistant_message_id, int)
                or isinstance(assistant_message_id, bool)
                or assistant_message_id <= 0
                or user_message_id == assistant_message_id
            ):
                raise GateError("completion message ids are invalid")
            script_id = armed.script_id
            index = self._next_index[script_id]
            specs = self._scripts[script_id]
            if index >= len(specs) or specs[index].turn_id != armed.turn_id:
                raise GateError("in-flight turn is not next in the script")
            if (script_id, armed.turn_id) in self._completed:
                raise GateError("turn was already completed")

            completion = TurnCompletion(
                instance_id=armed.instance_id,
                session_id=armed.session_id,
                script_id=script_id,
                turn_id=armed.turn_id,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
                content_sha256=armed.content_sha256,
            )
            self._completed[(script_id, armed.turn_id)] = completion
            self._next_index[script_id] = index + 1
            self._in_flight = None
            return completion


def spec_content_hash(spec: TurnSpec) -> str:
    # Kept as a named function so the arm object cannot silently diverge from
    # the frozen content hash.
    return spec.content_sha256
