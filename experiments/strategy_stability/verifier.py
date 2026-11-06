"""Pure-function prototype for constrained R1 teaching verification.

This module is deliberately independent of production ChatService and of any
model client.  It consumes teacher messages and a candidate decision, and
returns either a deterministic, verified response or a structured rejection.
The candidate cannot provide the final prose; the renderer is the only writer
of the saved answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

STRATEGIES = ("standard", "a", "b", "ab")
RELATIONS = ("links",)
POLARITIES = ("positive", "negative")
ASK_KINDS = ("missing_fact", "missing_rule", "direct_target")

_IDENT = r"[A-Za-z][A-Za-z0-9_]*"
_FACT_RE = re.compile(
    rf"^(?P<prefix>Correction:\s*)?(?P<subject>{_IDENT}) "
    rf"(?P<neg>does not|doesn't) link to (?P<object>{_IDENT})\.$"
)
_POS_FACT_RE = re.compile(
    rf"^(?P<prefix>Correction:\s*)?(?P<subject>{_IDENT}) "
    rf"links to (?P<object>{_IDENT})\.$"
)
_RULE_RE = re.compile(
    rf"^Rule: if (?P<x>{_IDENT}) links to (?P<y>{_IDENT}) and "
    rf"(?P<y2>{_IDENT}) links to (?P<z>{_IDENT}) then "
    rf"(?P<x2>{_IDENT}) links to (?P<z2>{_IDENT})\."
    rf"(?: Condition: (?P<condition>{_IDENT})\.)?"
    rf"(?: Exception: (?P<exception>{_IDENT})\.)?$"
)
_STATUS_COND_RE = re.compile(
    rf"^Condition (?P<name>{_IDENT}) is (?P<value>true|false)\.$"
)
_STATUS_EXC_RE = re.compile(
    rf"^Exception (?P<name>{_IDENT}) is (?P<value>active|inactive)\.$"
)
_REVOKE_RE = re.compile(r"^Revoke rule\.$")
_CLAIM_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class ContractError(Exception):
    """Raised for malformed candidate/evidence contracts."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class TeacherMessage:
    message_id: int
    session_id: int
    role: str
    content: str


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    object: str
    polarity: str
    source_message_id: int
    quote: str


@dataclass(frozen=True)
class Rule:
    rule_id: int
    x: str
    y: str
    z: str
    condition: str | None
    exception: str | None
    quote: str


@dataclass(frozen=True)
class Status:
    kind: str
    name: str
    value: bool
    source_message_id: int
    quote: str


@dataclass
class EvidenceState:
    session_id: int
    messages: dict[int, TeacherMessage]
    allowed_ids: set[int]
    facts: dict[tuple[str, str], Fact] = field(default_factory=dict)
    rules: dict[int, Rule] = field(default_factory=dict)
    statuses: dict[tuple[str, str], Status] = field(default_factory=dict)
    revoked_rules: set[int] = field(default_factory=set)
    unsupported_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class InteractionState:
    asked: int = 0
    remediated: int = 0
    closed: bool = False

    def next_ask(self) -> "InteractionState":
        return InteractionState(asked=self.asked + 1, remediated=self.remediated, closed=self.closed)


@dataclass(frozen=True, order=True)
class Relation:
    subject: str
    object: str


@dataclass(frozen=True)
class GoalContext:
    original: Relation
    current_subgoals: tuple[Relation, ...] = ()
    askable_gaps: tuple[Relation, ...] = ()
    missing_relations: tuple[Relation, ...] = ()
    missing_rule: bool = False
    # Optional server-owned mapping from an askable gap to the target it
    # serves.  If omitted, the gap is interpreted as serving ``original``.
    gap_targets: tuple[tuple[Relation, Relation], ...] = ()


@dataclass(frozen=True)
class TurnContext:
    goal: GoalContext | None = None
    turn_kind: str = "question"
    allow_infer: bool = True
    allow_ask: bool = True
    allow_clarify: bool = False
    required_targets: tuple[Relation, ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    status: str
    code: str | None
    final_text: str | None
    next_state: InteractionState
    used_source_ids: tuple[int, ...] = ()
    detail: str = ""


def _strip(text: str) -> str:
    return " ".join(text.strip().split())


def parse_teacher_evidence(
    messages: Iterable[TeacherMessage],
    *,
    session_id: int,
    allowed_ids: Iterable[int] | None = None,
) -> EvidenceState:
    """Parse the deliberately small relation grammar.

    Unsupported messages are retained by id and are not silently treated as
    facts.  The parser is deterministic and does not use a model.
    """
    allowed = None if allowed_ids is None else {int(i) for i in allowed_ids}
    selected: list[TeacherMessage] = []
    for message in messages:
        if message.role != "user" or message.session_id != session_id:
            continue
        if allowed is not None and message.message_id not in allowed:
            continue
        selected.append(message)
    selected.sort(key=lambda m: m.message_id)

    state = EvidenceState(
        session_id=session_id,
        messages={m.message_id: m for m in selected},
        allowed_ids={m.message_id for m in selected},
    )
    latest_rule_id: int | None = None
    for message in selected:
        text = _strip(message.content)
        mid = message.message_id
        parsed = False
        match = _RULE_RE.fullmatch(text)
        if match:
            x, y, y2, z, x2, z2 = (
                match.group("x"), match.group("y"), match.group("y2"),
                match.group("z"), match.group("x2"), match.group("z2"),
            )
            if y == y2 and x == x2 and z == z2:
                rule = Rule(
                    rule_id=mid,
                    x=x, y=y, z=z,
                    condition=match.group("condition"),
                    exception=match.group("exception"),
                    quote=text,
                )
                state.rules[mid] = rule
                latest_rule_id = mid
                parsed = True
        if not parsed:
            match = _STATUS_COND_RE.fullmatch(text)
            if match:
                name = match.group("name")
                state.statuses[("condition", name)] = Status(
                    kind="condition", name=name,
                    value=match.group("value") == "true",
                    source_message_id=mid, quote=text,
                )
                parsed = True
        if not parsed:
            match = _STATUS_EXC_RE.fullmatch(text)
            if match:
                name = match.group("name")
                state.statuses[("exception", name)] = Status(
                    kind="exception", name=name,
                    value=match.group("value") == "active",
                    source_message_id=mid, quote=text,
                )
                parsed = True
        if not parsed:
            match = _REVOKE_RE.fullmatch(text)
            if match and latest_rule_id is not None:
                state.revoked_rules.add(latest_rule_id)
                parsed = True
        if not parsed:
            match = _FACT_RE.fullmatch(text) or _POS_FACT_RE.fullmatch(text)
            if match:
                negative = bool(match.groupdict().get("neg"))
                subject = match.group("subject")
                object_ = match.group("object")
                state.facts[(subject, object_)] = Fact(
                    subject=subject,
                    predicate="links",
                    object=object_,
                    polarity="negative" if negative else "positive",
                    source_message_id=mid,
                    quote=text,
                )
                parsed = True
        if not parsed:
            state.unsupported_ids.add(mid)
    return state


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(_IDENT, value):
        raise ContractError("schema", f"{field_name} must be an identifier")
    return value


def _require_claim_id(value: Any) -> str:
    if not isinstance(value, str) or not _CLAIM_ID_RE.fullmatch(value):
        raise ContractError("schema", "invalid claim_id")
    return value


def _require_object(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("schema", f"{name} must be an object")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ContractError("schema", f"{name} has unknown fields")
    if missing:
        raise ContractError("schema", f"{name} is missing fields")
    return value


def _parse_claim(raw: Any) -> dict[str, Any]:
    obj = _require_object(
        raw,
        {"claim_id", "kind", "subject", "predicate", "object", "polarity",
         "source_message_id", "quote"},
        "claim",
    )
    if obj["kind"] != "direct":
        raise ContractError("schema", "claim kind must be direct")
    if obj["predicate"] != "links":
        raise ContractError("schema", "claim predicate must be links")
    if obj["polarity"] not in POLARITIES:
        raise ContractError("schema", "invalid polarity")
    if not _is_int(obj["source_message_id"]):
        raise ContractError("schema", "source_message_id must be int")
    if not isinstance(obj["quote"], str) or not obj["quote"]:
        raise ContractError("schema", "quote must be non-empty")
    return {
        "claim_id": _require_claim_id(obj["claim_id"]),
        "kind": "direct",
        "subject": _require_identifier(obj["subject"], "subject"),
        "predicate": "links",
        "object": _require_identifier(obj["object"], "object"),
        "polarity": obj["polarity"],
        "source_message_id": obj["source_message_id"],
        "quote": obj["quote"],
    }


def _parse_conclusion(raw: Any) -> dict[str, Any]:
    obj = _require_object(
        raw,
        {"subject", "predicate", "object", "polarity"},
        "conclusion",
    )
    if obj["predicate"] != "links":
        raise ContractError("schema", "conclusion predicate must be links")
    if obj["polarity"] != "positive":
        raise ContractError("schema", "R1 conclusion must be positive")
    return {
        "subject": _require_identifier(obj["subject"], "conclusion.subject"),
        "predicate": "links",
        "object": _require_identifier(obj["object"], "conclusion.object"),
        "polarity": "positive",
    }


def _parse_inference(raw: Any) -> dict[str, Any]:
    obj = _require_object(
        raw,
        {"premise_claim_ids", "rule_source_message_id",
         "condition_source_message_ids", "exception_source_message_ids",
         "conclusion"},
        "inference",
    )
    if not isinstance(obj["premise_claim_ids"], list) or len(obj["premise_claim_ids"]) != 2:
        raise ContractError("schema", "R1 requires exactly two premises")
    premise_ids = [_require_claim_id(v) for v in obj["premise_claim_ids"]]
    if len(set(premise_ids)) != 2:
        raise ContractError("schema", "premise claim ids must be unique")
    if not _is_int(obj["rule_source_message_id"]):
        raise ContractError("schema", "rule_source_message_id must be int")
    for name in ("condition_source_message_ids", "exception_source_message_ids"):
        values = obj[name]
        if not isinstance(values, list) or any(not _is_int(v) for v in values):
            raise ContractError("schema", f"{name} must be integer ids")
    return {
        "premise_claim_ids": premise_ids,
        "rule_source_message_id": obj["rule_source_message_id"],
        "condition_source_message_ids": list(obj["condition_source_message_ids"]),
        "exception_source_message_ids": list(obj["exception_source_message_ids"]),
        "conclusion": _parse_conclusion(obj["conclusion"]),
    }


def _parse_ask(raw: Any) -> dict[str, Any]:
    obj = _require_object(raw, {"kind", "subject", "object"}, "ask")
    if obj["kind"] not in ASK_KINDS:
        raise ContractError("schema", "invalid ask kind")
    return {
        "kind": obj["kind"],
        "subject": _require_identifier(obj["subject"], "ask.subject"),
        "object": _require_identifier(obj["object"], "ask.object"),
    }


def _parse_unknown_target(raw: Any) -> dict[str, Any]:
    obj = _require_object(raw, {"subject", "object"}, "unknown_target")
    return {
        "subject": _require_identifier(obj["subject"], "unknown_target.subject"),
        "object": _require_identifier(obj["object"], "unknown_target.object"),
    }


def _parse_items(raw: Mapping[str, Any] | str) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ContractError("candidate_schema", "invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ContractError("candidate_schema", "top-level must be object")
    if "items" in raw:
        if set(raw) != {"items"}:
            raise ContractError("candidate_schema", "items candidate has unknown top-level field")
        raw_items = raw["items"]
        if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= 4:
            raise ContractError("candidate_schema", "items must be a non-empty list of at most four")
        items: list[dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                raise ContractError("candidate_schema", "item must be an object")
            kind = raw_item.get("type")
            if kind == "direct":
                obj = _require_object(raw_item, {"type", "claims"}, "item")
                if not isinstance(obj["claims"], list) or not obj["claims"]:
                    raise ContractError("candidate_schema", "direct requires a non-empty claims list")
                items.append({"type": "direct", "claims": [_parse_claim(c) for c in obj["claims"]]})
            elif kind == "infer":
                obj = _require_object(raw_item, {"type", "claims", "inference"}, "item")
                if not isinstance(obj["claims"], list) or len(obj["claims"]) != 2:
                    raise ContractError("candidate_schema", "infer requires exactly two claims")
                items.append({
                    "type": "infer",
                    "claims": [_parse_claim(c) for c in obj["claims"]],
                    "inference": _parse_inference(obj["inference"]),
                })
            elif kind == "insufficient":
                obj = _require_object(raw_item, {"type", "unknown_target"}, "item")
                items.append({"type": "insufficient", "unknown_target": _parse_unknown_target(obj["unknown_target"])})
            elif kind == "ask":
                obj = _require_object(raw_item, {"type", "ask"}, "item")
                items.append({"type": "ask", "ask": _parse_ask(obj["ask"])})
            elif kind == "clarify":
                obj = _require_object(raw_item, {"type", "target"}, "item")
                items.append({"type": "clarify", "target": _parse_unknown_target(obj["target"])})
            else:
                raise ContractError("candidate_schema", "unknown item type")
        return items

    # Backward-compatible single-action shape; it is normalized to one item.
    allowed = {"action", "claims", "inference", "ask", "unknown_target"}
    if set(raw) - allowed:
        raise ContractError("candidate_schema", "unknown top-level field")
    action = raw.get("action")
    if action not in ("direct", "infer", "insufficient", "ask"):
        raise ContractError("candidate_schema", "unknown action")
    claims_raw = raw.get("claims", [])
    if not isinstance(claims_raw, list):
        raise ContractError("candidate_schema", "claims must be a list")
    claims = [_parse_claim(c) for c in claims_raw]
    inference = raw.get("inference")
    ask = raw.get("ask")
    unknown_target = raw.get("unknown_target")
    if action == "direct":
        if not claims or inference is not None or ask is not None or unknown_target is not None:
            raise ContractError("candidate_schema", "direct shape invalid")
        return [{"type": "direct", "claims": claims}]
    if action == "infer":
        if len(claims) != 2 or inference is None or ask is not None or unknown_target is not None:
            raise ContractError("candidate_schema", "infer shape invalid")
        return [{"type": "infer", "claims": claims, "inference": _parse_inference(inference)}]
    if action == "insufficient":
        if claims or inference is not None or ask is not None or unknown_target is None:
            raise ContractError("candidate_schema", "insufficient shape invalid")
        return [{"type": "insufficient", "unknown_target": _parse_unknown_target(unknown_target)}]
    if claims or inference is not None or unknown_target is not None or ask is None:
        raise ContractError("candidate_schema", "ask shape invalid")
    return [{"type": "ask", "ask": _parse_ask(ask)}]


def parse_candidate(raw: Mapping[str, Any] | str) -> dict[str, Any]:
    return {"items": _parse_items(raw)}


def _relation_targets(items: Sequence[Mapping[str, Any]]) -> set[Relation]:
    targets: set[Relation] = set()
    for item in items:
        if item["type"] == "direct":
            targets.update(Relation(c["subject"], c["object"]) for c in item["claims"])
        elif item["type"] == "infer":
            c = item["inference"]["conclusion"]
            targets.add(Relation(c["subject"], c["object"]))
        elif item["type"] == "insufficient":
            t = item["unknown_target"]
            targets.add(Relation(t["subject"], t["object"]))
        elif item["type"] in ("ask", "clarify"):
            t = item["ask"] if item["type"] == "ask" else item["target"]
            targets.add(Relation(t["subject"], t["object"]))
    return targets


def _target_in_goal(target: Relation, goal: GoalContext) -> bool:
    return target in {goal.original, *goal.current_subgoals}


def _served_target(goal: GoalContext, gap: Relation) -> Relation:
    for mapped_gap, mapped_target in goal.gap_targets:
        if mapped_gap == gap:
            return mapped_target
    return goal.original


def _condition_statuses_ok(evidence: EvidenceState, rule: Rule) -> bool:
    if rule.condition is not None:
        status = evidence.statuses.get(("condition", rule.condition))
        if status is None or not status.value:
            return False
    if rule.exception is not None:
        status = evidence.statuses.get(("exception", rule.exception))
        if status is None or status.value:
            return False
    return True


def _inference_available(target: Relation, evidence: EvidenceState) -> bool:
    for rule in evidence.rules.values():
        if rule.rule_id in evidence.revoked_rules:
            continue
        if not _condition_statuses_ok(evidence, rule):
            continue
        if rule.x != target.subject or rule.z != target.object:
            continue
        first = evidence.facts.get((rule.x, rule.y))
        second = evidence.facts.get((rule.y, rule.z))
        if first and second and first.polarity == second.polarity == "positive":
            return True
    return False


def _derivation_available(target: Relation, evidence: EvidenceState) -> bool:
    direct = evidence.facts.get((target.subject, target.object))
    if direct is not None:
        return True
    for rule in evidence.rules.values():
        if rule.rule_id in evidence.revoked_rules:
            continue
        if not _condition_statuses_ok(evidence, rule):
            continue
        if rule.x != target.subject or rule.z != target.object:
            continue
        first = evidence.facts.get((rule.x, rule.y))
        second = evidence.facts.get((rule.y, rule.z))
        if first and second and first.polarity == second.polarity == "positive":
            return True
    return False


def _reject(code: str, state: InteractionState, detail: str = "") -> ValidationResult:
    return ValidationResult(False, "rejected", code, None, state, detail=detail)


def _validate_claim(claim: Mapping[str, Any], evidence: EvidenceState) -> Fact:
    sid = claim["source_message_id"]
    if sid not in evidence.messages:
        raise ContractError("source_not_allowed", "source is not an allowed teacher message")
    if sid in evidence.unsupported_ids:
        raise ContractError("unsupported", "source message is not supported by the grammar")
    fact = evidence.facts.get((claim["subject"], claim["object"]))
    if fact is None:
        raise ContractError("source_not_fact", "source does not state this relation")
    if fact.source_message_id != sid:
        raise ContractError("source_stale", "source is not the current statement")
    if fact.polarity != claim["polarity"]:
        raise ContractError("source_mismatch", "polarity does not match source")
    if claim["quote"] not in evidence.messages[sid].content:
        raise ContractError("quote_mismatch", "quote is not in source")
    return fact


def _status_for(evidence: EvidenceState, kind: str, name: str) -> Status | None:
    return evidence.statuses.get((kind, name))


def _check_status(*, evidence: EvidenceState, kind: str, name: str, expected: bool, supplied_ids: Sequence[int], code_unmet: str) -> int:
    status = _status_for(evidence, kind, name)
    if status is None:
        raise ContractError("condition_unknown", f"{kind} {name} has no status")
    if status.value != expected:
        raise ContractError(code_unmet, f"{kind} {name} is not satisfied")
    if status.source_message_id not in supplied_ids:
        raise ContractError("condition_source_missing", f"{kind} {name} source is missing")
    return status.source_message_id


def _render_direct(claims: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for claim in claims:
        if claim["polarity"] == "positive":
            text = f"Yes — {claim['subject']} links to {claim['object']}"
        else:
            text = f"No — {claim['subject']} does not link to {claim['object']}"
        parts.append(f"{text} [message {claim['source_message_id']}].")
    return " ".join(parts)


def _render_inferred(claims: Sequence[Mapping[str, Any]], inference: Mapping[str, Any]) -> str:
    conclusion = inference["conclusion"]
    premise_text = ", ".join(
        f"{c['subject']} links to {c['object']} [message {c['source_message_id']}]"
        for c in claims
    )
    return (
        f"Inferred [inferred]: {conclusion['subject']} links to "
        f"{conclusion['object']}, using {premise_text} and the rule "
        f"[message {inference['rule_source_message_id']}]."
    )


def _render_insufficient(target: Mapping[str, str]) -> str:
    return (
        f"I cannot confirm whether {target['subject']} links to "
        f"{target['object']} from the teaching provided."
    )


def _render_ask(ask: Mapping[str, str]) -> str:
    if ask["kind"] == "missing_rule":
        return (
            f"Could you teach me the rule that would let me determine whether "
            f"{ask['subject']} links to {ask['object']}?"
        )
    return (
        f"Could you teach me whether {ask['subject']} links to {ask['object']}?"
    )

def _render_item(item: Mapping[str, Any], goal: GoalContext | None) -> str:
    kind = item["type"]
    if kind == "direct":
        return _render_direct(item["claims"])
    if kind == "infer":
        return _render_inferred(item["claims"], item["inference"])
    if kind == "ask":
        return _render_ask(item["ask"])
    if kind == "clarify":
        target = item["target"]
        return (
            f"Could you clarify which teaching you mean for "
            f"{target['subject']} links to {target['object']}?"
        )
    target = item["unknown_target"]
    base = (
        f"I cannot confirm whether {target['subject']} links to "
        f"{target['object']} from the teaching provided."
    )
    if goal is None or Relation(target["subject"], target["object"]) != goal.original:
        return base
    missing = []
    missing.extend(f"{r.subject} links to {r.object}" for r in goal.missing_relations)
    if goal.missing_rule:
        missing.append("the composition rule")
    if not missing:
        return base
    return base + " The missing teaching is " + ", ".join(missing) + "."


def validate_candidate(
    raw: Mapping[str, Any] | str,
    *,
    strategy: str,
    evidence: EvidenceState,
    state: InteractionState | None = None,
    turn_context: TurnContext | None = None,
) -> ValidationResult:
    """Validate a single- or multi-item candidate against server-owned context.

    The context is the server's frozen request/target state, not a model claim.
    Ask/unknown actions require it; direct/infer still validate evidence.
    """
    if strategy not in STRATEGIES:
        raise ValueError("unknown strategy")
    state = state or InteractionState()
    try:
        items = _parse_items(raw)
    except ContractError as exc:
        return _reject(exc.code, state)

    if state.closed:
        return _reject("interaction_closed", state)
    if turn_context is None:
        return _reject("missing_context", state)
    if turn_context.goal is None:
        return _reject("missing_context", state)
    # Any unparsed teaching is potentially relevant.  Do not certify a
    # verified answer while a later correction/rule could be hidden in it.
    if evidence.unsupported_ids:
        return _reject("unsupported_evidence", state)

    infer_count = sum(1 for item in items if item["type"] == "infer")
    question_items = [item for item in items if item["type"] in ("ask", "clarify")]
    if infer_count > 1:
        return _reject("inference_limit", state)
    if len(question_items) > 1:
        return _reject("question_limit", state)
    if question_items and items[-1] not in question_items:
        return _reject("question_must_be_last", state)

    if strategy == "standard" and any(
        item["type"] == "ask" for item in items
    ):
        return _reject("strategy_forbidden", state)
    if strategy == "standard" and infer_count:
        return _reject("strategy_forbidden", state)
    if strategy == "a" and any(item["type"] == "ask" for item in items):
        return _reject("strategy_forbidden", state)
    if strategy == "b" and infer_count:
        return _reject("strategy_forbidden", state)

    context_goal = turn_context.goal
    allow_infer = strategy in ("a", "ab") and turn_context.allow_infer
    allow_ask = strategy in ("b", "ab") and turn_context.allow_ask
    allow_clarify = turn_context.allow_clarify

    if infer_count and (not allow_infer or turn_context.turn_kind == "teaching"):
        return _reject("turn_inference_forbidden", state)

    allowed_targets = {
        context_goal.original,
        *context_goal.current_subgoals,
        *turn_context.required_targets,
    }
    def _allowed_target(relation: Relation) -> bool:
        return relation in allowed_targets

    used_ids: list[int] = []
    rendered: list[str] = []
    targets: set[Relation] = set()
    question_used = False

    try:
        for item in items:
            kind = item["type"]
            if kind == "ask":
                if context_goal is None:
                    return _reject("missing_context", state)
                if not allow_ask:
                    return _reject("turn_question_forbidden", state)
                if state.asked >= 1:
                    return _reject("question_limit", state)
                if state.remediated >= 1:
                    return _reject("question_after_remediation", state)
                ask = item["ask"]
                target = Relation(ask["subject"], ask["object"])
                served = _served_target(context_goal, target)
                if not _allowed_target(served):
                    return _reject("question_target_unbound", state)
                if evidence.facts.get((target.subject, target.object)) is not None:
                    return _reject("question_unnecessary", state)
                if evidence.facts.get((served.subject, served.object)) is not None:
                    return _reject("question_unnecessary", state)
                if ask["kind"] == "missing_fact":
                    if target == served:
                        return _reject("ask_targets_goal", state)
                    if target == context_goal.original:
                        return _reject("ask_targets_goal", state)
                    if target not in context_goal.askable_gaps:
                        return _reject("ask_not_allowed_gap", state)
                elif ask["kind"] == "missing_rule":
                    if not context_goal.missing_rule or target != context_goal.original:
                        return _reject("ask_not_allowed_rule", state)
                elif ask["kind"] == "direct_target":
                    if strategy != "b" or target != context_goal.original:
                        return _reject("direct_target_forbidden", state)
                if ask["kind"] != "direct_target" and _derivation_available(target, evidence):
                    return _reject("question_unnecessary", state)
                rendered.append(_render_item(item, context_goal))
                question_used = True
                continue

            if kind == "clarify":
                if context_goal is None:
                    return _reject("missing_context", state)
                if not allow_clarify:
                    return _reject("turn_clarification_forbidden", state)
                if state.asked >= 1:
                    return _reject("question_limit", state)
                target = item["target"]
                if not _target_in_goal(Relation(target["subject"], target["object"]), context_goal):
                    return _reject("clarify_unrelated_target", state)
                rendered.append(_render_item(item, context_goal))
                question_used = True
                continue

            if kind == "insufficient":
                if context_goal is None:
                    return _reject("missing_context", state)
                target = item["unknown_target"]
                relation = Relation(target["subject"], target["object"])
                allowed = {context_goal.original, *context_goal.current_subgoals}
                if turn_context is not None:
                    allowed.update(turn_context.required_targets)
                if relation not in allowed:
                    return _reject("unknown_unrelated_target", state)
                direct_fact = evidence.facts.get((relation.subject, relation.object))
                if direct_fact is not None:
                    return _reject("target_already_available", state)
                if (
                    _inference_available(relation, evidence)
                    and strategy in ("a", "ab")
                    and allow_infer
                    and turn_context.turn_kind != "teaching"
                ):
                    return _reject("target_already_available", state)
                if not _allowed_target(relation):
                    return _reject("target_not_requested", state)
                targets.add(relation)
                rendered.append(_render_item(item, context_goal))
                continue

            if kind == "direct":
                for claim in item["claims"]:
                    relation = Relation(claim["subject"], claim["object"])
                    if not _allowed_target(relation):
                        return _reject("target_not_requested", state)
                    used_ids.append(claim["source_message_id"])
                    targets.add(relation)
                rendered.append(_render_item(item, context_goal))
                continue

            # infer
            inference = item["inference"]
            premise_ids = inference["premise_claim_ids"]
            by_id = {c["claim_id"]: c for c in item["claims"]}
            if set(by_id) != set(premise_ids):
                return _reject("inference_premises", state)
            p1 = by_id[premise_ids[0]]
            p2 = by_id[premise_ids[1]]
            if p1["polarity"] != "positive" or p2["polarity"] != "positive":
                return _reject("inference_requires_positive", state)
            if p1["object"] != p2["subject"]:
                return _reject("inference_chain", state)
            rule_id = inference["rule_source_message_id"]
            if rule_id not in evidence.messages:
                return _reject("source_not_allowed", state)
            if rule_id in evidence.unsupported_ids:
                return _reject("unsupported", state)
            rule = evidence.rules.get(rule_id)
            if rule is None:
                return _reject("rule_unknown", state)
            if rule_id in evidence.revoked_rules:
                return _reject("rule_revoked", state)
            if not (
                p1["subject"] == rule.x and p1["object"] == rule.y
                and p2["subject"] == rule.y and p2["object"] == rule.z
            ):
                return _reject("rule_mismatch", state)
            conclusion = inference["conclusion"]
            relation = Relation(conclusion["subject"], conclusion["object"])
            if not _allowed_target(relation):
                return _reject("target_not_requested", state)
            if relation != Relation(rule.x, rule.z):
                return _reject("conclusion_mismatch", state)
            if evidence.facts.get((relation.subject, relation.object)) is not None:
                return _reject("direct_precedence", state)
            for claim in item["claims"]:
                used_ids.append(claim["source_message_id"])
            used_ids.extend([rule_id] + list(inference["condition_source_message_ids"]) + list(inference["exception_source_message_ids"]))
            targets.add(relation)
            rendered.append(_render_item(item, context_goal))

        # Validate all claims against evidence after shape/kind checks.
        for item in items:
            if item["type"] in ("direct", "infer"):
                for claim in item["claims"]:
                    _validate_claim(claim, evidence)
            if item["type"] == "infer":
                inference = item["inference"]
                rule_id = inference["rule_source_message_id"]
                if rule_id not in evidence.messages:
                    return _reject("source_not_allowed", state)
                if rule_id in evidence.unsupported_ids:
                    return _reject("unsupported", state)
                rule = evidence.rules.get(rule_id)
                if rule is None:
                    return _reject("rule_unknown", state)
                if rule_id in evidence.revoked_rules:
                    return _reject("rule_revoked", state)
                if rule.condition is not None:
                    condition_ids = list(inference["condition_source_message_ids"])
                    source_id = _check_status(evidence=evidence, kind="condition", name=rule.condition, expected=True, supplied_ids=condition_ids, code_unmet="condition_unmet")
                    if set(condition_ids) != {source_id}:
                        return _reject("condition_source_extra", state)
                elif inference["condition_source_message_ids"]:
                    return _reject("condition_unexpected", state)
                if rule.exception is not None:
                    exception_ids = list(inference["exception_source_message_ids"])
                    source_id = _check_status(evidence=evidence, kind="exception", name=rule.exception, expected=False, supplied_ids=exception_ids, code_unmet="exception_active")
                    if set(exception_ids) != {source_id}:
                        return _reject("exception_source_extra", state)
                elif inference["exception_source_message_ids"]:
                    return _reject("exception_unexpected", state)

        if len(targets) != sum(
            len(i["claims"]) if i["type"] == "direct" else 1
            for i in items if i["type"] in ("direct", "infer", "insufficient")
        ):
            return _reject("duplicate_target", state)

        if turn_context.required_targets:
            required = set(turn_context.required_targets)
            if not required.issubset(targets):
                return _reject("missing_subanswer", state)

        final_text = " ".join(rendered).strip()
        if not final_text:
            return _reject("empty_final", state)
        status = "verified"
        if question_used and len(items) == 1:
            status = "verified_ask" if items[0]["type"] == "ask" else "verified_clarify"
        return ValidationResult(
            True,
            status,
            None,
            final_text,
            state.next_ask() if question_used else state,
            tuple(dict.fromkeys(used_ids)),
        )
    except ContractError as exc:
        return _reject(exc.code, state)
