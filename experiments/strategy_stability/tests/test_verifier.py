import pytest

from experiments.strategy_stability.verifier import (
    ContractError,
    GoalContext,
    InteractionState,
    Relation,
    TeacherMessage,
    TurnContext,
    parse_candidate,
    parse_teacher_evidence,
    validate_candidate as _validate_candidate,
)


def ctx(
    original=("A", "C"),
    *,
    gaps=(),
    subgoals=(),
    missing=(),
    missing_rule=False,
    gap_targets=(),
    turn_kind="question",
    allow_infer=True,
    allow_ask=True,
    allow_clarify=False,
    required=(),
):
    return TurnContext(
        goal=GoalContext(
            original=Relation(*original),
            current_subgoals=tuple(Relation(*x) for x in subgoals),
            askable_gaps=tuple(Relation(*x) for x in gaps),
            missing_relations=tuple(Relation(*x) for x in missing),
            missing_rule=missing_rule,
            gap_targets=tuple((Relation(*g), Relation(*t)) for g, t in gap_targets),
        ),
        turn_kind=turn_kind,
        allow_infer=allow_infer,
        allow_ask=allow_ask,
        allow_clarify=allow_clarify,
        required_targets=tuple(Relation(*x) for x in required),
    )

_MISSING = object()


def _candidate_targets(raw):
    if not isinstance(raw, dict):
        return []
    items = raw.get("items")
    if items is None:
        items = [raw]
    if not isinstance(items, list):
        return []
    targets = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type") or item.get("action")
        if kind == "direct":
            for c in item.get("claims") or []:
                if isinstance(c, dict) and c.get("subject") and c.get("object"):
                    targets.append((c["subject"], c["object"]))
        elif kind == "infer":
            c = (item.get("inference") or {}).get("conclusion") or {}
            if c.get("subject") and c.get("object"):
                targets.append((c["subject"], c["object"]))
        elif kind in ("insufficient", "ask", "clarify"):
            target = item.get("unknown_target") or item.get("ask") or item.get("target") or {}
            if target.get("subject") and target.get("object"):
                targets.append((target["subject"], target["object"]))
    return targets


def validate_candidate(raw, *, strategy, evidence, state=None, turn_context=_MISSING):
    if turn_context is _MISSING:
        targets = _candidate_targets(raw)
        original = targets[0] if targets else ("A", "C")
        turn_context = ctx(
            original=original,
            required=tuple(targets),
            allow_infer=True,
            allow_ask=True,
        )
    return _validate_candidate(
        raw,
        strategy=strategy,
        evidence=evidence,
        state=state,
        turn_context=turn_context,
    )


def evidence(*texts, session_id=1, allowed_ids=None):
    messages = [
        TeacherMessage(i + 1, session_id, "user", text)
        for i, text in enumerate(texts)
    ]
    return parse_teacher_evidence(
        messages, session_id=session_id, allowed_ids=allowed_ids
    )


def claim(cid, s, o, *, polarity="positive", source=1, quote=None):
    if quote is None:
        quote = f"{s} {'does not link' if polarity == 'negative' else 'links'} to {o}."
    return {
        "claim_id": cid,
        "kind": "direct",
        "subject": s,
        "predicate": "links",
        "object": o,
        "polarity": polarity,
        "source_message_id": source,
        "quote": quote,
    }


def infer(p1, p2, *, rule=3, conclusion=("A", "C"), conditions=(), exceptions=()):
    return {
        "action": "infer",
        "claims": [p1, p2],
        "inference": {
            "premise_claim_ids": [p1["claim_id"], p2["claim_id"]],
            "rule_source_message_id": rule,
            "condition_source_message_ids": list(conditions),
            "exception_source_message_ids": list(exceptions),
            "conclusion": {
                "subject": conclusion[0],
                "predicate": "links",
                "object": conclusion[1],
                "polarity": "positive",
            },
        },
    }


def test_valid_direct_and_negative():
    ev = evidence("A links to B.", "Correction: C does not link to D.")
    r = validate_candidate({"action": "direct", "claims": [claim("c1", "A", "B")]},
                           strategy="standard", evidence=ev)
    assert r.accepted and r.final_text == "Yes — A links to B [message 1]."
    r = validate_candidate(
        {"action": "direct", "claims": [claim("c2", "C", "D", polarity="negative", source=2)]},
        strategy="standard", evidence=ev,
    )
    assert r.accepted and r.final_text.startswith("No — C does not link to D")


def test_direct_requires_real_current_fact_and_quote():
    ev = evidence("A links to B.", "Correction: A does not link to B.")
    r = validate_candidate(
        {"action": "direct", "claims": [claim("c", "A", "B", source=1)]},
        strategy="standard", evidence=ev,
    )
    assert not r.accepted and r.code == "source_stale"
    r = validate_candidate(
        {"action": "direct", "claims": [claim("c", "A", "B", polarity="negative", source=2, quote="A links to Z")]},
        strategy="standard", evidence=ev,
    )
    assert not r.accepted and r.code == "quote_mismatch"


def test_assistant_and_cross_session_sources_are_rejected():
    ev = parse_teacher_evidence(
        [
            TeacherMessage(1, 1, "user", "A links to B."),
            TeacherMessage(2, 1, "assistant", "A links to C."),
            TeacherMessage(3, 2, "user", "A links to D."),
        ],
        session_id=1,
    )
    for sid in (2, 3):
        r = validate_candidate(
            {"action": "direct", "claims": [claim("c", "A", "C", source=sid)]},
            strategy="standard", evidence=ev,
        )
        assert not r.accepted and r.code == "source_not_allowed"


def test_rule_observation_is_not_an_adopted_rule():
    ev = evidence("Rule: if A links to B and B links to C then A links to C.", "A links to B.", "B links to C.")
    # The rule is a teacher-authored statement, so it is adopted.  A merely
    # mentioned rule is represented as an unsupported message instead.
    assert validate_candidate(infer(claim("p1", "A", "B", source=2), claim("p2", "B", "C", source=3), rule=1),
                              strategy="a", evidence=ev).accepted
    ev2 = evidence("I mentioned a rule: A B C.", "A links to B.", "B links to C.")
    r = validate_candidate(infer(claim("p1", "A", "B", source=2), claim("p2", "B", "C", source=3), rule=3),
                           strategy="a", evidence=ev2)
    assert not r.accepted and r.code == "unsupported_evidence"


def test_valid_r1_and_reverse_or_two_step_rejected():
    ev = evidence(
        "A links to B.", "B links to C.",
        "Rule: if A links to B and B links to C then A links to C.",
    )
    r = validate_candidate(infer(claim("p1", "A", "B"), claim("p2", "B", "C", source=2)),
                           strategy="a", evidence=ev)
    assert r.accepted and r.status == "verified"
    assert "[inferred]" in r.final_text and r.used_source_ids == (1, 2, 3)
    r = validate_candidate(infer(claim("p1", "B", "C", source=2), claim("p2", "A", "B", source=1)),
                           strategy="a", evidence=ev)
    assert not r.accepted and r.code in {"inference_chain", "rule_mismatch"}
    r = validate_candidate(
        infer(claim("p1", "A", "B"), claim("p2", "B", "D", source=2)),
        strategy="a", evidence=ev,
    )
    assert not r.accepted and r.code in {"source_not_fact", "rule_mismatch"}


def test_direct_precedence_blocks_inference():
    ev = evidence(
        "A links to B.", "B links to C.", "A links to D.", "A links to C.",
        "Rule: if A links to B and B links to C then A links to C.",
    )
    r = validate_candidate(infer(claim("p1", "A", "B"), claim("p2", "B", "C", source=2), rule=5),
                           strategy="a", evidence=ev)
    assert not r.accepted and r.code == "direct_precedence"


def test_strategy_permissions():
    ev = evidence("A links to B.", "B links to C.", "Rule: if A links to B and B links to C then A links to C.")
    base = infer(claim("p1", "A", "B"), claim("p2", "B", "C", source=2))
    assert not validate_candidate(base, strategy="standard", evidence=ev).accepted
    assert validate_candidate(base, strategy="a", evidence=ev).accepted
    assert not validate_candidate(base, strategy="b", evidence=ev).accepted
    assert validate_candidate(base, strategy="ab", evidence=ev).accepted
    ask = {"action": "ask", "ask": {"kind": "missing_fact", "subject": "C", "object": "D"}}
    qctx = ctx(gaps=(("C", "D"),))
    assert not validate_candidate(ask, strategy="standard", evidence=ev, turn_context=qctx).accepted
    assert not validate_candidate(ask, strategy="a", evidence=ev, turn_context=qctx).accepted
    assert validate_candidate(ask, strategy="b", evidence=ev, turn_context=qctx).accepted
    assert validate_candidate(ask, strategy="ab", evidence=ev, turn_context=qctx).accepted
    direct_target = {"action": "ask", "ask": {"kind": "direct_target", "subject": "A", "object": "C"}}
    target_ctx = ctx()
    assert validate_candidate(direct_target, strategy="b", evidence=ev, turn_context=target_ctx).accepted
    assert not validate_candidate(direct_target, strategy="ab", evidence=ev, turn_context=target_ctx).accepted

def test_condition_and_exception_are_current_and_cited():
    ev = evidence(
        "A links to B.", "B links to C.",
        "Rule: if A links to B and B links to C then A links to C. Condition: ready. Exception: blocked.",
        "Condition ready is true.", "Exception blocked is inactive.",
    )
    base = infer(claim("p1", "A", "B"), claim("p2", "B", "C", source=2), conditions=(4,), exceptions=(5,))
    assert validate_candidate(base, strategy="a", evidence=ev).accepted
    bad = infer(claim("p1", "A", "B"), claim("p2", "B", "C", source=2), conditions=(), exceptions=(5,))
    assert not validate_candidate(bad, strategy="a", evidence=ev).accepted
    ev2 = evidence(
        "A links to B.", "B links to C.",
        "Rule: if A links to B and B links to C then A links to C. Condition: ready.",
        "Condition ready is false.",
    )
    r = validate_candidate(
        infer(claim("p1", "A", "B"), claim("p2", "B", "C", source=2), conditions=(4,)),
        strategy="a", evidence=ev2,
    )
    assert not r.accepted and r.code == "condition_unmet"


def test_conditions_cannot_be_supplied_when_rule_has_none():
    ev = evidence(
        "A links to B.", "B links to C.",
        "Rule: if A links to B and B links to C then A links to C.",
        "Condition ready is true.",
    )
    r = validate_candidate(
        infer(claim("p1", "A", "B"), claim("p2", "B", "C", source=2), conditions=(4,)),
        strategy="a", evidence=ev,
    )
    assert not r.accepted and r.code == "condition_unexpected"


def test_revoked_rule_and_old_correction_are_rejected():
    ev = evidence(
        "A links to B.", "B links to C.",
        "Rule: if A links to B and B links to C then A links to C.",
        "Revoke rule.",
    )
    r = validate_candidate(infer(claim("p1", "A", "B"), claim("p2", "B", "C", source=2)),
                           strategy="a", evidence=ev)
    assert not r.accepted and r.code == "rule_revoked"
    ev2 = evidence("A links to B.", "Correction: A does not link to B.")
    r = validate_candidate(
        {"action": "direct", "claims": [claim("c", "A", "B", source=1)]},
        strategy="standard", evidence=ev2,
    )
    assert not r.accepted and r.code == "source_stale"


def test_question_limit_state_and_insufficient_no_answer():
    ev = evidence("A links to B.")
    ask = {"action": "ask", "ask": {"kind": "missing_fact", "subject": "C", "object": "D"}}
    qctx = ctx(gaps=(("C", "D"),))
    first = validate_candidate(ask, strategy="b", evidence=ev, turn_context=qctx)
    assert first.accepted and first.next_state.asked == 1
    second = validate_candidate(ask, strategy="b", evidence=ev, state=first.next_state, turn_context=qctx)
    assert not second.accepted and second.code == "question_limit"
    remediated = validate_candidate(ask, strategy="b", evidence=ev, state=InteractionState(remediated=1), turn_context=qctx)
    assert not remediated.accepted and remediated.code == "question_after_remediation"
    unknown = validate_candidate(
        {"action": "insufficient", "unknown_target": {"subject": "B", "object": "C"}},
        strategy="standard", evidence=ev, turn_context=ctx(subgoals=(("B", "C"),)),
    )
    assert unknown.accepted and unknown.status == "verified"
    assert "cannot confirm" in unknown.final_text


def test_candidate_cannot_supply_final_prose():
    ev = evidence("A links to B.")
    raw = {"action": "direct", "claims": [claim("c", "A", "B")], "final_text": "A links to B."}
    r = validate_candidate(raw, strategy="standard", evidence=ev)
    assert not r.accepted and r.code == "candidate_schema"


def test_unsupported_evidence_cannot_be_used():
    ev = evidence("A links to B.", "I had a thought about C.")
    r = validate_candidate(
        {"action": "direct", "claims": [claim("c", "A", "B", source=2, quote="I had a thought about C.")]},
        strategy="standard", evidence=ev,
    )
    assert not r.accepted and r.code == "unsupported_evidence"


def test_malformed_candidate_json_is_rejected():
    with pytest.raises(ContractError):
        parse_candidate("{not json")


def test_exception_active_blocks_otherwise_valid_r1():
    ev = evidence(
        "A links to B.", "B links to C.",
        "Rule: if A links to B and B links to C then A links to C. Exception: blocked.",
        "Exception blocked is active.",
    )
    r = validate_candidate(
        infer(claim("p1", "A", "B"), claim("p2", "B", "C", source=2), exceptions=(4,)),
        strategy="a", evidence=ev,
    )
    assert not r.accepted and r.code == "exception_active"


def test_missing_rule_ask_is_a_plain_question_not_free_text():
    ev = evidence("A links to B.")
    candidate = {
        "action": "ask",
        "ask": {"kind": "missing_rule", "subject": "A", "object": "C"},
    }
    r = validate_candidate(candidate, strategy="b", evidence=ev, turn_context=ctx(missing_rule=True))
    assert r.accepted and r.status == "verified_ask"
    assert r.final_text.startswith("Could you teach me the rule")
    assert "?" in r.final_text and r.next_state.asked == 1


def test_b_direct_report_and_b_direct_target_ask_are_allowed():
    ev = evidence("A links to B.")
    direct = validate_candidate(
        {"action": "direct", "claims": [claim("c", "A", "B")]},
        strategy="b", evidence=ev,
    )
    assert direct.accepted
    ask = validate_candidate(
        {"action": "ask", "ask": {"kind": "direct_target", "subject": "A", "object": "C"}},
        strategy="b", evidence=ev, turn_context=ctx(),
    )
    assert ask.accepted


def test_extra_inference_claims_are_schema_rejected():
    ev = evidence(
        "A links to B.", "B links to C.", "C links to D.",
        "Rule: if A links to B and B links to C then A links to C.",
    )
    candidate = infer(claim("p1", "A", "B"), claim("p2", "B", "C", source=2))
    candidate["claims"].append(claim("p3", "C", "D", source=3))
    r = validate_candidate(candidate, strategy="a", evidence=ev)
    assert not r.accepted and r.code == "candidate_schema"


def test_question_limit_does_not_apply_to_insufficient_or_direct():
    ev = evidence("A links to B.")
    state = InteractionState(asked=1, remediated=1)
    insufficient = validate_candidate(
        {"action": "insufficient", "unknown_target": {"subject": "B", "object": "C"}},
        strategy="b", evidence=ev, state=state,
        turn_context=ctx(subgoals=(("B", "C"),)),
    )
    assert insufficient.accepted
    direct = validate_candidate(
        {"action": "direct", "claims": [claim("c", "A", "B")]},
        strategy="b", evidence=ev, state=state,
    )
    assert direct.accepted


def test_standard_and_b_cannot_report_untaught_relation():
    ev = evidence("A links to B.")
    candidate = {"action": "direct", "claims": [claim("c", "A", "C", source=1)]}
    for strategy in ("standard", "b"):
        r = validate_candidate(candidate, strategy=strategy, evidence=ev)
        assert not r.accepted and r.code == "source_not_fact"


def test_wrong_polarity_is_rejected():
    ev = evidence("A links to B.")
    candidate = {
        "action": "direct",
        "claims": [claim("c", "A", "B", polarity="negative", source=1)],
    }
    r = validate_candidate(candidate, strategy="standard", evidence=ev)
    assert not r.accepted and r.code == "source_mismatch"


def _items(*items):
    return {"items": list(items)}


def test_goal_context_rejects_disguised_and_unrelated_questions():
    ev = evidence("A links to B.", "B links to C.")
    context = ctx(gaps=(("B", "C"),))
    disguised = {"items": [{"type": "ask", "ask": {"kind": "missing_fact", "subject": "A", "object": "C"}}]}
    r = validate_candidate(disguised, strategy="ab", evidence=ev, turn_context=context)
    assert not r.accepted and r.code == "ask_targets_goal"
    unrelated = {"items": [{"type": "ask", "ask": {"kind": "missing_fact", "subject": "Z", "object": "W"}}]}
    r = validate_candidate(unrelated, strategy="ab", evidence=ev, turn_context=context)
    assert not r.accepted and r.code == "ask_not_allowed_gap"


def test_unknown_is_not_allowed_when_target_is_available():
    ev = evidence("A links to C.")
    r = validate_candidate(
        {"items": [{"type": "insufficient", "unknown_target": {"subject": "A", "object": "C"}}]},
        strategy="standard",
        evidence=ev,
        turn_context=ctx(original=("A", "C")),
    )
    assert not r.accepted and r.code == "target_already_available"


def test_unparsed_correction_blocks_any_verified_answer():
    ev = evidence("A links to B.", "Correction: A is not linked to B.")
    r = validate_candidate(
        {"items": [{"type": "direct", "claims": [claim("c", "A", "B")]}]},
        strategy="standard",
        evidence=ev,
    )
    assert not r.accepted and r.code == "unsupported_evidence"


def test_s1_multi_item_uses_one_r1_and_answers_all_subgoals():
    ev = evidence(
        "A links to B.", "B links to C.", "A links to D.",
        "Rule: if A links to B and B links to C then A links to C.",
    )
    candidate = _items(
        {"type": "direct", "claims": [claim("d", "A", "D", source=3)]},
        {"type": "infer", "claims": [claim("p1", "A", "B", source=1), claim("p2", "B", "C", source=2)],
         "inference": infer(claim("p1", "A", "B", source=1), claim("p2", "B", "C", source=2), rule=4)["inference"]},
        {"type": "insufficient", "unknown_target": {"subject": "C", "object": "A"}},
    )
    r = validate_candidate(
        candidate,
        strategy="a",
        evidence=ev,
        turn_context=ctx(
            original=("A", "C"),
            required=(("A", "D"), ("A", "C"), ("C", "A")),
            allow_infer=True,
        ),
    )
    assert r.accepted
    assert r.final_text.count("[inferred]") == 1
    assert "A links to D" in r.final_text
    assert "A links to C" in r.final_text
    assert "cannot confirm" in r.final_text
    assert "C links to A" in r.final_text


def test_s2_missing_statement_names_the_specific_gap():
    ev = evidence("A links to B.")
    r = validate_candidate(
        {"items": [{"type": "insufficient", "unknown_target": {"subject": "B", "object": "C"}}]},
        strategy="b",
        evidence=ev,
        turn_context=ctx(subgoals=(("B", "C"),), missing=(("B", "C"),)),
    )
    assert r.accepted
    assert "B links to C" in r.final_text


def test_standard_clarification_is_allowed_but_ask_is_not():
    ev = evidence("A links to B.")
    clarify = {"items": [{"type": "clarify", "target": {"subject": "A", "object": "C"}}]}
    r = validate_candidate(
        clarify,
        strategy="standard",
        evidence=ev,
        turn_context=ctx(allow_clarify=True),
    )
    assert r.accepted and r.status == "verified_clarify"
    ask = {"items": [{"type": "ask", "ask": {"kind": "missing_fact", "subject": "B", "object": "C"}}]}
    r = validate_candidate(
        ask,
        strategy="standard",
        evidence=ev,
        turn_context=ctx(gaps=(("B", "C"),)),
    )
    assert not r.accepted and r.code == "strategy_forbidden"


def test_teaching_turn_cannot_infer():
    ev = evidence(
        "A links to B.", "B links to C.",
        "Rule: if A links to B and B links to C then A links to C.",
    )
    r = validate_candidate(
        infer(claim("p1", "A", "B", source=1), claim("p2", "B", "C", source=2)),
        strategy="a",
        evidence=ev,
        turn_context=ctx(turn_kind="teaching", allow_infer=False),
    )
    assert not r.accepted and r.code == "turn_inference_forbidden"


def test_one_reply_cannot_contain_two_r1_items():
    ev = evidence(
        "A links to B.", "B links to C.", "C links to D.",
        "Rule: if A links to B and B links to C then A links to C.",
    )
    one = infer(claim("p1", "A", "B", source=1), claim("p2", "B", "C", source=2))
    two = infer(claim("q1", "A", "B", source=1), claim("q2", "B", "C", source=2))
    candidate = _items(
        {"type": "infer", "claims": one["claims"], "inference": one["inference"]},
        {"type": "infer", "claims": two["claims"], "inference": two["inference"]},
    )
    r = validate_candidate(candidate, strategy="a", evidence=ev, turn_context=ctx())
    assert not r.accepted and r.code == "inference_limit"


def test_question_cannot_be_combined_with_another_item():
    ev = evidence("A links to B.")
    candidate = _items(
        {"type": "ask", "ask": {"kind": "missing_fact", "subject": "B", "object": "C"}},
        {"type": "direct", "claims": [claim("c", "A", "B")]},
    )
    r = validate_candidate(candidate, strategy="b", evidence=ev, turn_context=ctx(gaps=(("B", "C"),)))
    assert not r.accepted and r.code == "question_must_be_last"


@pytest.mark.parametrize(
    "subject, object_",
    [("P", "Q"), ("M", "N"), ("X1", "Y2")],
)
def test_parameterized_names_are_not_hardcoded(subject, object_):
    ev = evidence(f"{subject} links to {object_}.")
    r = validate_candidate(
        {"items": [{"type": "direct", "claims": [claim("c", subject, object_)]}]},
        strategy="standard",
        evidence=ev,
    )
    assert r.accepted
    assert subject in r.final_text and object_ in r.final_text


def test_policy_requires_server_context():
    ev = evidence("A links to B.")
    raw = {"items": [{"type": "direct", "claims": [claim("c", "A", "B")]}]}
    r = _validate_candidate(raw, strategy="standard", evidence=ev, turn_context=None)
    assert not r.accepted and r.code == "missing_context"


def test_direct_claims_must_be_non_empty_and_well_typed():
    ev = evidence("A links to B.")
    for claims in ([], None):
        raw = {"items": [{"type": "direct", "claims": claims}]}
        r = _validate_candidate(raw, strategy="standard", evidence=ev, turn_context=ctx(original=("A", "B")))
        assert not r.accepted and r.code == "candidate_schema"


def test_B_and_Standard_can_withhold_when_only_R1_exists():
    ev = evidence(
        "A links to B.", "B links to C.",
        "Rule: if A links to B and B links to C then A links to C.",
    )
    raw = {"items": [{"type": "insufficient", "unknown_target": {"subject": "A", "object": "C"}}]}
    for strategy in ("b", "standard"):
        r = _validate_candidate(
            raw, strategy=strategy, evidence=ev,
            turn_context=ctx(required=(("A", "C"),)),
        )
        assert r.accepted
        assert "missing" not in r.final_text.lower()
    r = _validate_candidate(
        raw, strategy="a", evidence=ev,
        turn_context=ctx(required=(("A", "C"),)),
    )
    assert not r.accepted and r.code == "target_already_available"


def test_A_keeps_necessary_clarification_but_not_content_ask():
    ev = evidence("A links to B.")
    clarify = {"items": [{"type": "clarify", "target": {"subject": "A", "object": "C"}}]}
    r = _validate_candidate(
        clarify, strategy="a", evidence=ev,
        turn_context=ctx(allow_clarify=True, allow_ask=False),
    )
    assert r.accepted and r.status == "verified_clarify"
    ask = {"items": [{"type": "ask", "ask": {"kind": "missing_fact", "subject": "B", "object": "C"}}]}
    r = _validate_candidate(
        ask, strategy="a", evidence=ev,
        turn_context=ctx(gaps=(("B", "C"),)),
    )
    assert not r.accepted and r.code == "strategy_forbidden"


@pytest.mark.parametrize("polarity", ["positive", "negative"])
def test_known_target_cannot_be_reasked(polarity):
    text = "A links to C." if polarity == "positive" else "A does not link to C."
    ev = evidence(text)
    raw = {"items": [{"type": "ask", "ask": {"kind": "direct_target", "subject": "A", "object": "C"}}]}
    r = _validate_candidate(
        raw, strategy="b", evidence=ev,
        turn_context=ctx(allow_ask=True),
    )
    assert not r.accepted and r.code == "question_unnecessary"


def test_direct_known_subgoal_can_precede_one_gap_question():
    ev = evidence(
        "A links to D.", "A links to B.",
        "Rule: if A links to B and B links to C then A links to C.",
    )
    raw = _items(
        {"type": "direct", "claims": [claim("d", "A", "D", source=1)]},
        {"type": "ask", "ask": {"kind": "missing_fact", "subject": "B", "object": "C"}},
    )
    r = _validate_candidate(
        raw, strategy="ab", evidence=ev,
        turn_context=ctx(subgoals=(("A", "D"),), gaps=(("B", "C"),), required=(("A", "D"),)),
    )
    assert r.accepted
    assert r.final_text.rstrip().endswith("?")
    assert r.next_state.asked == 1


def test_question_must_be_last():
    ev = evidence("A links to B.")
    raw = _items(
        {"type": "ask", "ask": {"kind": "missing_fact", "subject": "B", "object": "C"}},
        {"type": "direct", "claims": [claim("c", "A", "B")]},
    )
    r = _validate_candidate(
        raw, strategy="b", evidence=ev,
        turn_context=ctx(subgoals=(("A", "B"),), gaps=(("B", "C"),)),
    )
    assert not r.accepted and r.code == "question_must_be_last"


def test_unrequested_inference_target_is_rejected():
    ev = evidence(
        "X links to Y.", "Y links to Z.",
        "Rule: if X links to Y and Y links to Z then X links to Z.",
    )
    raw = _items(
        {"type": "infer", "claims": [claim("p1", "X", "Y", source=1), claim("p2", "Y", "Z", source=2)],
         "inference": infer(claim("p1", "X", "Y", source=1), claim("p2", "Y", "Z", source=2), rule=3, conclusion=("X", "Z"))["inference"]},
        {"type": "insufficient", "unknown_target": {"subject": "A", "object": "C"}},
    )
    r = _validate_candidate(
        raw, strategy="ab", evidence=ev,
        turn_context=ctx(required=(("A", "C"),)),
    )
    assert not r.accepted and r.code == "target_not_requested"


def _qctx(original=("A", "C"), *, subgoals=(), gaps=(), required=(), gap_targets=()):
    return ctx(
        original=original,
        subgoals=subgoals,
        gaps=gaps,
        required=required,
        gap_targets=gap_targets,
        allow_ask=True,
    )


def _direct_d(item):
    return {"items": [{"type": "direct", "claims": [item]}]}


def _mixed_d_ask(item):
    return {"items": [
        {"type": "direct", "claims": [item]},
        {"type": "ask", "ask": {"kind": "missing_fact", "subject": "B", "object": "C"}},
    ]}


@pytest.mark.parametrize(
    "evidence_texts, bad_claim, code",
    [
        (("A links to B.",), claim("d", "A", "D", source=1), "source_not_fact"),
        (
            ("A links to D.", "Correction: A does not link to D.", "A links to B."),
            claim("d", "A", "D", source=1),
            "source_stale",
        ),
        (
            ("A links to D.", "A links to B."),
            dict(claim("d", "A", "D", source=1), quote="C links to Z."),
            "quote_mismatch",
        ),
    ],
)
def test_bad_fact_alone_and_before_ask(tmp_path, evidence_texts, bad_claim, code):
    ev = evidence(*evidence_texts)
    context = _qctx(subgoals=(("A", "D"),), gaps=(("B", "C"),), required=(("A", "D"),))
    for raw in (_direct_d(bad_claim), _mixed_d_ask(bad_claim)):
        state = InteractionState()
        r = _validate_candidate(
            raw, strategy="ab", evidence=ev, state=state, turn_context=context
        )
        assert not r.accepted and r.code == code
        assert r.final_text is None
        assert r.next_state == state


def test_false_condition_inference_before_ask_is_rejected_atomically():
    ev = evidence(
        "A links to B.", "B links to C.",
        "Rule: if A links to B and B links to C then A links to C. Condition: ready.",
        "Condition ready is false.", "X links to Y.",
    )
    context = _qctx(
        original=("X", "Z"), subgoals=(("A", "C"),), gaps=(("Y", "Z"),),
        required=(("A", "C"),),
    )
    bad = infer(claim("p1", "A", "B", source=1), claim("p2", "B", "C", source=2))
    bad["inference"]["condition_source_message_ids"] = [4]
    raw = {"items": [
        {"type": "infer", "claims": bad["claims"], "inference": bad["inference"]},
        {"type": "ask", "ask": {"kind": "missing_fact", "subject": "Y", "object": "Z"}},
    ]}
    state = InteractionState()
    r = _validate_candidate(raw, strategy="ab", evidence=ev, state=state, turn_context=context)
    assert not r.accepted and r.code == "condition_unmet"
    assert r.final_text is None and r.next_state == state


def test_valid_direct_plus_ask_preserves_sources_and_state():
    ev = evidence("A links to D.", "A links to B.")
    context = _qctx(
        original=("A", "C"), subgoals=(("A", "D"),), gaps=(("B", "C"),),
        required=(("A", "D"),),
    )
    r = _validate_candidate(
        _mixed_d_ask(claim("d", "A", "D", source=1)),
        strategy="ab", evidence=ev, turn_context=context,
    )
    assert r.accepted
    assert r.used_source_ids == (1,)
    assert r.next_state.asked == 1
    assert r.final_text.rstrip().endswith("?")


def test_missing_required_subanswer_is_not_hidden_by_a_question():
    ev = evidence("A links to D.", "A links to B.")
    context = _qctx(
        original=("A", "C"), subgoals=(("A", "D"), ("E", "F")),
        gaps=(("B", "C"),), required=(("A", "D"), ("E", "F")),
    )
    state = InteractionState()
    r = _validate_candidate(
        _mixed_d_ask(claim("d", "A", "D", source=1)),
        strategy="ab", evidence=ev, state=state, turn_context=context,
    )
    assert not r.accepted and r.code == "missing_subanswer"
    assert r.final_text is None and r.next_state == state


@pytest.mark.parametrize("text", ["A links to C.", "A does not link to C."])
def test_intermediate_gap_question_is_rejected_when_served_target_known(text):
    ev = evidence(text, "A links to B.")
    context = _qctx(gaps=(("B", "C"),))
    state = InteractionState()
    r = _validate_candidate(
        {"items": [{"type": "ask", "ask": {"kind": "missing_fact", "subject": "B", "object": "C"}}]},
        strategy="b", evidence=ev, state=state, turn_context=context,
    )
    assert not r.accepted and r.code == "question_unnecessary"
    assert r.next_state == state


def test_known_original_does_not_block_a_different_unresolved_subgoal_question():
    ev = evidence(
        "A links to C.", "G links to F.",
        "Rule: if E links to G and G links to F then E links to F.",
    )
    context = _qctx(
        original=("A", "C"), subgoals=(("E", "F"),), gaps=(("E", "G"),),
        gap_targets=((("E", "G"), ("E", "F")),),
    )
    r = _validate_candidate(
        {"items": [{"type": "ask", "ask": {"kind": "missing_fact", "subject": "E", "object": "G"}}]},
        strategy="b", evidence=ev, turn_context=context,
    )
    assert r.accepted and r.next_state.asked == 1


def test_active_exception_before_ask_is_rejected_atomically():
    ev = evidence(
        "A links to B.", "B links to C.",
        "Rule: if A links to B and B links to C then A links to C. "
        "Exception: flagged.",
        "Exception flagged is active.", "X links to Y.",
    )
    context = _qctx(
        original=("X", "Z"), subgoals=(("A", "C"),), gaps=(("Y", "Z"),),
        required=(("A", "C"),),
    )
    bad = infer(
        claim("p1", "A", "B", source=1),
        claim("p2", "B", "C", source=2),
        exceptions=(4,),
    )
    raw = {"items": [
        {"type": "infer", "claims": bad["claims"], "inference": bad["inference"]},
        {"type": "ask", "ask": {"kind": "missing_fact", "subject": "Y", "object": "Z"}},
    ]}
    state = InteractionState()
    r = _validate_candidate(raw, strategy="ab", evidence=ev, state=state, turn_context=context)
    assert not r.accepted and r.code == "exception_active"
    assert r.final_text is None and r.next_state == state


def test_ab_mapped_gap_cannot_be_the_served_target():
    ev = evidence("A links to D.")
    context = _qctx(
        original=("A", "D"), subgoals=(("A", "C"),), gaps=(("A", "C"),),
        gap_targets=((("A", "C"), ("A", "C")),),
    )
    raw = {"items": [
        {"type": "direct", "claims": [claim("d", "A", "D", source=1)]},
        {"type": "ask", "ask": {"kind": "missing_fact", "subject": "A", "object": "C"}},
    ]}
    state = InteractionState()
    r = _validate_candidate(raw, strategy="ab", evidence=ev, state=state, turn_context=context)
    assert not r.accepted and r.code == "ask_targets_goal"
    assert r.final_text is None and r.next_state == state


def test_ab_intermediate_gap_still_allowed_and_b_direct_target_preserved():
    ev = evidence("A links to D.", "A links to B.")
    context = _qctx(
        original=("A", "D"), subgoals=(("A", "C"),), gaps=(("B", "C"),),
        gap_targets=((("B", "C"), ("A", "C")),),
    )
    raw = {"items": [
        {"type": "direct", "claims": [claim("d", "A", "D", source=1)]},
        {"type": "ask", "ask": {"kind": "missing_fact", "subject": "B", "object": "C"}},
    ]}
    r = _validate_candidate(raw, strategy="ab", evidence=ev, turn_context=context)
    assert r.accepted and r.used_source_ids == (1,)
    direct_ctx = _qctx(original=("A", "C"), gaps=())
    r2 = _validate_candidate(
        {"items": [{"type": "ask", "ask": {"kind": "direct_target", "subject": "A", "object": "C"}}]},
        strategy="b", evidence=evidence("A links to B."), turn_context=direct_ctx,
    )
    assert r2.accepted
