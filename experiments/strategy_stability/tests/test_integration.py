"""Isolated integration tests for the strategy-stability service."""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from experiments.strategy_stability.integration_app import FakeCandidateModel, create_app
from experiments.strategy_stability.models import ExperimentMessage, ExperimentSession


def _rel(subject, object_):
    return {"subject": subject, "object": object_}


def _turn(*, original=("A", "C"), subgoals=(), gaps=(), missing_rule=False, allowed_infer=True, allowed_ask=True, allowed_clarify=False, required=()):
    return {
        "goal": {
            "original": _rel(*original),
            "current_subgoals": [_rel(*x) for x in subgoals],
            "askable_gaps": [_rel(*x) for x in gaps],
            "missing_relations": [],
            "missing_rule": missing_rule,
            "gap_targets": [],
        },
        "turn_kind": "question",
        "allow_infer": allowed_infer,
        "allow_ask": allowed_ask,
        "allow_clarify": allowed_clarify,
        "required_targets": [_rel(*x) for x in required],
    }


def _evidence(*texts):
    return [{"message_id": i + 1, "content": text} for i, text in enumerate(texts)]


def _direct(subject="A", object_="B", source=1, *, claim_id="c1"):
    return {
        "items": [
            {
                "type": "direct",
                "claims": [
                    {
                        "claim_id": claim_id,
                        "kind": "direct",
                        "subject": subject,
                        "predicate": "links",
                        "object": object_,
                        "polarity": "positive",
                        "source_message_id": source,
                        "quote": f"{subject} links to {object_}.",
                    }
                ],
            }
        ]
    }


def _infer(*, rule_source=3, second_source=2):
    return {
        "items": [
            {
                "type": "infer",
                "claims": [
                    {
                        "claim_id": "p1",
                        "kind": "direct",
                        "subject": "A",
                        "predicate": "links",
                        "object": "B",
                        "polarity": "positive",
                        "source_message_id": 1,
                        "quote": "A links to B.",
                    },
                    {
                        "claim_id": "p2",
                        "kind": "direct",
                        "subject": "B",
                        "predicate": "links",
                        "object": "C",
                        "polarity": "positive",
                        "source_message_id": second_source,
                        "quote": "B links to C.",
                    },
                ],
                "inference": {
                    "premise_claim_ids": ["p1", "p2"],
                    "rule_source_message_id": rule_source,
                    "condition_source_message_ids": [],
                    "exception_source_message_ids": [],
                    "conclusion": {
                        "subject": "A",
                        "predicate": "links",
                        "object": "C",
                        "polarity": "positive",
                    },
                },
            }
        ]
    }


def _ask(subject="B", object_="C", kind="missing_fact"):
    return {
        "items": [
            {
                "type": "ask",
                "ask": {
                    "kind": kind,
                    "subject": subject,
                    "object": object_,
                },
            }
        ]
    }


def _client(tmp_path, responses):
    app = create_app(tmp_path / "integration.sqlite3", FakeCandidateModel(responses))
    return app, TestClient(app)


def _new_session(client, *, strategy="standard", language="en", evidence=None, turn=None):
    response = client.post(
        "/sessions",
        json={
            "strategy": strategy,
            "language": language,
            "model_name": "fake-candidate-v1",
            "evidence": evidence or _evidence("A links to B."),
            "turn_context": turn or _turn(original=("A", "B")),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _post(client, session_id, content="What about A?"):
    return client.post(f"/sessions/{session_id}/messages", json={"content": content})


def _state(client, session_id):
    response = client.get(f"/sessions/{session_id}")
    assert response.status_code == 200
    return response.json()


def test_direct_is_saved_with_snapshots(tmp_path):
    app, client = _client(tmp_path, [_direct()])
    try:
        sid = _new_session(client, strategy="standard", turn=_turn(original=("A", "B")))
        response = _post(client, sid)
        assert response.status_code == 200, response.text
        payload = _state(client, sid)
        assert [m["role"] for m in payload["messages"]] == ["user", "assistant"]
        assert payload["asked"] == 0
        assert payload["messages"][-1]["content"] == "Yes — A links to B [message 1]."
        assert payload["messages"][-1]["strategy"] == "standard"
        assert payload["messages"][-1]["contract_version"] == "strategy-stability-integration-v1"
    finally:
        client.close()
        app.state.service.close()


def test_a_and_ab_inference_is_saved(tmp_path):
    app, client = _client(tmp_path, [_infer(), _infer()])
    try:
        sid = _new_session(
            client,
            strategy="a",
            evidence=_evidence(
                "A links to B.",
                "B links to C.",
                "Rule: if A links to B and B links to C then A links to C.",
            ),
            turn=_turn(),
        )
        response = _post(client, sid)
        assert response.status_code == 200, response.text
        payload = _state(client, sid)
        assert len(payload["messages"]) == 2
        assert "[inferred]" in payload["messages"][-1]["content"]
        assert payload["asked"] == 0

        sid_ab = _new_session(
            client,
            strategy="ab",
            evidence=_evidence(
                "A links to B.",
                "B links to C.",
                "Rule: if A links to B and B links to C then A links to C.",
            ),
            turn=_turn(),
        )
        response_ab = _post(client, sid_ab)
        assert response_ab.status_code == 200, response_ab.text
        assert "[inferred]" in _state(client, sid_ab)["messages"][-1]["content"]
    finally:
        client.close()
        app.state.service.close()


@pytest.mark.parametrize("strategy", ["standard", "b"])
def test_legal_insufficient_does_not_invent_derivation(tmp_path, strategy):
    app, client = _client(
        tmp_path,
        [
            {
                "items": [
                    {
                        "type": "insufficient",
                        "unknown_target": _rel("A", "C"),
                    }
                ]
            }
        ],
    )
    try:
        sid = _new_session(
            client,
            strategy=strategy,
            evidence=_evidence(
                "A links to B.",
                "B links to C.",
                "Rule: if A links to B and B links to C then A links to C.",
            ),
            turn=_turn(),
        )
        response = _post(client, sid)
        assert response.status_code == 200, response.text
        payload = _state(client, sid)
        assert payload["messages"][-1]["content"].startswith("I cannot confirm")
        assert payload["asked"] == 0
    finally:
        client.close()
        app.state.service.close()


def test_invalid_source_keeps_user_and_no_assistant(tmp_path):
    app, client = _client(tmp_path, [_direct(source=99)])
    try:
        sid = _new_session(client, strategy="standard", turn=_turn(original=("A", "B")))
        response = _post(client, sid)
        assert response.status_code == 502
        assert response.json()["code"] == "source_not_allowed"
        payload = _state(client, sid)
        assert [m["role"] for m in payload["messages"]] == ["user"]
        assert payload["asked"] == 0
    finally:
        client.close()
        app.state.service.close()


def test_language_failure_keeps_user_and_state(tmp_path):
    app, client = _client(tmp_path, [_direct()])
    try:
        sid = _new_session(client, strategy="standard", language="zh", turn=_turn(original=("A", "B")))
        response = _post(client, sid)
        assert response.status_code == 502
        assert response.json()["code"] == "receive_reply_language_mismatch"
        payload = _state(client, sid)
        assert [m["role"] for m in payload["messages"]] == ["user"]
        assert payload["asked"] == 0
    finally:
        client.close()
        app.state.service.close()


def test_malformed_candidate_and_model_exception_keep_user(tmp_path):
    app, client = _client(tmp_path, [{"items": []}, RuntimeError("boom")])
    try:
        sid = _new_session(client, strategy="standard", turn=_turn(original=("A", "B")))
        first = _post(client, sid)
        assert first.status_code == 502
        second = _post(client, sid, "Again?")
        assert second.status_code == 502
        payload = _state(client, sid)
        assert [m["role"] for m in payload["messages"]] == ["user", "user"]
        assert payload["asked"] == 0
    finally:
        client.close()
        app.state.service.close()


def test_assistant_insert_failure_rolls_back_state(tmp_path):
    app, client = _client(tmp_path, [_ask()])
    try:
        sid = _new_session(
            client,
            strategy="b",
            evidence=_evidence("A links to B."),
            turn=_turn(gaps=(("B", "C"),)),
        )

        def fail_assistant(_mapper, _connection, target):
            if target.role == "assistant":
                raise RuntimeError("assistant insert failed")

        event.listen(ExperimentMessage, "before_insert", fail_assistant)
        try:
            response = _post(client, sid)
        finally:
            event.remove(ExperimentMessage, "before_insert", fail_assistant)
        assert response.status_code == 500
        assert response.json()["code"] == "assistant_state_commit_failed"
        payload = _state(client, sid)
        assert [m["role"] for m in payload["messages"]] == ["user"]
        assert payload["asked"] == 0
    finally:
        client.close()
        app.state.service.close()


def test_state_update_failure_rolls_back_assistant(tmp_path):
    app, client = _client(tmp_path, [_ask()])
    try:
        sid = _new_session(
            client,
            strategy="b",
            evidence=_evidence("A links to B."),
            turn=_turn(gaps=(("B", "C"),)),
        )

        def fail_state(_mapper, _connection, target):
            raise RuntimeError("state update failed")

        event.listen(ExperimentSession, "before_update", fail_state)
        try:
            response = _post(client, sid)
        finally:
            event.remove(ExperimentSession, "before_update", fail_state)
        assert response.status_code == 500
        payload = _state(client, sid)
        assert [m["role"] for m in payload["messages"]] == ["user"]
        assert payload["asked"] == 0
    finally:
        client.close()
        app.state.service.close()


def test_ask_then_matching_remediation_then_inference(tmp_path):
    app, client = _client(tmp_path, [_ask(), _infer(rule_source=2, second_source=50)])
    try:
        sid = _new_session(
            client,
            strategy="ab",
            evidence=_evidence(
                "A links to B.",
                "Rule: if A links to B and B links to C then A links to C.",
            ),
            turn=_turn(gaps=(("B", "C"),)),
        )
        first = _post(client, sid, "What about A?")
        assert first.status_code == 200, first.text
        state = _state(client, sid)
        assert state["asked"] == 1
        assert state["pending_gap"] == {"kind": "missing_fact", "target": {"subject": "B", "object": "C"}, "served_target": {"subject": "A", "object": "C"}}

        bad = client.post(
            f"/sessions/{sid}/remediation",
            json={
                "gap": _rel("B", "D"),
                "teacher_message_id": 50,
                "content": "B links to D.",
            },
        )
        assert bad.status_code == 422

        good = client.post(
            f"/sessions/{sid}/remediation",
            json={
                "gap": _rel("B", "C"),
                "teacher_message_id": 50,
                "content": "B links to C.",
            },
        )
        assert good.status_code == 200, good.text
        state = _state(client, sid)
        assert state["remediated"] == 1
        assert state["pending_gap"] is None

        second = _post(client, sid, "Now what?")
        assert second.status_code == 200, second.text
        state = _state(client, sid)
        assert state["messages"][-1]["role"] == "assistant"
        assert "[inferred]" in state["messages"][-1]["content"]
    finally:
        client.close()
        app.state.service.close()


def test_second_ask_is_rejected_after_first_is_saved(tmp_path):
    app, client = _client(tmp_path, [_ask(), _ask()])
    try:
        sid = _new_session(
            client,
            strategy="b",
            evidence=_evidence("A links to B."),
            turn=_turn(gaps=(("B", "C"),)),
        )
        first = _post(client, sid)
        assert first.status_code == 200
        second = _post(client, sid, "And?")
        assert second.status_code == 502
        assert second.json()["validation_code"] == "question_limit"
        payload = _state(client, sid)
        assert [m["role"] for m in payload["messages"]] == ["user", "assistant", "user"]
        assert payload["asked"] == 1
    finally:
        client.close()
        app.state.service.close()


def test_same_session_concurrency_cannot_bypass_one_ask(tmp_path):
    release = threading.Event()
    entered = threading.Event()

    def block(_index):
        entered.set()
        assert release.wait(5)

    model = FakeCandidateModel([_ask(), _ask()])
    model.before_response = block
    app = create_app(tmp_path / "concurrency.sqlite3", model)
    service = app.state.service
    try:
        created = service.create_session(
            strategy="b",
            language="en",
            evidence=_evidence("A links to B."),
            turn_context=_turn(gaps=(("B", "C"),)),
            model_name="fake-candidate-v1",
        )
        sid = created["id"]
        results = []

        def run():
            try:
                results.append(service.send(sid, "What about A?"))
            except Exception as exc:  # pragma: no cover - assertion below
                results.append(exc)

        first = threading.Thread(target=run)
        second = threading.Thread(target=run)
        first.start()
        assert entered.wait(5)
        second.start()
        release.set()
        first.join(5)
        second.join(5)
        assert len(results) == 2
        payload = service.get_session(sid)
        assert payload["asked"] == 1
        assert [m["role"] for m in payload["messages"]].count("assistant") == 1
    finally:
        service.close()


def test_different_sessions_do_not_share_ask_state(tmp_path):
    app, client = _client(tmp_path, [_ask(), _ask()])
    try:
        first = _new_session(
            client,
            strategy="b",
            evidence=_evidence("A links to B."),
            turn=_turn(gaps=(("B", "C"),)),
        )
        second = _new_session(
            client,
            strategy="b",
            evidence=_evidence("A links to B."),
            turn=_turn(gaps=(("B", "C"),)),
        )
        assert _post(client, first).status_code == 200
        assert _post(client, second).status_code == 200
        one, two = _state(client, first), _state(client, second)
        assert one["asked"] == two["asked"] == 1
        assert [m["role"] for m in one["messages"]] == ["user", "assistant"]
        assert [m["role"] for m in two["messages"]] == ["user", "assistant"]
        assert app.state.candidate_model.calls == 2
    finally:
        client.close()
        app.state.service.close()


def test_ask_after_remediation_is_rejected(tmp_path):
    app, client = _client(tmp_path, [_ask(), _ask()])
    try:
        sid = _new_session(
            client,
            strategy="b",
            evidence=_evidence("A links to B."),
            turn=_turn(gaps=(("B", "C"),)),
        )
        first = _post(client, sid)
        assert first.status_code == 200
        state = _state(client, sid)
        assert state["pending_gap"] == {"kind": "missing_fact", "target": {"subject": "B", "object": "C"}, "served_target": {"subject": "A", "object": "C"}}
        remediated = client.post(
            f"/sessions/{sid}/remediation",
            json={
                "gap": _rel("B", "C"),
                "teacher_message_id": 50,
                "content": "B links to C.",
            },
        )
        assert remediated.status_code == 200
        second = _post(client, sid, "Can you ask again?")
        assert second.status_code == 502
        assert second.json()["validation_code"] in {"question_limit", "question_after_remediation"}
        state = _state(client, sid)
        assert state["asked"] == 1
        assert state["remediated"] == 1
        assert [m["role"] for m in state["messages"]].count("assistant") == 1
    finally:
        client.close()
        app.state.service.close()


def test_direct_plus_final_ask_saves_sources_and_one_state_increment(tmp_path):
    candidate = _direct("A", "B")
    candidate["items"].append(_ask()["items"][0])
    app, client = _client(tmp_path, [candidate])
    try:
        sid = _new_session(
            client,
            strategy="ab",
            evidence=_evidence("A links to B."),
            turn=_turn(subgoals=(("A", "B"),), gaps=(("B", "C"),)),
        )
        response = _post(client, sid)
        assert response.status_code == 200, response.text
        state = _state(client, sid)
        assistant = state["messages"][-1]["content"]
        assert assistant.endswith("?")
        assert state["asked"] == 1
        assert state["pending_gap"] == {"kind": "missing_fact", "target": {"subject": "B", "object": "C"}, "served_target": {"subject": "A", "object": "C"}}
        assert app.state.candidate_model.calls == 1
    finally:
        client.close()
        app.state.service.close()


def _pending_fact(subject="B", object_="C", served=("A", "C")):
    return {
        "kind": "missing_fact",
        "target": _rel(subject, object_),
        "served_target": _rel(*served),
    }


def test_remediation_checks_body_not_just_gap_and_preserves_state_on_reject(tmp_path):
    app, client = _client(tmp_path, [_ask()])
    try:
        sid = _new_session(
            client,
            strategy="b",
            evidence=_evidence("A links to B."),
            turn=_turn(gaps=(("B", "C"),)),
        )
        assert _post(client, sid).status_code == 200
        before = _state(client, sid)
        evidence_before = app.state.service.get_session(sid)
        assert evidence_before["pending_gap"] == _pending_fact()

        wrong_relation = client.post(
            f"/sessions/{sid}/remediation",
            json={"gap": _rel("B", "C"), "teacher_message_id": 50, "content": "X links to Y."},
        )
        assert wrong_relation.status_code == 422
        target_answer = client.post(
            f"/sessions/{sid}/remediation",
            json={"gap": _rel("B", "C"), "teacher_message_id": 51, "content": "A links to C."},
        )
        assert target_answer.status_code == 422
        middle = _state(client, sid)
        assert middle["remediated"] == 0
        assert middle["pending_gap"] == _pending_fact()

        negative = client.post(
            f"/sessions/{sid}/remediation",
            json={"gap": _rel("B", "C"), "teacher_message_id": 52, "content": "B does not link to C."},
        )
        assert negative.status_code == 200
        after = _state(client, sid)
        assert after["remediated"] == 1
        assert after["pending_gap"] is None
        assert after["messages"] == before["messages"]
    finally:
        client.close()
        app.state.service.close()


def test_missing_rule_rejects_fact_and_accepts_real_rule(tmp_path):
    app, client = _client(tmp_path, [_ask(subject="A", object_="C", kind="missing_rule")])
    try:
        sid = _new_session(
            client,
            strategy="b",
            evidence=_evidence("A links to B.", "B links to C."),
            turn=_turn(missing_rule=True, gaps=(("B", "C"),)),
        )
        assert _post(client, sid).status_code == 200
        state = _state(client, sid)
        assert state["pending_gap"]["kind"] == "missing_rule"

        fact = client.post(
            f"/sessions/{sid}/remediation",
            json={"gap": _rel("A", "C"), "teacher_message_id": 60, "content": "A links to C."},
        )
        assert fact.status_code == 422
        assert _state(client, sid)["remediated"] == 0

        rule = client.post(
            f"/sessions/{sid}/remediation",
            json={
                "gap": _rel("A", "C"),
                "teacher_message_id": 61,
                "content": "Rule: if A links to B and B links to C then A links to C.",
            },
        )
        assert rule.status_code == 200, rule.text
        assert _state(client, sid)["remediated"] == 1
    finally:
        client.close()
        app.state.service.close()


def test_legacy_ask_shape_gets_consistent_pending(tmp_path):
    legacy = {"action": "ask", "ask": {"kind": "missing_fact", "subject": "B", "object": "C"}}
    app, client = _client(tmp_path, [legacy])
    try:
        sid = _new_session(
            client,
            strategy="b",
            evidence=_evidence("A links to B."),
            turn=_turn(gaps=(("B", "C"),)),
        )
        assert _post(client, sid).status_code == 200
        state = _state(client, sid)
        assert state["pending_gap"] == _pending_fact()
        assert state["asked"] == 1
    finally:
        client.close()
        app.state.service.close()


def test_fake_receives_frozen_full_request(tmp_path):
    from dataclasses import FrozenInstanceError

    app, client = _client(tmp_path, [_direct()])
    try:
        sid = _new_session(client, strategy="standard", turn=_turn(original=("A", "B")))
        assert _post(client, sid).status_code == 200
        request = app.state.candidate_model.frozen_requests[0]
        assert request.session_id == sid
        assert request.question == "What about A?"
        assert request.strategy == "standard"
        assert request.language == "en"
        assert request.model == "fake-candidate-v1"
        assert request.contract_version == "strategy-stability-integration-v1"
        assert request.verifier_version == "strategy-stability-verifier-v5"
        assert request.behavior_version == "fake-candidate-v1"
        assert request.turn_context.goal.original.subject == "A"
        assert request.evidence[0].message_id == 1
        assert request.evidence[0].content == "A links to B."
        with __import__("pytest").raises(FrozenInstanceError):
            request.question = "mutated"
    finally:
        client.close()
        app.state.service.close()


def test_user_store_failure_reports_not_saved_and_no_model_call(tmp_path):
    app, client = _client(tmp_path, [_direct()])
    try:
        sid = _new_session(client, strategy="standard", turn=_turn(original=("A", "B")))

        def fail_user(_mapper, _connection, target):
            if target.role == "user":
                raise RuntimeError("user insert failed")

        event.listen(ExperimentMessage, "before_insert", fail_user)
        try:
            response = _post(client, sid)
        finally:
            event.remove(ExperimentMessage, "before_insert", fail_user)
        assert response.status_code == 500
        assert response.json()["code"] == "user_message_store_failed"
        assert "not saved" in response.json()["message"]
        assert app.state.candidate_model.calls == 0
        payload = _state(client, sid)
        assert payload["messages"] == []
    finally:
        client.close()
        app.state.service.close()
