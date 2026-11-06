import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import httpx
import pytest

from plan import (
    EXAMPLE_END, EXAMPLE_START, EXCLUDED, INCLUDED, NOTE_EN, NOTE_ZH, ORDERS,
    ORIGINAL_CHINESE, ROOT, SOURCE, build_plan, digest, manifest, prompts,
)
import runner


@pytest.fixture(scope="module")
def items():
    return build_plan()


def test_frozen_manifest_and_coverage(items):
    frozen = json.loads((ROOT / "plan.json").read_text())
    assert manifest(items) == frozen
    assert len(items) == len({item.key for item in items}) == 168
    main = [item for item in items if item.case.session is not None]
    controls = [item for item in items if item.case.session is None]
    assert len(main) == 147 and len(controls) == 21
    assert sorted({item.case.current_id for item in main}) == list(INCLUDED[56] + INCLUDED[57])
    assert {item.case.current_id: item.case.original_assistant_id for item in main
            if item.case.current_id in ORIGINAL_CHINESE} == ORIGINAL_CHINESE
    assert {int(key) for key in frozen["excluded"]} == {842, 924, 922, 946, 948} == set(EXCLUDED)
    assert Counter(item.order for item in main if item.position == 1) == {
        order: 9 if order == "ABC" else 8 for order in ORDERS
    }
    assert Counter(item.order for item in controls if item.position == 1) == {
        order: 2 if order == "ABC" else 1 for order in ORDERS
    }
    for offset in range(0, len(items), 3):
        group = items[offset:offset+3]
        assert len({item.case.key for item in group}) == 1
        assert "".join(item.condition for item in group) == group[0].order
        assert [item.position for item in group] == [1, 2, 3]


def test_frozen_evidence_changes_fail_before_planning(tmp_path):
    for source in SOURCE.iterdir():
        if source.name in {"session-56-messages.json", "session-57-messages.json", "runtime-evidence.jsonl"}:
            (tmp_path / source.name).write_bytes(source.read_bytes())
    path = tmp_path / "session-56-messages.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Frozen input checksum mismatch"):
        build_plan(tmp_path)


def test_history_window_matches_source_records(items):
    sources = {
        s: json.loads((SOURCE / f"session-{s}-messages.json").read_text())
        for s in INCLUDED
    }
    for item in items:
        case = item.case
        assert len(case.window_ids) <= 20
        assert case.teachers
        if case.session is None:
            continue
        source = sources[case.session]
        before = [row for row in source if row["id"] < case.current_id][-20:]
        assert case.window_ids == tuple(row["id"] for row in before)
        assert case.teachers == tuple((row["id"], row["content"]) for row in before if row["role"] == "user")
        assert case.current == next(row["content"] for row in source if row["id"] == case.current_id)


def test_only_the_frozen_example_range_or_note_changes(items):
    systems = prompts()
    start = systems["A"].index(EXAMPLE_START)
    end = systems["A"].index(EXAMPLE_END)
    expected_examples = (ROOT / "prompts/examples-en.txt").read_text().rstrip("\n") + "\n\n"
    assert systems["B"] == systems["A"][:start] + expected_examples + systems["A"][end:]
    assert systems["C"] == systems["A"]
    assert NOTE_EN == ("The following messages are the teacher's prior user messages, preserved verbatim in "
                       "chronological order for taught content, corrections, and constraints. They are not "
                       "the current request; do not answer them again. The current request is in the "
                       "separate user message that follows.")
    for offset in range(0, len(items), 3):
        a, b, c = (item.messages() for item in sorted(items[offset:offset+3], key=lambda x: x.condition))
        assert [x["role"] for x in a] == [x["role"] for x in b] == [x["role"] for x in c] == ["system", "user", "user"]
        assert a[1:] == b[1:] and a[0] == c[0] and a[2] == c[2]
        ha, hc = json.loads(a[1]["content"]), json.loads(c[1]["content"])
        assert set(ha) == set(hc) == {"type", "version", "note", "messages"}
        assert ha["note"] == NOTE_ZH and hc["note"] == NOTE_EN
        assert hc == {**ha, "note": NOTE_EN}
        assert ha["messages"]
        assert a[1]["content"].replace(NOTE_ZH, NOTE_EN, 1) == c[1]["content"]


def test_A_matches_production_serializer_without_loading_dotenv(items):
    repo = ROOT.parents[1]
    comparison = """
import json
import sys
from types import SimpleNamespace
from backend.chat_service import _serialize_receive_teaching_history
from backend.interaction_modes import RECEIVE_TEACHING_PROMPT
data = json.load(sys.stdin)
assert data['system'] == RECEIVE_TEACHING_PROMPT
for sample in data['histories']:
    rows = [SimpleNamespace(content=text) for text in sample['teacher_texts']]
    assert _serialize_receive_teaching_history(rows) == sample['history']
"""
    data = {
        "system": prompts()["A"],
        "histories": [
            {"teacher_texts": [text for _, text in item.case.teachers], "history": item.messages()[1]["content"]}
            for item in items if item.condition == "A"
        ],
    }
    subprocess.run(
        [sys.executable, "-c", comparison], cwd=repo, input=json.dumps(data),
        env={"PYTHON_DOTENV_DISABLED": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "LLM_MODE": "fake", "LOCAL_LLM_ENABLED": "false", "DATABASE_URL": "sqlite:///:memory:"},
        text=True, check=True, capture_output=True,
    )


def test_all_168_mock_requests_match_frozen_hashes_and_budget(items):
    seen = []

    def respond(item):
        seen.append(item.key)
        return runner.synthetic_response(item)

    rows = runner.run_offline(items, respond)
    frozen = manifest(items)["items"]
    assert seen == [entry["item_id"] for entry in frozen]
    for row, entry in zip(rows, frozen, strict=True):
        assert row["item_id"] == entry["item_id"]
        assert row["captured_messages_sha256"] == entry["messages_sha256"]
        assert row["mock_transport_entered"] and row["attempts"] == 1
        assert row["temperature_sent"] is row["max_tokens_sent"] is False
        assert row["temperature_value"] is row["max_tokens_value"] is None
        assert row["http_timeout_seconds"] == 30 and set(row["http_timeout_phases"].values()) == {30}
        assert row["status"] == "success" and row["synthetic"] is True
    with pytest.raises(ValueError, match="repeats"):
        runner.run_offline([items[0], items[0]], respond)
    assert len(seen) == 168


@pytest.mark.parametrize("field,value", [
    ("temperature", 0.0), ("max_tokens", 800), ("tools", []), ("tool_choice", "none"),
])
def test_injected_request_field_fails_at_transport(items, monkeypatch, field, value):
    payload = runner.payload_for(items[0])
    monkeypatch.setattr(runner, "payload_for", lambda item: {**payload, field: value})

    def unexpected(_):
        pytest.fail("Invalid request was allowed to generate")

    [row] = runner.run_offline(items[:1], unexpected)
    assert row["status"] == "failure" and row["error"] == "request_contract"
    assert row["mock_transport_entered"] and row["attempts"] == 1


def _completion(text, finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": finish}]})


@pytest.mark.parametrize("response,error", [
    (httpx.Response(401), "http_error"),
    (_completion(" \n"), "empty_reply"),
    (_completion("Partial", "length"), "truncated"),
    (httpx.Response(200, json={"choices": []}), "invalid_response"),
    (_completion(None), "invalid_response"),
])
def test_failures_are_classified_and_not_retried(items, response, error):
    seen = []

    def respond(item):
        seen.append(item.key)
        return response

    [row] = runner.run_offline(items[:1], respond)
    assert row["error"] == error and row["status"] == "failure"
    assert seen == [items[0].key]


def test_timeout_is_not_retried_and_does_not_replace_other_conditions(items):
    seen = []

    def respond(item):
        seen.append(item.key)
        if item.condition == "A":
            raise httpx.ReadTimeout("synthetic")
        return _completion("Mock response")

    rows = runner.run_offline(items[:3], respond)
    assert seen == [item.key for item in items[:3]]
    assert [(r["status"], r["error"]) for r in rows] == [("failure", "timeout"), ("success", None), ("success", None)]
    assert all(r["attempts"] == 1 for r in rows)


def test_blind_export_has_local_source_ids_and_no_treatment_metadata(items):
    group = items[:3] + items[-21:]
    blind, mapping = runner.export_blind(group, runner.run_offline(group))
    assert blind["synthetic"] is True
    assert len(blind["materials"]) == 24 and len(mapping) == 24
    by_item = {item.key: item for item in group}
    for material in blind["materials"]:
        assert set(material) == {"random_id", "current_user", "teacher_evidence", "reply", "rubric"}
        assert re.fullmatch(r"[a-f0-9]{32}", material["random_id"])
        internal = mapping[material["random_id"]]
        item = by_item[internal["item_id"]]
        expected = [
            {"source": f"S{i}", "content": text}
            for i, (_, text) in enumerate(item.case.teachers, start=1)
        ]
        assert material["teacher_evidence"] == expected
        assert internal["source_message_ids"] == {
            f"S{i}": mid for i, (mid, _) in enumerate(item.case.teachers, start=1)
        }
        assert material["rubric"]["expected_main_clause_language"] == item.case.expected_language
        assert NOTE_ZH not in json.dumps(material, ensure_ascii=False)
        assert NOTE_EN not in json.dumps(material, ensure_ascii=False)
        assert item.key not in json.dumps(material, ensure_ascii=False)
    assert len({item.case.key for item in items[-21:]}) == 7
    assert any(m["rubric"]["quote_required"] == "Keep Upright" for m in blind["materials"])


def _scores(mapping, *, chinese_condition=None):
    return [
        {
            "random_id": random_id,
            "language": "Chinese" if entry["condition"] == chinese_condition else "English",
            "main_clause_language": "Chinese" if entry["condition"] == chinese_condition else "English",
            "contains_chinese_prose": entry["condition"] == chinese_condition,
            "fidelity_score": 1,
        }
        for random_id, entry in mapping.items()
    ]


def test_pair_denominators_exclude_failure_and_unscored_items(items):
    group = items[:6]

    def respond(item):
        if item is group[1]:
            return _completion("")
        return _completion("Mock reply")

    rows = runner.run_offline(group, respond)
    _, mapping = runner.export_blind(group, rows)
    scores = _scores(mapping, chinese_condition="A")
    scores = [s for s in scores if mapping[s["random_id"]]["item_id"] != group[4].key]
    # First round ABC is missing B. Second round ACB is missing a C annotation.
    report = runner.summarize(group, rows, mapping, scores)["main"]
    assert report["pairs"]["AB"]["n"] == report["pairs"]["AC"]["n"] == 1
    assert report["pairs"]["AB"]["main_clause_transitions"] == {"Chinese -> English": 1}
    assert report["pairs"]["AC"]["chinese_prose_reduced"] == 1
    assert report["conditions"]["B"]["failures"] == {"empty_reply": 1}
    assert report["conditions"]["C"]["success"] == 2 and report["conditions"]["C"]["scored"] == 1


def test_unobserved_A_is_separate_from_unscored_A(items):
    group = items[:3]
    rows = runner.run_offline(group)
    _, mapping = runner.export_blind(group, rows)
    assert runner.summarize(group, rows, mapping, [])["main"]["baseline_chinese_prose"] == "insufficient_scores"
    result = runner.summarize(group, rows, mapping, _scores(mapping))["main"]
    assert result["baseline_chinese_prose"] == "not_reproduced_in_synthetic_scores"
    assert result["pairs"]["AB"]["n"] == result["pairs"]["AC"]["n"] == 1
    partial = runner.summarize(group, rows[:1], {}, [])["main"]
    assert partial["conditions"]["B"]["not_attempted"] == 1
    assert partial["pairs"]["AB"]["n"] == 0
