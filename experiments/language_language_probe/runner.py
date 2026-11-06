"""Mock replay, request evidence, and blind scoring for the frozen plan."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from random import SystemRandom
from typing import Callable
from uuid import uuid4

import httpx

from plan import MODEL, NOTE_EN, NOTE_ZH, ROOT, SOURCE, TIMEOUT, Item, build_plan, digest, json_text, manifest

MOCK_URL = "https://language-probe.invalid/v1/chat/completions"
LANGUAGES = {"English", "Chinese", "Mixed"}


def payload_for(item: Item) -> dict:
    return {"model": MODEL, "messages": item.messages()}


def synthetic_response(item: Item) -> httpx.Response:
    language = item.case.expected_language
    if language == "Chinese":
        text = "这是合成样例。"
    else:
        text = "This is a synthetic example."
    if item.case.quote_required:
        text += f' “{item.case.quote_required}”'
    return httpx.Response(200, json={
        "choices": [{"finish_reason": "stop", "message": {"content": text}}],
    })


def _observe_request(item: Item, request: httpx.Request, row: dict, *, url: str = MOCK_URL) -> bool:
    body = json.loads(request.content)
    if not isinstance(body, dict):
        return False
    timeout = request.extensions.get("timeout", {})
    row.update(
        temperature_sent="temperature" in body,
        temperature_value=body.get("temperature"),
        max_tokens_sent="max_tokens" in body,
        max_tokens_value=body.get("max_tokens"),
        http_timeout_seconds=timeout.get("read"),
        http_timeout_phases=timeout,
        captured_messages_sha256=digest(json_text(body.get("messages"), canonical=True)),
        captured_request_sha256=digest(request.content),
    )
    return (
        request.method == "POST" and str(request.url) == url
        and set(body) == {"model", "messages"}
        and body == {"model": MODEL, "messages": item.messages()}
        and timeout == dict.fromkeys(("connect", "read", "write", "pool"), TIMEOUT)
    )


def _classify_response(response: httpx.Response, row: dict) -> None:
    row["http_status"] = response.status_code
    if not response.is_success:
        row["error"] = "http_error"
        return
    try:
        choice = response.json()["choices"][0]
        finish = choice["finish_reason"]
        reply = choice["message"]["content"]
    except (ValueError, TypeError, KeyError, IndexError):
        row["error"] = "invalid_response"
        return
    row["finish_reason"] = finish if finish in ("stop", "length", "content_filter", "tool_calls") else None
    if finish == "length":
        row["error"] = "truncated"
    elif finish != "stop" or not isinstance(reply, str):
        row["error"] = "invalid_response"
    elif not reply.strip():
        row["error"] = "empty_reply"
    else:
        row.update(status="success", reply=reply)


def run_offline(
    items: list[Item],
    respond: Callable[[Item], httpx.Response] = synthetic_response,
) -> list[dict]:
    """Send each selected item once through a local mock transport."""
    if len(items) > 168 or len({item.key for item in items}) != len(items):
        raise ValueError("Offline plan exceeds budget or repeats an item")
    rows = []
    for item in items:
        row = {
            "item_id": item.key,
            "synthetic": True,
            "attempts": 1,
            "mock_transport_entered": False,
            "status": "failure",
            "http_status": None,
            "finish_reason": None,
            "error": None,
        }

        def handle(request: httpx.Request) -> httpx.Response:
            row["mock_transport_entered"] = True
            if not _observe_request(item, request, row):
                row["error"] = "request_contract"
                return httpx.Response(400)
            return respond(item)

        try:
            with httpx.Client(transport=httpx.MockTransport(handle), timeout=TIMEOUT, trust_env=False) as client:
                response = client.post(MOCK_URL, json=payload_for(item))
            if row["error"] == "request_contract":
                row["http_status"] = response.status_code
            else:
                _classify_response(response, row)
        except httpx.TimeoutException:
            row["error"] = "timeout"
        except httpx.TransportError:
            row["error"] = "transport_error"
        except (ValueError, TypeError):
            row["error"] = "request_contract" if row["mock_transport_entered"] else "request_prepare"
        rows.append(row)
    return rows


def export_blind(items: list[Item], rows: list[dict], *, synthetic: bool = True) -> tuple[dict, dict]:
    planned = {item.key: item for item in items}
    materials, mapping = [], {}
    for row in rows:
        if row.get("synthetic") is not synthetic:
            raise ValueError("Attempt origin does not match export mode")
        if row["status"] != "success":
            continue
        item = planned[row["item_id"]]
        case = item.case
        random_id = uuid4().hex
        materials.append({
            "random_id": random_id,
            "current_user": case.current,
            "teacher_evidence": [
                {"source": f"S{index}", "content": text}
                for index, (_, text) in enumerate(case.teachers, start=1)
            ],
            "reply": row["reply"],
            "rubric": {
                "expected_main_clause_language": case.expected_language,
                "answer_standard": case.answer_standard,
                "quote_required": case.quote_required,
                "language_rule": "Score natural-language prose. An exact quotation, formula or identifier alone is not mixed prose.",
                "fidelity_rule": "2 = faithful and sufficient; 1 = partial or minor drift; 0 = wrong or missing the core content.",
            },
        })
        mapping[random_id] = {
            "item_id": item.key,
            "condition": item.condition,
            "session": case.session,
            "turn": case.current_id if case.session is not None else case.key,
            "order_slot": item.order,
            "position": item.position,
            "note_language": "en" if item.condition == "C" else "zh",
            "source_message_ids": {
                f"S{index}": mid for index, (mid, _) in enumerate(case.teachers, start=1)
            },
        }
    SystemRandom().shuffle(materials)
    return {"synthetic": synthetic, "materials": materials}, mapping


def _index_scores(mapping: dict, annotations: list[dict]) -> dict:
    scores = {}
    for score in annotations:
        random_id = score["random_id"]
        if random_id not in mapping:
            raise ValueError("Unknown blind ID")
        item_id = mapping[random_id]["item_id"]
        if item_id in scores:
            raise ValueError("Duplicate score")
        quote_score = score.get("quoted_original_preserved")
        if (score.get("language") not in LANGUAGES
                or score.get("main_clause_language") not in LANGUAGES
                or type(score.get("contains_chinese_prose")) is not bool
                or type(score.get("fidelity_score")) is not int
                or score["fidelity_score"] not in (0, 1, 2)
                or not isinstance(score.get("notes", ""), str)
                or (quote_score is not None and type(quote_score) is not bool)):
            raise ValueError("Invalid score fields")
        scores[item_id] = score
    return scores


def _condition_totals(group: list[Item], attempts: dict, scores: dict) -> dict:
    planned = Counter(item.condition for item in group)
    totals = {}
    for condition, count in sorted(planned.items()):
        ids = [item.key for item in group if item.condition == condition]
        success = sum(attempts[key]["status"] == "success" for key in ids if key in attempts)
        failures = Counter(
            attempts[key]["error"]
            for key in ids if key in attempts and attempts[key]["status"] == "failure"
        )
        annotated = [item for item in group if item.condition == condition and item.key in scores]
        totals[condition] = {
            "planned": count,
            "success": success,
            "failure": sum(failures.values()),
            "not_attempted": count - success - sum(failures.values()),
            "failures": dict(failures),
            "scored": len(annotated),
            "chinese_prose": sum(scores[item.key]["contains_chinese_prose"] for item in annotated),
            "language": dict(Counter(scores[item.key]["language"] for item in annotated)),
            "main_clause_language": dict(Counter(scores[item.key]["main_clause_language"] for item in annotated)),
            "language_compliant": sum(
                scores[item.key]["main_clause_language"] == item.case.expected_language
                and (item.case.quote_required is None
                     or scores[item.key].get("quoted_original_preserved") is True)
                for item in annotated
            ),
            "fidelity": dict(Counter(scores[item.key]["fidelity_score"] for item in annotated)),
            "quote_preserved": sum(scores[item.key].get("quoted_original_preserved") is True for item in annotated),
        }
    return totals


def _pair_totals(group: list[Item], scores: dict) -> dict:
    by_case = defaultdict(dict)
    for item in group:
        if item.key in scores:
            by_case[item.case.key][item.condition] = scores[item.key]
    pairs = {}
    for other in ("B", "C"):
        paired = [
            (labels["A"], labels[other])
            for labels in by_case.values() if "A" in labels and other in labels
        ]
        pairs[f"A{other}"] = {
            "n": len(paired),
            "main_clause_transitions": dict(Counter(
                f"{a['main_clause_language']} -> {b['main_clause_language']}" for a, b in paired
            )),
            "chinese_prose_reduced": sum(a["contains_chinese_prose"] and not b["contains_chinese_prose"] for a, b in paired),
            "chinese_prose_increased": sum(not a["contains_chinese_prose"] and b["contains_chinese_prose"] for a, b in paired),
        }
    return pairs


def summarize(items: list[Item], rows: list[dict], mapping: dict, annotations: list[dict], *, synthetic: bool = True) -> dict:
    if any(row.get("synthetic") is not synthetic for row in rows):
        raise ValueError("Attempt origin does not match scoring mode")
    attempts = {row["item_id"]: row for row in rows}
    planned = {item.key for item in items}
    if len(attempts) != len(rows) or not attempts.keys() <= planned:
        raise ValueError("Duplicate or unknown attempt")
    scores = _index_scores(mapping, annotations)
    if any(key not in attempts or attempts[key]["status"] != "success" for key in scores):
        raise ValueError("Only completed items with usable text can be scored")
    main = [item for item in items if item.case.session is not None]
    controls = defaultdict(list)
    for item in items:
        if item.case.session is None:
            controls[item.case.key].append(item)
    main_totals = _condition_totals(main, attempts, scores)
    baseline = main_totals.get("A", {})
    if baseline.get("chinese_prose", 0):
        baseline_status = "observed_in_synthetic_scores" if synthetic else "observed"
    elif baseline.get("scored", 0) and baseline["scored"] == baseline["planned"]:
        baseline_status = "not_reproduced_in_synthetic_scores" if synthetic else "not_reproduced"
    else:
        baseline_status = "insufficient_scores"
    return {
        "synthetic": synthetic,
        "main": {
            "conditions": main_totals,
            "pairs": _pair_totals(main, scores),
            "by_session": {
                str(session): _condition_totals(
                    [item for item in main if item.case.session == session], attempts, scores
                )
                for session in sorted({item.case.session for item in main})
            },
            "baseline_chinese_prose": baseline_status,
        },
        "controls": {
            key: {
                "conditions": _condition_totals(group, attempts, scores),
                "pairs": _pair_totals(group, scores),
            }
            for key, group in controls.items()
        },
    }


def _write_json(path: Path, value, *, exclusive=False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Read-only directory with frozen acceptance evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="Write a new plan file")
    freeze.add_argument("output", type=Path)
    demo = sub.add_parser("demo", help="Run the frozen plan through MockTransport")
    demo.add_argument("output_dir", type=Path)
    scoring = sub.add_parser("score", help="Summarize blind annotations")
    scoring.add_argument("run_dir", type=Path)
    scoring.add_argument("annotations", type=Path)
    scoring.add_argument("output", type=Path)
    args = parser.parse_args()
    items = build_plan(args.source)
    expected = manifest(items)
    if args.command == "freeze":
        _write_json(args.output, expected, exclusive=True)
        return
    if expected != json.loads((ROOT / "plan.json").read_text(encoding="utf-8")):
        raise SystemExit("Plan differs from frozen manifest")
    if args.command == "demo":
        out = args.output_dir
        out.mkdir(mode=0o700)
        (out / "internal").mkdir(mode=0o700)
        rows = run_offline(items)
        blind, mapping = export_blind(items, rows)
        _write_json(out / "blind.json", blind)
        _write_json(out / "internal/attempts.json", rows)
        _write_json(out / "internal/blind-map.json", mapping)
        _write_json(out / "internal/request-notes.json", {"zh": NOTE_ZH, "en": NOTE_EN})
        _write_json(out / "internal/summary.json", summarize(items, rows, mapping, []))
        print(f"synthetic: {len(rows)} attempts, {len(blind['materials'])} blind materials")
        return
    rows = json.loads((args.run_dir / "internal/attempts.json").read_text(encoding="utf-8"))
    mapping = json.loads((args.run_dir / "internal/blind-map.json").read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    blind = json.loads((args.run_dir / "blind.json").read_text(encoding="utf-8"))
    _write_json(args.output, summarize(items, rows, mapping, annotations, synthetic=blind["synthetic"]), exclusive=True)


if __name__ == "__main__":
    main()
