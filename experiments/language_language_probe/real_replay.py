"""Run one explicitly authorized provider replay of the existing frozen plan."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from dotenv import dotenv_values
import httpx

from plan import MODEL, NOTE_EN, NOTE_ZH, ROOT, TIMEOUT, build_plan, digest, manifest
from runner import _classify_response, _observe_request, _write_json, export_blind, payload_for, summarize

CONFIG = ROOT.parents[1] / ".env"
BLOCKING_HTTP = {401, 402, 403, 404, 429}


def load_api_settings(path: Path) -> tuple[str, str]:
    values = dotenv_values(path, interpolate=False)
    mode = os.environ.get("LLM_MODE", values.get("LLM_MODE") or "").strip()
    model = os.environ.get("LLM_MODEL", values.get("LLM_MODEL") or "").strip()
    reasoning = os.environ.get("LLM_REASONING_EFFORT", values.get("LLM_REASONING_EFFORT") or "").strip()
    base = os.environ.get("LLM_API_BASE_URL", values.get("LLM_API_BASE_URL") or "").strip().rstrip("/")
    key = os.environ.get("LLM_API_KEY", values.get("LLM_API_KEY") or "").strip()
    if mode != "real" or model != MODEL or reasoning:
        raise ValueError("API model or generation settings do not match the frozen production baseline")
    parts = urlsplit(base)
    if (parts.scheme != "https" or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment):
        raise ValueError("API configuration requires a clean HTTPS base URL")
    if not key or not key.isascii() or any(ch.isspace() for ch in key):
        raise ValueError("API credential is missing or malformed")
    return base + "/chat/completions", key


def _journal(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_once(items, client: httpx.Client, url: str, key: str, journal) -> tuple[list[dict], str | None]:
    if len(items) != 168 or len({item.key for item in items}) != len(items):
        raise ValueError("The authorized batch requires exactly 168 unique items")
    rows = []
    stop_reason = None
    for index, item in enumerate(items, start=1):
        row = {
            "item_id": item.key, "synthetic": False, "attempts": 0, "status": "failure",
            "send_started": False, "http_status": None, "finish_reason": None, "error": None,
        }
        started = time.monotonic()
        try:
            request = client.build_request("POST", url, json=payload_for(item))
            if not _observe_request(item, request, row, url=url):
                row["error"] = "request_contract"
                stop_reason = "request_contract"
            else:
                _journal(journal, {"event": "before_send", "time_utc": _utc(),
                                   "index": index, "item_id": item.key, **{
                                       field: row[field] for field in (
                                           "captured_messages_sha256", "captured_request_sha256",
                                           "temperature_sent", "max_tokens_sent", "http_timeout_seconds",
                                       )
                                   }})
                row.update(attempts=1, send_started=True)
                response = client.send(request, follow_redirects=False)
                _classify_response(response, row)
                if response.is_success:
                    try:
                        reported = response.json().get("model")
                    except (ValueError, AttributeError):
                        reported = None
                    if isinstance(reported, str) and re.fullmatch(r"[\w./:-]{1,96}", reported) and key not in reported:
                        row["model_reported"] = reported
                if key in row.get("reply", ""):
                    row.pop("reply")
                    row.update(status="failure", error="credential_echo")
                    stop_reason = "credential_echo"
                if response.status_code in BLOCKING_HTTP:
                    stop_reason = f"provider_http_{response.status_code}"
        except httpx.TimeoutException:
            row["error"] = "timeout"
        except httpx.TransportError:
            row["error"] = "transport_error"
        except (ValueError, TypeError):
            row["error"] = "invalid_response" if row["send_started"] else "request_prepare"
            if not row["send_started"]:
                stop_reason = "request_prepare"
        row["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        rows.append(row)
        _journal(journal, {"event": "result", "index": index, **row})
        if index == 1 or index % 12 == 0 or stop_reason:
            print(f"{index}/168 settled; successes={sum(r['status'] == 'success' for r in rows)}"
                  + (f"; stopped={stop_reason}" if stop_reason else ""), flush=True)
        if stop_reason:
            break
    return rows, stop_reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--confirm-real-api", action="store_true", required=True)
    args = parser.parse_args()

    items = build_plan()
    raw_plan = (ROOT / "plan.json").read_bytes()
    if manifest(items) != json.loads(raw_plan):
        raise SystemExit("Plan differs from frozen manifest")
    try:
        url, key = load_api_settings(CONFIG)
    except (ValueError, OSError) as exc:
        raise SystemExit("API configuration preflight failed") from None

    os.umask(0o077)
    out = args.output_dir
    out.mkdir(mode=0o700)
    internal = out / "internal"
    internal.mkdir(mode=0o700)
    meta = {
        "synthetic": False, "started_utc": _utc(), "planned": len(items),
        "model_requested": MODEL, "http_timeout_seconds": TIMEOUT,
        "provider_host": urlsplit(url).hostname, "plan_sha256": digest(raw_plan),
    }
    _write_json(internal / "start.json", meta, exclusive=True)
    with (internal / "attempt-journal.jsonl").open("x", encoding="utf-8") as journal:
        # httpx's default HTTP transport has zero retries. Redirect following is disabled.
        with httpx.Client(timeout=TIMEOUT, follow_redirects=False,
                          headers={"Authorization": f"Bearer {key}"}) as client:
            rows, stop = run_once(items, client, url, key, journal)
    blind, mapping = export_blind(items, rows, synthetic=False)
    _write_json(out / "blind.json", blind, exclusive=True)
    _write_json(internal / "attempts.json", rows, exclusive=True)
    _write_json(internal / "blind-map.json", mapping, exclusive=True)
    _write_json(internal / "request-notes.json", {"zh": NOTE_ZH, "en": NOTE_EN}, exclusive=True)
    _write_json(internal / "summary.json", summarize(items, rows, mapping, [], synthetic=False), exclusive=True)
    meta.update(finished_utc=_utc(), send_started=sum(row["send_started"] for row in rows),
                settled=len(rows), outcomes=dict(Counter(row["status"] for row in rows)),
                failures=dict(Counter(row["error"] for row in rows if row["error"])),
                stop_reason=stop)
    _write_json(internal / "run.json", meta, exclusive=True)
    print(f"Real replay settled {len(rows)}/{len(items)}; usable blind materials={len(blind['materials'])}.", flush=True)


if __name__ == "__main__":
    main()
