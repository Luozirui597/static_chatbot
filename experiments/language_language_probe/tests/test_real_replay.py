import json

import httpx

from plan import TIMEOUT, build_plan, manifest
from real_replay import load_api_settings, run_once

URL = "https://language-probe.invalid/v1/chat/completions"
KEY = "offline-synthetic-key"


def test_provider_replay_validates_all_prepared_requests_and_never_retries(tmp_path):
    items = build_plan()
    received = []

    def mock(request):
        body = json.loads(request.content)
        assert str(request.url) == URL and request.method == "POST"
        assert set(body) == {"model", "messages"}
        assert set(request.extensions["timeout"].values()) == {30}
        received.append(body)
        if len(received) == 2:
            raise httpx.ReadTimeout("synthetic timeout")
        return httpx.Response(200, json={
            "model": "deepseek-v4-flash",
            "choices": [{"finish_reason": "stop", "message": {"content": "Synthetic test only."}}],
        })

    journal_path = tmp_path / "journal.jsonl"
    with httpx.Client(transport=httpx.MockTransport(mock), timeout=TIMEOUT) as client:
        with journal_path.open("w") as journal:
            rows, stop = run_once(items, client, URL, KEY, journal)
    assert stop is None and len(received) == len(rows) == 168
    assert all(row["attempts"] == 1 and row["synthetic"] is False for row in rows)
    assert rows[1]["error"] == "timeout"
    assert [r["captured_messages_sha256"] for r in rows] == [
        row["messages_sha256"] for row in manifest(items)["items"]
    ]
    events = [json.loads(line) for line in journal_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["before_send", "result"] * 168
    assert KEY not in journal_path.read_text()


def test_provider_credential_failure_stops_with_one_attempt(tmp_path):
    received = []
    def mock(request):
        received.append(request)
        return httpx.Response(401, text=KEY)

    with httpx.Client(transport=httpx.MockTransport(mock), timeout=TIMEOUT) as client:
        with (tmp_path / "journal").open("w+") as journal:
            rows, stop = run_once(build_plan(), client, URL, KEY, journal)
            journal.seek(0)
            assert KEY not in journal.read()
    assert len(received) == len(rows) == 1 and stop == "provider_http_401"
    assert rows[0]["error"] == "http_error"


def test_config_preflight_with_synthetic_dotenv(tmp_path, monkeypatch):
    config = tmp_path / ".env"
    for var in ("LLM_MODE", "LLM_MODEL", "LLM_API_BASE_URL", "LLM_API_KEY", "LLM_REASONING_EFFORT"):
        monkeypatch.delenv(var, raising=False)
    config.write_text(f"LLM_MODE=real\nLLM_MODEL=deepseek-v4-flash\n"
                      f"LLM_API_BASE_URL=https://language-probe.invalid/v1\nLLM_API_KEY={KEY}\n")
    assert load_api_settings(config) == (URL, KEY)
