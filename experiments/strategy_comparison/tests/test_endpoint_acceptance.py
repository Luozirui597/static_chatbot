"""Zero-network endpoint acceptance for the four production strategies."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

import backend.main as main
from backend.chat_service import ChatService
from backend.database import create_database_engine, create_tables, get_db
from backend.reply_language import LANGUAGE_INSTRUCTIONS
from backend.teaching_strategies import get_teaching_prompt, get_teaching_prompt_version

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = [
    ("standard", "baseline"),
    ("a", "a"),
    ("b", "b"),
    ("ab", "ab"),
]


class FakeModel:
    def __init__(self):
        self.calls: list[list[dict]] = []
        self.reply = "Understood."

    async def generate(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        return self.reply


@pytest.mark.parametrize("condition_id,switch_value", STRATEGIES)
def test_real_fastapi_path_creates_switches_sends_and_reads(
    tmp_path, monkeypatch, condition_id, switch_value
):
    engine = create_database_engine(f"sqlite:///{tmp_path / 'endpoint.db'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    model = FakeModel()
    service = ChatService(model)
    monkeypatch.setattr(main, "chat_service", service)

    def database():
        with factory() as db:
            yield db

    main.app.dependency_overrides[get_db] = database
    client = TestClient(main.app)
    try:
        response = client.post("/api/sessions")
        assert response.status_code == 201, response.text
        session_id = response.json()["id"]

        response = client.patch(
            f"/api/sessions/{session_id}/teaching-strategy",
            json={"teaching_strategy": switch_value},
        )
        assert response.status_code == 200, response.text
        assert response.json()["session"]["teaching_strategy"] == switch_value

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "Please confirm this teaching."},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        user = payload["user_message"]
        assistant = payload["assistant_message"]

        assert len(model.calls) == 1
        system = model.calls[0][0]["content"]
        assert system.startswith(get_teaching_prompt(switch_value).rstrip())
        assert system.endswith(LANGUAGE_INSTRUCTIONS["en"] + "\n")
        assert system.count(LANGUAGE_INSTRUCTIONS["en"]) == 1
        assert "Language instruction" not in system

        expected_version = (
            get_teaching_prompt_version(switch_value, "ordinary") + "+l1-en"
        )
        condition = next(
            c for c in json.loads((ROOT / "conditions.json").read_text())["conditions"]
            if c["strategy_id"] == condition_id
        )
        assert condition["prompt_version"] == expected_version
        assert user["teaching_strategy_snapshot"] == switch_value
        assert assistant["teaching_strategy_snapshot"] == switch_value
        assert user["prompt_version_snapshot"] == expected_version
        assert assistant["prompt_version_snapshot"] == expected_version

        response = client.get(f"/api/sessions/{session_id}/messages")
        assert response.status_code == 200, response.text
        rows = response.json()
        assert [row["role"] for row in rows] == ["user", "assistant"]
        assert [row["teaching_strategy_snapshot"] for row in rows] == [
            switch_value,
            switch_value,
        ]
        assert [row["prompt_version_snapshot"] for row in rows] == [
            expected_version,
            expected_version,
        ]
        assert rows[0]["content"] == "Please confirm this teaching."
    finally:
        client.close()
        main.app.dependency_overrides.pop(get_db, None)
        engine.dispose()
