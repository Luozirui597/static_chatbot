"""Production strategy API, provenance, locking and additive storage contracts."""
import asyncio
import copy

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

import backend.main as main
from backend.chat_service import ChatService
from backend.database import create_database_engine, create_tables, get_db, run_migrations
from backend.models import ChatSession, Message, ModeSwitchEvent
from backend.session_id_migration import inspect_session_schema
from backend.teaching_strategies import get_teaching_prompt, get_teaching_prompt_version
from backend.reply_language import LANGUAGE_INSTRUCTIONS


class Model:
    def __init__(self):
        self.calls = []
        self.reply = "Understood."

    async def generate(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        return self.reply


@pytest.fixture
def setup(tmp_path, monkeypatch):
    engine = create_database_engine(f"sqlite:///{tmp_path / 'strategies.db'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    model = Model()
    service = ChatService(model)
    monkeypatch.setattr(main, "chat_service", service)

    def database():
        with factory() as db:
            yield db

    main.app.dependency_overrides[get_db] = database
    try:
        with TestClient(main.app) as client:
            yield client, factory, model, service, engine
    finally:
        main.app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def create(client):
    response = client.post("/api/sessions")
    assert response.status_code == 201, response.text
    return response.json()["id"]


@pytest.mark.parametrize("strategy", ["baseline", "a", "b", "ab"])
def test_strategy_is_persisted_and_used_with_language_constraint(setup, strategy):
    client, factory, model, _, _ = setup
    sid = create(client)
    response = client.patch(f"/api/sessions/{sid}/teaching-strategy", json={"teaching_strategy": strategy})
    assert response.status_code == 200, response.text
    assert response.json()["session"]["teaching_strategy"] == strategy
    assert not model.calls
    assert client.get(f"/api/sessions/{sid}").json()["teaching_strategy"] == strategy
    assert client.get("/api/sessions").json()[0]["teaching_strategy"] == strategy
    response = client.post(f"/api/sessions/{sid}/messages", json={"message": "Please confirm this teaching."})
    assert response.status_code == 200, response.text
    system = model.calls[0][0]["content"]
    assert system.startswith(get_teaching_prompt(strategy).rstrip())
    assert system.endswith(LANGUAGE_INSTRUCTIONS["en"] + "\n")
    assert system.count(LANGUAGE_INSTRUCTIONS["en"]) == 1
    for role in ["user", "assistant"]:
        row = response.json()[role + "_message"]
        assert row["teaching_strategy_snapshot"] == strategy
        assert row["prompt_version_snapshot"] == get_teaching_prompt_version(strategy) + "+l1-en"
    with factory() as db:
        assert len(db.scalars(select(ModeSwitchEvent)).all()) == 0
        assert db.scalar(text("SELECT COUNT(*) FROM message_teaching_strategies")) == 2


def test_switch_only_changes_future_messages_and_corrective_retains_choice(setup):
    client, _, model, _, _ = setup
    sid = create(client)
    def switch(value):
        return client.patch(f"/api/sessions/{sid}/teaching-strategy", json={"teaching_strategy": value})
    def send():
        response = client.post(f"/api/sessions/{sid}/messages", json={"message": "Please confirm this teaching."})
        assert response.status_code == 200, response.text
        return response.json()
    assert switch("a").status_code == 200
    old = send()
    assert switch("b").status_code == 200
    new = send()
    assert old["assistant_message"]["teaching_strategy_snapshot"] == "a"
    assert new["assistant_message"]["teaching_strategy_snapshot"] == "b"
    assert client.patch(f"/api/sessions/{sid}/interaction-mode", json={"interaction_mode": "corrective"}).status_code == 200
    assert switch("ab").status_code == 409
    corrective = send()
    assert corrective["assistant_message"]["teaching_strategy_snapshot"] is None
    assert corrective["assistant_message"]["prompt_version_snapshot"] == "corrective-v1"
    assert client.get(f"/api/sessions/{sid}").json()["teaching_strategy"] == "b"
    assert client.patch(f"/api/sessions/{sid}/interaction-mode", json={"interaction_mode": "receive_teaching"}).status_code == 200
    assert send()["assistant_message"]["teaching_strategy_snapshot"] == "b"
    assert client.get(f"/api/sessions/{create(client)}").json()["teaching_strategy"] == "baseline"


@pytest.mark.parametrize("value", ["A", "a+b", "a\n", "", None, 1, {}, []])
def test_invalid_strategy_is_rejected_without_model_or_setting_change(setup, value):
    client, _, model, _, _ = setup
    sid = create(client)
    response = client.patch(f"/api/sessions/{sid}/teaching-strategy", json={"teaching_strategy": value})
    assert response.status_code == 422
    assert client.get(f"/api/sessions/{sid}").json()["teaching_strategy"] == "baseline"
    assert not model.calls


@pytest.mark.parametrize("strategy", ["a", "b", "ab"])
def test_wrong_language_keeps_actual_strategy_only_on_saved_user(setup, strategy):
    client, factory, model, _, _ = setup
    sid = create(client)
    client.patch(f"/api/sessions/{sid}/teaching-strategy", json={"teaching_strategy": strategy})
    model.reply = "这是错误的中文回复。"
    response = client.post(f"/api/sessions/{sid}/messages", json={"message": "Please confirm this teaching."})
    assert response.status_code == 502
    with factory() as db:
        rows = db.scalars(select(Message).where(Message.session_id == sid)).all()
        assert len(rows) == 1 and rows[0].role == "user"
        assert rows[0].teaching_strategy_snapshot == strategy
        assert rows[0].prompt_version_snapshot.endswith("+l1-en")


def test_additive_storage_accepts_old_database_and_preserves_unknown_snapshots(tmp_path):
    path = tmp_path / "old.db"
    url = f"sqlite:///{path}"
    engine = create_database_engine(url)
    create_tables(engine)
    run_migrations(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE message_teaching_strategies"))
        conn.execute(text("DROP TABLE session_teaching_strategies"))
        conn.execute(text("DELETE FROM schema_migrations WHERE version='teaching_strategies_v1'"))
        conn.execute(text("INSERT INTO chat_sessions(id,title,created_at,updated_at) VALUES (1,'old',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
        conn.execute(text("INSERT INTO messages(id,session_id,role,content,created_at) VALUES (1,1,'user','Original',CURRENT_TIMESTAMP)"))
    assert inspect_session_schema(url) == "current"
    run_migrations(engine)
    run_migrations(engine)
    with sessionmaker(bind=engine)() as db:
        assert db.get(ChatSession, 1).teaching_strategy == "baseline"
        assert db.get(Message, 1).teaching_strategy_snapshot is None
        assert db.get(Message, 1).content == "Original"
    assert inspect_session_schema(url) == "current"
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE session_teaching_strategies"))
        conn.execute(text("CREATE TABLE session_teaching_strategies(session_id INTEGER PRIMARY KEY,strategy TEXT)"))
    assert inspect_session_schema(url) == "unknown"
    engine.dispose()


def test_strategy_switch_waits_for_inflight_send_without_tearing_snapshots(setup):
    _, factory, _, service, _ = setup

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        class BlockingModel(Model):
            async def generate(self, messages, **kwargs):
                started.set()
                await release.wait()
                return "Understood."
        current = ChatService(BlockingModel(), lock_registry=service._lock_registry)
        with factory() as first, factory() as second:
            session = current.create_session("default", first)
            sending = asyncio.create_task(current.handle_session_message(session.id, "Please confirm.", first))
            await started.wait()
            switching = asyncio.create_task(current.switch_teaching_strategy(session.id, "ab", second))
            await asyncio.sleep(0)
            assert not switching.done()
            release.set()
            user, assistant = await sending
            assert user.teaching_strategy_snapshot == assistant.teaching_strategy_snapshot == "baseline"
            await switching
        with factory() as third:
            assert third.get(ChatSession, session.id).teaching_strategy == "ab"
    asyncio.run(scenario())
