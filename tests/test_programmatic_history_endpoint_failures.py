"""Failure and zero-retrieval endpoint cases for programmatic history."""
from __future__ import annotations
import copy
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from backend.chat_service import ChatService
from backend.database import create_database_engine, create_tables, get_db
from backend.llm_profiles import LLMProfile, LLMProfileRegistry
from backend.main import app
from backend.models import ChatSession, Message
from backend.programmatic_history.service import ProgrammaticHistoryService, ProgrammaticSettings
from backend.programmatic_history.types import ToolResult


class Spy:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
    async def generate(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        if self.fail:
            from backend.llm_client import LLMError
            raise LLMError(detail="forced model failure", status_code=502)
        return "answer"


class ResultService(ProgrammaticHistoryService):
    def __init__(self, settings, status):
        super().__init__(settings)
        self.status = status
        self.retrievals = 0
    async def retrieve_async(self, db, *, session_id, current_message_id, plan):
        self.retrievals += 1
        return ToolResult(
            tool_call_id="programmatic_call_1", name="search_teaching_history",
            content={"status": self.status, "error_code": "forced", "results": []},
            is_error=True,
        )


class CountService(ProgrammaticHistoryService):
    def __init__(self, settings):
        super().__init__(settings)
        self.retrievals = 0
    async def retrieve_async(self, db, *, session_id, current_message_id, plan):
        self.retrievals += 1
        return await super().retrieve_async(
            db, session_id=session_id, current_message_id=current_message_id, plan=plan
        )


def run_case(tmp_path, *, status=None, enabled=True, profile_id="default",
             profile_model="model-x", settings_profile="default",
             settings_model="model-x", mode="receive_teaching",
             message="Does B link to C?", fail_model=False, history=None):
    import backend.main as main_module
    engine = create_database_engine(f"sqlite:///{tmp_path/'x.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    spy = Spy(fail=fail_model)
    profile = LLMProfile(id=profile_id, label="API", kind="api",
                         model=profile_model, client=spy, is_default=True)
    settings = ProgrammaticSettings(enabled, settings_profile, settings_model)
    service = ResultService(settings, status) if status else CountService(settings)
    previous = main_module.chat_service
    main_module.chat_service = ChatService(
        profiles=LLMProfileRegistry([profile]), programmatic_history=service
    )
    def override_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = override_db
    db = factory()
    session = ChatSession(title="case", llm_profile_id=profile_id,
                          llm_model_snapshot=profile_model,
                          interaction_mode=mode)
    db.add(session); db.flush()
    if history is None:
        history = ["I teach you this fact: B links to C."] + [
            f"I teach you filler fact {i}." for i in range(20)
        ]
    for content in history:
        db.add(Message(session_id=session.id, role="user", content=content,
                       interaction_mode_snapshot=mode,
                       prompt_version_snapshot="receive-teaching-v7+history-context-b1",
                       llm_profile_id_snapshot=profile_id,
                       llm_profile_kind_snapshot="api",
                       llm_model_snapshot=profile_model))
    db.commit(); sid = session.id; db.close()
    try:
        with TestClient(app) as client:
            response = client.post(f"/api/sessions/{sid}/messages", json={"message": message})
        return response, spy, service, factory, sid
    finally:
        app.dependency_overrides.pop(get_db, None)
        main_module.chat_service = previous
        engine.dispose()


def _assert_saved_user_only(factory, sid, content, version):
    db = factory()
    rows = db.query(Message).filter(Message.session_id == sid).all()
    current = [row for row in rows if row.role == "user" and row.content == content]
    assert len(current) == 1
    assert current[0].prompt_version_snapshot == version
    assert not [row for row in rows if row.role == "assistant"]
    db.close()


def test_retrieval_timeout_returns_504_without_model(tmp_path, caplog):
    response, spy, service, factory, sid = run_case(tmp_path, status="timeout", message="Does B link to C?")
    assert response.status_code == 504
    assert spy.calls == [] and service.retrievals == 1
    _assert_saved_user_only(factory, sid, "Does B link to C?", "receive-teaching-v7+history-context-b1+l1-en")
    assert "stage=retrieval" in caplog.text and "trigger_reason=hidden_retrieval" in caplog.text


def test_visible_complete_is_zero_retrieval(tmp_path, caplog):
    history = [
        "I teach you this fact: X links to Y.",
        "I teach you this fact: Y links to Z.",
        "I teach you this fact: B links to C.",
        *[f"I teach you filler fact {i}." for i in range(19)],
    ]
    response, spy, service, _, _ = run_case(tmp_path, history=history)
    assert response.status_code == 200
    assert len(spy.calls) == 1 and service.retrievals == 0
    base = "receive-teaching-v7+history-context-b1+l1-en"
    assert response.json()["user_message"]["prompt_version_snapshot"] == base
    assert response.json()["assistant_message"]["prompt_version_snapshot"] == base
    assert "trigger_reason=visible_complete" in caplog.text


def test_feature_disabled_is_zero_retrieval(tmp_path, caplog):
    response, spy, service, _, _ = run_case(tmp_path, enabled=False)
    assert "trigger_reason=feature_disabled" in caplog.text
    assert response.status_code == 200
    assert len(spy.calls) == 1 and service.retrievals == 0


def test_profile_mismatch_is_zero_retrieval(tmp_path, caplog):
    response, spy, service, _, _ = run_case(tmp_path, settings_profile="other")
    assert "trigger_reason=profile_id_mismatch" in caplog.text
    assert response.status_code == 200
    assert len(spy.calls) == 1 and service.retrievals == 0


def test_model_mismatch_is_zero_retrieval(tmp_path, caplog):
    response, spy, service, _, _ = run_case(tmp_path, settings_model="other")
    assert "trigger_reason=model_mismatch" in caplog.text
    assert response.status_code == 200
    assert len(spy.calls) == 1 and service.retrievals == 0


def test_corrective_is_zero_retrieval(tmp_path, caplog):
    response, spy, service, _, _ = run_case(tmp_path, mode="corrective")
    assert "trigger_reason=mode_not_receive" in caplog.text
    assert response.status_code == 200
    assert len(spy.calls) == 1 and service.retrievals == 0
    assert response.json()["user_message"]["prompt_version_snapshot"] == "corrective-v1"


class RaiseRetrieval(ProgrammaticHistoryService):
    async def retrieve_async(self, db, *, session_id, current_message_id, plan):
        from backend.programmatic_history.service import ProgrammaticHistoryError
        raise ProgrammaticHistoryError("forced")


class RaiseDecision(ProgrammaticHistoryService):
    def prepare(self, db, *, session_id, current_message_id, content):
        from backend.programmatic_history.service import ProgrammaticHistoryError
        raise ProgrammaticHistoryError("forced")


def _run_service(tmp_path, service):
    import backend.main as main_module
    from backend.database import create_database_engine, create_tables, get_db
    from backend.llm_profiles import LLMProfile, LLMProfileRegistry
    from backend.models import ChatSession, Message
    engine = create_database_engine(f"sqlite:///{tmp_path/'y.sqlite3'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    spy = Spy()
    profile = LLMProfile(id="default", label="API", kind="api", model="model-x",
                         client=spy, is_default=True)
    previous = main_module.chat_service
    main_module.chat_service = ChatService(
        profiles=LLMProfileRegistry([profile]), programmatic_history=service
    )
    def override_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = override_db
    db = factory()
    session = ChatSession(title="failure", llm_profile_id="default",
                          llm_model_snapshot="model-x")
    db.add(session); db.flush()
    for content in ["I teach you this fact: B links to C.", *[
        f"I teach you filler fact {i}." for i in range(20)
    ]]:
        db.add(Message(session_id=session.id, role="user", content=content,
                       interaction_mode_snapshot="receive_teaching",
                       prompt_version_snapshot="receive-teaching-v7+history-context-b1",
                       llm_profile_id_snapshot="default",
                       llm_profile_kind_snapshot="api",
                       llm_model_snapshot="model-x"))
    db.commit(); sid = session.id; db.close()
    try:
        with TestClient(app) as client:
            response = client.post(f"/api/sessions/{sid}/messages",
                                   json={"message": "Does B link to C?"})
        return response, spy, factory, sid
    finally:
        app.dependency_overrides.pop(get_db, None)
        main_module.chat_service = previous
        engine.dispose()


def test_retrieval_exception_returns_502(tmp_path):
    service = RaiseRetrieval(ProgrammaticSettings(True, "default", "model-x"))
    response, spy, factory, sid = _run_service(tmp_path, service)
    assert response.status_code == 502 and spy.calls == []
    _assert_saved_user_only(factory, sid, "Does B link to C?", "receive-teaching-v7+history-context-b1+l1-en")


def test_decision_exception_returns_500(tmp_path):
    service = RaiseDecision(ProgrammaticSettings(True, "default", "model-x"))
    response, spy, factory, sid = _run_service(tmp_path, service)
    assert response.status_code == 500 and spy.calls == []
    _assert_saved_user_only(factory, sid, "Does B link to C?", "receive-teaching-v7+history-context-b1+l1-en")


def test_non_target_pattern_is_zero_retrieval(tmp_path):
    response, spy, service, _, _ = run_case(tmp_path, message="What is the weather?")
    assert response.status_code == 200
    assert len(spy.calls) == 1 and service.retrievals == 0



def test_service_log_errors_do_not_change_success(tmp_path, monkeypatch):
    from backend.programmatic_history import service as service_module
    def boom(*args, **kwargs):
        raise RuntimeError("logging failed")
    monkeypatch.setattr(service_module.logger, "warning", boom)
    monkeypatch.setattr(service_module.logger, "info", boom)
    response, spy, _, _, _ = run_case(tmp_path)
    assert response.status_code == 200 and len(spy.calls) == 1


def test_service_log_errors_do_not_change_502(tmp_path, monkeypatch):
    from backend.programmatic_history import service as service_module
    def boom(*args, **kwargs):
        raise RuntimeError("logging failed")
    monkeypatch.setattr(service_module.logger, "warning", boom)
    monkeypatch.setattr(service_module.logger, "info", boom)
    response, spy, _, _, _ = run_case(tmp_path, status="error")
    assert response.status_code == 502 and spy.calls == []


P_VERSION = "receive-teaching-v7+programmatic-history-v1+l1-en"


def test_snapshot_flush_failure_and_rollback_success(tmp_path, caplog):
    from sqlalchemy import event
    from sqlalchemy.orm import Session
    def fail_flush(session, flush_context, instances):
        for obj in session.dirty:
            if isinstance(obj, Message) and obj.prompt_version_snapshot == P_VERSION:
                raise RuntimeError("snapshot flush failed")
    event.listen(Session, "before_flush", fail_flush)
    try:
        response, spy, service, factory, sid = run_case(tmp_path)
    finally:
        event.remove(Session, "before_flush", fail_flush)
    assert response.status_code == 500 and spy.calls == [] and service.retrievals == 1
    _assert_saved_user_only(factory, sid, "Does B link to C?", "receive-teaching-v7+history-context-b1+l1-en")
    assert "snapshot_update_rolled_back" in caplog.text


def test_snapshot_flush_failure_and_rollback_callback_failure(tmp_path, caplog):
    from sqlalchemy import event
    from sqlalchemy.orm import Session
    armed = [False]
    def fail_flush(session, flush_context, instances):
        for obj in session.dirty:
            if isinstance(obj, Message) and obj.prompt_version_snapshot == P_VERSION:
                armed[0] = True
                raise RuntimeError("snapshot flush failed")
    def fail_rollback(session):
        if armed[0]:
            raise RuntimeError("rollback failed")
    event.listen(Session, "before_flush", fail_flush)
    event.listen(Session, "after_rollback", fail_rollback)
    try:
        response, spy, service, factory, sid = run_case(tmp_path)
    finally:
        event.remove(Session, "before_flush", fail_flush)
        event.remove(Session, "after_rollback", fail_rollback)
    assert response.status_code == 500 and spy.calls == [] and service.retrievals == 1
    _assert_saved_user_only(factory, sid, "Does B link to C?", "receive-teaching-v7+history-context-b1+l1-en")
    assert "snapshot_update_rollback_failed" in caplog.text


def test_snapshot_commit_unknown_stops_generation(tmp_path, caplog):
    from sqlalchemy import event
    from sqlalchemy.orm import Session
    def fail_commit(session):
        if any(getattr(obj, "prompt_version_snapshot", None) == P_VERSION
               for obj in session.identity_map.values()):
            raise RuntimeError("commit failed")
    event.listen(Session, "before_commit", fail_commit)
    try:
        response, spy, service, factory, sid = run_case(tmp_path)
    finally:
        event.remove(Session, "before_commit", fail_commit)
    assert response.status_code == 500 and spy.calls == [] and service.retrievals == 1
    _assert_saved_user_only(factory, sid, "Does B link to C?", "receive-teaching-v7+history-context-b1+l1-en")
    assert "snapshot_commit_unknown" in caplog.text


def test_model_failure_after_snapshot_keeps_programmatic_version(tmp_path, caplog):
    response, spy, service, factory, sid = run_case(tmp_path, fail_model=True)
    assert response.status_code == 502 and len(spy.calls) == 1 and service.retrievals == 1
    db = factory()
    rows = db.query(Message).filter(Message.session_id == sid).all()
    current = [r for r in rows if r.role == "user" and r.content == "Does B link to C?"]
    assert len(current) == 1 and current[0].prompt_version_snapshot == P_VERSION
    assert not [r for r in rows if r.role == "assistant"]
    db.close()
    assert "model_failed_after_snapshot" in caplog.text


def test_real_retrieve_async_timeout_with_broken_logging(tmp_path, monkeypatch):
    from backend.programmatic_history import service as sm
    from backend.programmatic_history.types import ToolTimeout
    hits = {"executor": 0, "logger": 0}
    class Boom:
        def __init__(self, *a, **k): pass
        async def execute(self, *a, **k):
            hits["executor"] += 1
            raise ToolTimeout("forced")
    def logboom(*a, **k):
        hits["logger"] += 1
        raise RuntimeError("log failed")
    monkeypatch.setattr(sm, "ToolExecutor", Boom)
    monkeypatch.setattr(sm.logger, "warning", logboom)
    monkeypatch.setattr(sm.logger, "info", logboom)
    response, spy, service, factory, sid = run_case(tmp_path)
    assert response.status_code == 504 and spy.calls == [] and service.retrievals == 1
    assert hits["executor"] == 1 and hits["logger"] >= 1
    _assert_saved_user_only(factory, sid, "Does B link to C?", "receive-teaching-v7+history-context-b1+l1-en")


def test_real_retrieve_async_error_with_broken_logging(tmp_path, monkeypatch):
    from backend.programmatic_history import service as sm
    hits = {"executor": 0, "logger": 0}
    class Boom:
        def __init__(self, *a, **k): pass
        async def execute(self, *a, **k):
            hits["executor"] += 1
            raise RuntimeError("forced")
    def logboom(*a, **k):
        hits["logger"] += 1
        raise RuntimeError("log failed")
    monkeypatch.setattr(sm, "ToolExecutor", Boom)
    monkeypatch.setattr(sm.logger, "warning", logboom)
    monkeypatch.setattr(sm.logger, "info", logboom)
    response, spy, service, factory, sid = run_case(tmp_path)
    assert response.status_code == 502 and spy.calls == [] and service.retrievals == 1
    assert hits["executor"] == 1 and hits["logger"] >= 1
    _assert_saved_user_only(factory, sid, "Does B link to C?", "receive-teaching-v7+history-context-b1+l1-en")


def test_snapshot_after_commit_exception_is_unknown(tmp_path, caplog):
    from sqlalchemy import event
    from sqlalchemy.orm import Session
    hit = [False]
    def fail_after_commit(session):
        if any(getattr(obj, "prompt_version_snapshot", None) == P_VERSION
               for obj in session.identity_map.values()):
            hit[0] = True
            raise RuntimeError("after commit failed")
    event.listen(Session, "after_commit", fail_after_commit)
    try:
        response, spy, service, factory, sid = run_case(tmp_path)
    finally:
        event.remove(Session, "after_commit", fail_after_commit)
    assert hit[0] is True
    assert response.status_code == 500 and spy.calls == [] and service.retrievals == 1
    db = factory()
    rows = db.query(Message).filter(Message.session_id == sid).all()
    current = [r for r in rows if r.role == "user" and r.content == "Does B link to C?"]
    assert len(current) == 1 and current[0].prompt_version_snapshot == P_VERSION
    assert not [r for r in rows if r.role == "assistant"]
    db.close()
    assert "snapshot_commit_unknown" in caplog.text


def test_language_sources_are_separated_in_payload(tmp_path):
    import json
    history = [
        "I teach you this fact: B links to C. Answer in English.",
        *[f"I teach you filler fact {i}." for i in range(20)],
        "Answer in Chinese.",
    ]
    response, spy, service, _, _ = run_case(tmp_path, history=history)
    assert response.status_code == 200
    messages = spy.calls[0]
    data = json.loads(messages[1]["content"])
    assert "Answer in Chinese." in [s["content"] for s in data["window_sources"]]
    assert any("Answer in English." in s["content"] for s in data["retrieved_sources"])
    system = messages[0]["content"]
    assert "inherited only from window_sources" in system
    assert "retrieved_sources are evidence" in system


def test_memory_sqlite_is_database_unsupported(caplog):
    import backend.main as main_module
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from backend.database import create_tables, get_db
    from backend.llm_profiles import LLMProfile, LLMProfileRegistry
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    spy = Spy()
    profile = LLMProfile(id="default", label="API", kind="api", model="model-x",
                         client=spy, is_default=True)
    service = CountService(ProgrammaticSettings(True, "default", "model-x"))
    previous = main_module.chat_service
    main_module.chat_service = ChatService(
        profiles=LLMProfileRegistry([profile]), programmatic_history=service
    )
    def override_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = override_db
    db = factory()
    session = ChatSession(title="memory", llm_profile_id="default",
                          llm_model_snapshot="model-x")
    db.add(session); db.flush()
    for content in ["I teach you this fact: B links to C.", *[
        f"I teach you filler fact {i}." for i in range(20)
    ]]:
        db.add(Message(session_id=session.id, role="user", content=content,
                       interaction_mode_snapshot="receive_teaching",
                       prompt_version_snapshot="receive-teaching-v7+history-context-b1",
                       llm_profile_id_snapshot="default",
                       llm_profile_kind_snapshot="api",
                       llm_model_snapshot="model-x"))
    db.commit(); sid = session.id; db.close()
    try:
        with TestClient(app) as client:
            response = client.post(f"/api/sessions/{sid}/messages",
                                   json={"message": "Does B link to C?"})
        assert response.status_code == 200 and len(spy.calls) == 1 and service.retrievals == 0
        body = response.json()
        base = "receive-teaching-v7+history-context-b1+l1-en"
        assert body["user_message"]["prompt_version_snapshot"] == base
        assert body["assistant_message"]["prompt_version_snapshot"] == base
        assert "trigger_reason=database_unsupported" in caplog.text
    finally:
        app.dependency_overrides.pop(get_db, None)
        main_module.chat_service = previous
        engine.dispose()
