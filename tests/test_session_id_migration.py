"""Isolated tests for the durable SQLite session-id schema change."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import MetaData, text
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable

from backend.database import (
    create_database_engine,
    create_tables,
    run_migrations,
)
from backend.models import (
    Base,
    ChatSession,
    HistoryReview,
    HistoryReviewFinding,
    HistoryReviewSource,
    Message,
    ModeSwitchEvent,
)
from backend.session_id_migration import (
    SCHEMA_CURRENT,
    SCHEMA_MAINTENANCE,
    SCHEMA_NEW,
    SCHEMA_UNKNOWN,
    MODE_SWITCH_CHECK_VERSION,
    SESSION_ID_SCHEMA_VERSION,
    SessionIdMigrationError,
    backup_database,
    inspect_session_schema,
    migrate_session_id_schema,
)


def _legacy_database(tmp_path: Path):
    path = tmp_path / "legacy.db"
    url = f"sqlite:///{path}"
    engine = create_database_engine(url)
    old_table = ChatSession.__table__.to_metadata(
        MetaData(), name="chat_sessions"
    )
    old_table.dialect_options["sqlite"]["autoincrement"] = False
    with engine.begin() as conn:
        conn.execute(text(str(CreateTable(old_table).compile(dialect=sqlite_dialect.dialect()))))
        for table in Base.metadata.sorted_tables:
            if table.name != "chat_sessions":
                table.create(conn)
        conn.execute(text(
            "CREATE INDEX ix_chat_sessions_title_test "
            "ON chat_sessions(title)"
        ))
    run_migrations(engine)
    return engine, url, path


def _seed(engine, text_value="hello"):
    Local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Local() as db:
        session = ChatSession(title="legacy")
        db.add(session)
        db.commit()
        db.refresh(session)
        message = Message(
            session_id=session.id,
            role="user",
            content=text_value,
            llm_profile_id_snapshot="default",
            llm_profile_kind_snapshot="fake",
            llm_model_snapshot="fake",
            interaction_mode_snapshot="receive_teaching",
            prompt_version_snapshot="seed",
        )
        db.add(message)
        db.commit()
        event = ModeSwitchEvent(
            session_id=session.id,
            from_mode="receive_teaching",
            to_mode="corrective",
            history_through_message_id=message.id,
            history_boundary_version="history-boundary-v1",
            reviewable_user_message_count=1,
        )
        db.add(event)
        db.flush()
        review = HistoryReview(
            session_id=session.id,
            mode_switch_event_id=event.id,
            status="completed",
            lower_bound_kind="session_start",
            eligible_message_count=1,
            eligible_char_count=1,
            source_message_count=1,
            source_char_count=1,
            reviewer_llm_profile_id_snapshot="default",
            reviewer_llm_profile_kind_snapshot="fake",
            reviewer_llm_model_snapshot="fake",
            prompt_version_snapshot="history-review-v2",
            budget_version_snapshot="review-budget-v1",
            selection_policy_version="review-selection-v1",
        )
        db.add(review)
        db.flush()
        source = HistoryReviewSource(
            review_id=review.id,
            session_id=session.id,
            message_id=message.id,
            seq=1,
            content_char_count=1,
        )
        db.add(source)
        db.flush()
        finding = HistoryReviewFinding(
            review_id=review.id,
            seq=1,
            verdict="correct",
            claim_text="claim",
            source_message_id=message.id,
        )
        db.add(finding)
        db.commit()
        return session.id, message.id


def _rows(path: Path):
    conn = sqlite3.connect(path)
    try:
        sessions = conn.execute("SELECT id, title FROM chat_sessions ORDER BY id").fetchall()
        messages = conn.execute(
            "SELECT id, session_id, content FROM messages ORDER BY id"
        ).fetchall()
        triggers = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' "
            "AND tbl_name='chat_sessions' ORDER BY name"
        ).fetchall()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='chat_sessions' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
        mode_events = conn.execute(
            "SELECT id, session_id, from_mode, to_mode, "
            "history_through_message_id, reviewable_user_message_count "
            "FROM mode_switch_events ORDER BY id"
        ).fetchall()
        reviews = conn.execute(
            "SELECT id, session_id, mode_switch_event_id, status, "
            "eligible_message_count, source_message_count, "
            "reviewer_llm_model_snapshot, prompt_version_snapshot "
            "FROM history_reviews ORDER BY id"
        ).fetchall()
        sources = conn.execute(
            "SELECT id, review_id, session_id, message_id, seq, "
            "content_char_count FROM history_review_sources ORDER BY id"
        ).fetchall()
        findings = conn.execute(
            "SELECT id, review_id, seq, verdict, claim_text, "
            "source_message_id FROM history_review_findings ORDER BY id"
        ).fetchall()
        return (
            sessions, messages, triggers, indexes,
            mode_events, reviews, sources, findings,
        )
    finally:
        conn.close()


def test_new_database_uses_autoincrement_and_does_not_reuse(tmp_path):
    path = tmp_path / "new.db"
    url = f"sqlite:///{path}"
    assert inspect_session_schema(url) == SCHEMA_NEW
    engine = create_database_engine(url)
    create_tables(engine)
    assert inspect_session_schema(url) == SCHEMA_CURRENT

    Local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Local() as db:
        first = ChatSession(title="first")
        db.add(first)
        db.commit()
        first_id = first.id
        db.delete(first)
        db.commit()
        second = ChatSession(title="second")
        db.add(second)
        db.commit()
        assert second.id != first_id
    engine.dispose()


def test_legacy_database_requires_maintenance_then_migrates_related_data(tmp_path):
    engine, url, path = _legacy_database(tmp_path)
    session_id, message_id = _seed(engine)
    assert inspect_session_schema(url) == SCHEMA_MAINTENANCE
    (
        sessions_before, messages_before, triggers_before, indexes_before,
        mode_events_before, reviews_before, sources_before, findings_before,
    ) = _rows(path)
    backup = tmp_path / "backup.db"

    result = migrate_session_id_schema(url, backup)

    assert result.migrated is True
    assert result.backup_path == backup
    assert inspect_session_schema(url) == SCHEMA_CURRENT
    (
        sessions_after, messages_after, triggers_after, indexes_after,
        mode_events_after, reviews_after, sources_after, findings_after,
    ) = _rows(path)
    assert sessions_after == sessions_before
    assert messages_after == messages_before
    assert triggers_after == triggers_before
    assert indexes_after == indexes_before
    assert mode_events_after == mode_events_before
    assert reviews_after == reviews_before
    assert sources_after == sources_before
    assert findings_after == findings_before
    assert backup.exists()
    conn = sqlite3.connect(path)
    try:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_sessions'"
        ).fetchone()[0]
        assert "AUTOINCREMENT" in ddl.upper()
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == []
        version = conn.execute(
            "SELECT version FROM schema_migrations WHERE version=?",
            (SESSION_ID_SCHEMA_VERSION,),
        ).fetchone()
        assert version is not None
        conn.execute(
            "INSERT INTO chat_sessions(title, created_at, updated_at) "
            "VALUES ('after migration', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
        next_id = conn.execute(
            "SELECT id FROM chat_sessions WHERE title='after migration'"
        ).fetchone()[0]
        assert next_id > max(row[0] for row in sessions_after)
    finally:
        conn.close()
    assert session_id == sessions_after[0][0]
    assert message_id == messages_after[0][0]
    engine.dispose()


def test_legacy_migration_is_idempotent_and_backup_preserves_uncheckpointed_wal(tmp_path):
    engine, url, path = _legacy_database(tmp_path)
    _seed(engine)
    engine.dispose()

    # Keep a WAL connection open with autocheckpoint disabled.  The backup
    # must see committed frames that have not been checkpointed into the main
    # file yet; this is deliberately not a PRAGMA wal_checkpoint test.
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute(
            "INSERT INTO chat_sessions(title, created_at, updated_at) "
            "VALUES ('wal-uncheckpointed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
        conn.commit()
        backup = tmp_path / "wal-backup.db"
        backup_database(url, backup)
        assert backup.exists()
        copied = sqlite3.connect(backup)
        try:
            found = copied.execute(
                "SELECT 1 FROM chat_sessions WHERE title='wal-uncheckpointed'"
            ).fetchone()
            assert found is not None
        finally:
            copied.close()
    finally:
        conn.close()

    first = migrate_session_id_schema(url, backup, backup_already_done=True)
    second = migrate_session_id_schema(url, tmp_path / "unused.db")
    assert first.migrated is True
    assert second.migrated is False
    assert inspect_session_schema(url) == SCHEMA_CURRENT
    assert not (tmp_path / "unused.db").exists()


@pytest.mark.parametrize(
    "stage",
    [
        "after_create_table",
        "after_copy",
        "after_drop",
        "after_rename",
        "after_restore_objects",
        "after_version",
    ],
)
def test_migration_failure_rolls_back_and_keeps_old_table(tmp_path, stage):
    engine, url, path = _legacy_database(tmp_path)
    session_id, _message_id = _seed(engine)
    before = _rows(path)
    backup = tmp_path / f"backup-{stage}.db"

    def fail(when):
        if when == stage:
            raise RuntimeError(f"injected {stage}")

    with pytest.raises(SessionIdMigrationError) as exc_info:
        migrate_session_id_schema(url, backup, failure_hook=fail)

    assert exc_info.value.committed is False
    assert exc_info.value.cleanup_error is None
    assert inspect_session_schema(url) == SCHEMA_MAINTENANCE
    assert _rows(path) == before
    assert backup.exists()
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT id FROM chat_sessions").fetchone()[0] == session_id
    finally:
        conn.close()
    engine.dispose()


def test_backup_targets_must_be_explicit_and_not_overwritten(tmp_path):
    engine, url, path = _legacy_database(tmp_path)
    _seed(engine)
    existing = tmp_path / "exists.db"
    existing.write_bytes(b"not a database")
    with pytest.raises(Exception):
        backup_database(url, existing)
    assert existing.read_bytes() == b"not a database"

    inside_repo = Path(__file__).resolve().parent.parent / "should-not-backup.db"
    with pytest.raises(Exception):
        backup_database(url, inside_repo)
    assert not inside_repo.exists()
    engine.dispose()


def test_unknown_sqlite_schema_is_not_treated_as_new(tmp_path):
    path = tmp_path / "unrelated.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    assert inspect_session_schema(f"sqlite:///{path}") == SCHEMA_UNKNOWN


def test_startup_preflight_rejects_legacy_without_writing(tmp_path):
    engine, url, path = _legacy_database(tmp_path)
    _seed(engine)
    conn = sqlite3.connect(path)
    try:
        before_sessions = conn.execute(
            "SELECT id, title FROM chat_sessions ORDER BY id"
        ).fetchall()
        before_version = conn.execute(
            "SELECT version FROM schema_migrations WHERE version=?",
            (SESSION_ID_SCHEMA_VERSION,),
        ).fetchall()
        before_ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_sessions'"
        ).fetchone()[0]
    finally:
        conn.close()

    from backend.session_id_migration import (
        SessionIdMigrationRequired,
        assert_startup_schema,
    )

    with pytest.raises(SessionIdMigrationRequired):
        assert_startup_schema(url)

    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT id, title FROM chat_sessions ORDER BY id"
        ).fetchall() == before_sessions
        assert conn.execute(
            "SELECT version FROM schema_migrations WHERE version=?",
            (SESSION_ID_SCHEMA_VERSION,),
        ).fetchall() == before_version
        assert conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_sessions'"
        ).fetchone()[0] == before_ddl
    finally:
        conn.close()
    engine.dispose()


def test_migration_restore_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    engine, url, path = _legacy_database(tmp_path)
    _seed(engine)
    from backend import session_id_migration as sim

    monkeypatch.setattr(
        sim, "_restore_foreign_keys", lambda conn: (False, "injected cleanup")
    )
    with pytest.raises(SessionIdMigrationError) as exc_info:
        sim.migrate_session_id_schema(url, tmp_path / "restore-backup.db")
    assert exc_info.value.committed is True
    assert exc_info.value.cleanup_error == "injected cleanup"
    assert sim.inspect_session_schema(url) == SCHEMA_CURRENT
    engine.dispose()


def test_explicit_maintenance_entry_backs_up_then_migrates(tmp_path):
    engine, url, path = _legacy_database(tmp_path)
    _seed(engine)
    engine.dispose()
    backup = tmp_path / "maintenance-backup.db"
    from backend.session_id_migration import maintenance_command

    assert maintenance_command(url, str(backup)) == 0
    assert backup.exists()
    assert inspect_session_schema(url) == SCHEMA_CURRENT
    sessions, messages, triggers, indexes, *_ = _rows(path)
    assert len(sessions) == 1
    assert len(messages) == 1
    assert triggers and indexes


def _legacy_database_with_extra_unique(tmp_path: Path):
    path = tmp_path / "legacy-unique.db"
    url = f"sqlite:///{path}"
    engine = create_database_engine(url)
    from sqlalchemy import UniqueConstraint

    old_table = ChatSession.__table__.to_metadata(
        MetaData(), name="chat_sessions"
    )
    old_table.dialect_options["sqlite"]["autoincrement"] = False
    old_table.append_constraint(UniqueConstraint("title", name="uq_title_review"))
    with engine.begin() as conn:
        conn.execute(text(str(CreateTable(old_table).compile(dialect=sqlite_dialect.dialect()))))
        for table in Base.metadata.sorted_tables:
            if table.name != "chat_sessions":
                table.create(conn)
    run_migrations(engine)
    return engine, url, path


def test_extra_table_constraint_requires_manual_unknown(tmp_path):
    engine, url, path = _legacy_database_with_extra_unique(tmp_path)
    engine.dispose()
    assert inspect_session_schema(url) == SCHEMA_UNKNOWN
    from backend.session_id_migration import SessionSchemaUnknown

    with pytest.raises(SessionSchemaUnknown):
        migrate_session_id_schema(url, tmp_path / "must-not-exist.db")
    assert not (tmp_path / "must-not-exist.db").exists()


def test_view_and_missing_core_table_are_unknown(tmp_path):
    path = tmp_path / "current.db"
    url = f"sqlite:///{path}"
    engine = create_database_engine(url)
    create_tables(engine)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE VIEW application_view AS SELECT 1 AS value")
        conn.commit()
    finally:
        conn.close()
    assert inspect_session_schema(url) == SCHEMA_UNKNOWN
    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP VIEW application_view")
        conn.execute("DROP TABLE history_review_findings")
        conn.commit()
    finally:
        conn.close()
    assert inspect_session_schema(url) == SCHEMA_UNKNOWN
    engine.dispose()


def test_version_conflict_on_legacy_table_is_unknown(tmp_path):
    engine, url, path = _legacy_database(tmp_path)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
            (SESSION_ID_SCHEMA_VERSION,),
        )
        conn.commit()
    finally:
        conn.close()
    assert inspect_session_schema(url) == SCHEMA_UNKNOWN
    engine.dispose()


def test_backup_rejects_final_and_parent_symlinks(tmp_path):
    engine, url, path = _legacy_database(tmp_path)
    engine.dispose()
    repo_file = Path(__file__).resolve().parent.parent / "README.md"
    final_link = tmp_path / "final-link.db"
    final_link.symlink_to(repo_file)
    with pytest.raises(Exception):
        backup_database(url, final_link)

    parent_link = tmp_path / "repo-parent"
    parent_link.symlink_to(repo_file.parent, target_is_directory=True)
    with pytest.raises(Exception):
        backup_database(url, parent_link / "nested" / "backup.db")


class _ConnectionProxy:
    def __init__(
        self,
        inner,
        *,
        commit_failure=False,
        commit_then_raise=False,
        rollback_failure=False,
        close_failure=False,
    ):
        self._inner = inner
        self.commit_failure = commit_failure
        self.commit_then_raise = commit_then_raise
        self.rollback_failure = rollback_failure
        self.close_failure = close_failure

    @property
    def in_transaction(self):
        return self._inner.in_transaction

    def execute(self, sql, *args):
        normalized = " ".join(str(sql).split()).upper()
        if normalized == "COMMIT" and self.commit_failure:
            if self.commit_then_raise:
                self._inner.execute(sql, *args)
            raise sqlite3.OperationalError("synthetic commit failure")
        if normalized == "ROLLBACK" and self.rollback_failure:
            raise sqlite3.OperationalError("synthetic rollback failure")
        return self._inner.execute(sql, *args)

    def close(self):
        if self.close_failure:
            raise sqlite3.OperationalError("synthetic close failure")
        return self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _run_with_proxy(tmp_path, proxy_kwargs, *, failure_hook=None):
    engine, url, path = _legacy_database(tmp_path)
    engine.dispose()
    from backend import session_id_migration as sim

    def factory(path_arg):
        return _ConnectionProxy(
            sqlite3.connect(str(path_arg), isolation_level=None), **proxy_kwargs
        )

    with pytest.raises(SessionIdMigrationError) as exc_info:
        sim.migrate_session_id_schema(
            url,
            tmp_path / "proxy-backup.db",
            backup_already_done=True,
            failure_hook=failure_hook,
            connection_factory=factory,
        )
    return exc_info.value, url, path


def test_commit_failure_is_unknown_and_rollback_is_reported(tmp_path):
    from backend.session_id_migration import (
        COMMIT_UNKNOWN,
        ROLLBACK_SUCCEEDED,
    )

    error, url, _ = _run_with_proxy(
        tmp_path, {"commit_failure": True}, failure_hook=None
    )
    assert error.commit_state == COMMIT_UNKNOWN
    assert error.rollback_state == ROLLBACK_SUCCEEDED
    assert error.committed is None


def test_commit_then_raise_is_unknown_without_rollback(tmp_path):
    from backend.session_id_migration import (
        COMMIT_UNKNOWN,
        ROLLBACK_NOT_ATTEMPTED,
    )

    error, url, _ = _run_with_proxy(
        tmp_path, {"commit_failure": True, "commit_then_raise": True}
    )
    assert error.commit_state == COMMIT_UNKNOWN
    assert error.rollback_state == ROLLBACK_NOT_ATTEMPTED
    assert inspect_session_schema(url) == SCHEMA_CURRENT


def test_rollback_failure_is_reported_as_failed(tmp_path):
    from backend.session_id_migration import ROLLBACK_FAILED

    def fail_after_copy(stage):
        if stage == "after_copy":
            raise RuntimeError("synthetic after copy")

    error, _, _ = _run_with_proxy(
        tmp_path, {"rollback_failure": True}, failure_hook=fail_after_copy
    )
    assert error.rollback_state == ROLLBACK_FAILED
    assert error.committed is False


def test_close_failure_is_cleanup_error_not_success(tmp_path):
    from backend.session_id_migration import (
        COMMIT_COMMITTED,
        ROLLBACK_NOT_ATTEMPTED,
    )

    error, url, _ = _run_with_proxy(tmp_path, {"close_failure": True})
    assert error.commit_state == COMMIT_COMMITTED
    assert error.rollback_state == ROLLBACK_NOT_ATTEMPTED
    assert error.cleanup_error
    assert inspect_session_schema(url) == SCHEMA_CURRENT


def test_startup_legacy_and_unknown_stop_before_lifespan_writes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.main as main_module
    from backend.session_id_migration import (
        SessionIdMigrationRequired,
        SessionSchemaUnknown,
    )

    calls = []
    monkeypatch.setattr(main_module, "create_tables", lambda: calls.append("create"))
    monkeypatch.setattr(main_module, "run_migrations", lambda engine: calls.append("migrate"))
    monkeypatch.setattr(
        main_module,
        "record_session_schema_version",
        lambda engine: calls.append("version"),
    )
    monkeypatch.setattr(
        main_module.history_review_service,
        "recover_abandoned_running_reviews",
        lambda db: calls.append("recovery"),
    )

    engine, legacy_url, _ = _legacy_database(tmp_path)
    engine.dispose()
    monkeypatch.setattr(main_module, "database_url", legacy_url)
    with pytest.raises(SessionIdMigrationRequired):
        with TestClient(main_module.app):
            pass
    assert calls == []

    unknown_path = tmp_path / "unknown.db"
    conn = sqlite3.connect(unknown_path)
    try:
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(
        main_module, "database_url", f"sqlite:///{unknown_path}"
    )
    with pytest.raises(SessionSchemaUnknown):
        with TestClient(main_module.app):
            pass
    assert calls == []


def test_cli_reports_safe_states_and_startup_hint(capsys, monkeypatch):
    from backend import session_id_migration as sim

    def fail(database_url, backup_path):
        raise sim.SessionIdMigrationError(
            "after_copy",
            committed=False,
            commit_state=sim.COMMIT_UNKNOWN,
            rollback_state=sim.ROLLBACK_FAILED,
            cleanup_error="RAW-SECRET-INTERNAL",
        )

    monkeypatch.setattr(sim, "maintenance_command", fail)
    assert sim.main(["--database-url", "sqlite:///x", "--backup-path", "/tmp/x"]) == 1
    err = capsys.readouterr().err
    assert "stage=after_copy" in err
    assert "commit_state=commit_unknown" in err
    assert "rollback_state=rollback_failed" in err
    assert "cleanup_failed=yes" in err
    assert "RAW-SECRET-INTERNAL" not in err

    required = sim.SessionIdMigrationRequired("legacy", committed=False)
    text = str(required)
    assert "backend.session_id_migration" in text
    assert "--database-url" in text
    assert "--backup-path" in text
    assert "backup" in text.lower()


def test_empty_file_is_new_but_autoincrement_only_table_is_unknown(tmp_path):
    empty = tmp_path / "empty.db"
    empty.touch()
    assert inspect_session_schema(f"sqlite:///{empty}") == SCHEMA_NEW

    malformed = tmp_path / "malformed.db"
    conn = sqlite3.connect(malformed)
    try:
        conn.execute(
            "CREATE TABLE chat_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT)"
        )
        conn.commit()
    finally:
        conn.close()
    assert inspect_session_schema(f"sqlite:///{malformed}") == SCHEMA_UNKNOWN


def test_maintenance_backup_failure_does_not_modify_database(tmp_path):
    engine, url, path = _legacy_database(tmp_path)
    _seed(engine)
    engine.dispose()
    before = _rows(path)
    existing = tmp_path / "existing-backup.db"
    existing.write_bytes(b"keep")
    from backend.session_id_migration import BackupError, maintenance_command

    with pytest.raises(BackupError):
        maintenance_command(url, str(existing))
    assert existing.read_bytes() == b"keep"
    assert _rows(path) == before
    assert inspect_session_schema(url) == SCHEMA_MAINTENANCE


def test_sqliteX_user_table_is_unknown_and_lifespan_stops_before_writes(tmp_path, monkeypatch):
    path = tmp_path / "sqlitex.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE sqliteX_user_data (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    url = f"sqlite:///{path}"
    assert inspect_session_schema(url) == SCHEMA_UNKNOWN

    from fastapi.testclient import TestClient
    import backend.main as main_module
    from backend.session_id_migration import SessionSchemaUnknown

    calls = []
    monkeypatch.setattr(main_module, "database_url", url)
    monkeypatch.setattr(main_module, "create_tables", lambda: calls.append("create"))
    monkeypatch.setattr(main_module, "run_migrations", lambda engine: calls.append("migrate"))
    monkeypatch.setattr(
        main_module,
        "record_session_schema_version",
        lambda engine: calls.append("version"),
    )
    monkeypatch.setattr(
        main_module.history_review_service,
        "recover_abandoned_running_reviews",
        lambda db: calls.append("recovery"),
    )
    with pytest.raises(SessionSchemaUnknown):
        with TestClient(main_module.app):
            pass
    assert calls == []


def test_cli_hints_distinguish_limited_rollback_and_cleanup(capsys, monkeypatch):
    from backend import session_id_migration as sim

    cases = [
        (
            sim.SessionIdMigrationError(
                "copy",
                committed=None,
                commit_state=sim.COMMIT_UNKNOWN,
                rollback_state=sim.ROLLBACK_SUCCEEDED,
            ),
            ["commit outcome is unknown", "do not start"],
        ),
        (
            sim.SessionIdMigrationError(
                "copy",
                committed=False,
                commit_state=sim.COMMIT_NOT_COMMITTED,
                rollback_state=sim.ROLLBACK_SUCCEEDED,
            ),
            [
                "limited checks",
                "foreign_key_check",
                "associated data was not independently audited",
            ],
        ),
        (
            sim.SessionIdMigrationError(
                "copy",
                committed=False,
                commit_state=sim.COMMIT_NOT_COMMITTED,
                rollback_state=sim.ROLLBACK_FAILED,
            ),
            ["rollback is not confirmed"],
        ),
        (
            sim.SessionIdMigrationError(
                "cleanup",
                committed=True,
                commit_state=sim.COMMIT_COMMITTED,
                rollback_state=sim.ROLLBACK_NOT_ATTEMPTED,
                cleanup_error="RAW-CLEANUP",
            ),
            ["migration committed, but cleanup failed", "do not start"],
        ),
    ]

    for error, expected in cases:
        monkeypatch.setattr(
            sim,
            "maintenance_command",
            lambda database_url, backup_path, error=error: (_ for _ in ()).throw(error),
        )
        assert sim.main(["--database-url", "sqlite:///x", "--backup-path", "/tmp/x"]) == 1
        err = capsys.readouterr().err
        assert "RAW-CLEANUP" not in err
        for text in expected:
            assert text in err


def test_pre_fix_baseline_database_is_accepted_by_current_migration(tmp_path):
    """Generate a legacy DB with the actual pre-fix backend source."""
    import io
    import os
    import subprocess
    import sys
    import tarfile

    baseline = "fe68b250c65c448a0f220b9fa7856a03e6f3dff2"
    root = Path(__file__).resolve().parent.parent
    archive = subprocess.run(
        ["git", "archive", baseline, "backend"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    source_dir = tmp_path / "baseline-source"
    source_dir.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(source_dir, filter="data")

    legacy_db = tmp_path / "baseline.db"
    legacy_url = f"sqlite:///{legacy_db}"
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(source_dir),
        "DATABASE_URL": legacy_url,
        "LLM_MODE": "fake",
        "LLM_API_KEY": "",
        "LLM_API_BASE_URL": "",
        "LLM_MODEL": "",
        "LLM_REASONING_EFFORT": "",
        "LOCAL_LLM_ENABLED": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    script = (
        "from backend.database import create_tables, engine, run_migrations\n"
        "create_tables()\n"
        "run_migrations(engine)\n"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert inspect_session_schema(legacy_url) == SCHEMA_MAINTENANCE
    engine = create_database_engine(legacy_url)
    _seed(engine)
    engine.dispose()
    before = _rows(legacy_db)
    backup = tmp_path / "baseline-backup.db"
    from backend.session_id_migration import maintenance_command

    assert maintenance_command(legacy_url, str(backup)) == 0
    assert inspect_session_schema(legacy_url) == SCHEMA_CURRENT
    assert _rows(legacy_db) == before

    engine = create_database_engine(legacy_url)
    Local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Local() as db:
        highest = db.query(ChatSession).order_by(ChatSession.id.desc()).first()
        old_id = highest.id
        db.delete(highest)
        db.commit()
        replacement = ChatSession(title="after baseline migration")
        db.add(replacement)
        db.commit()
        assert replacement.id > old_id
    engine.dispose()


def _legacy_without_mode_check(tmp_path: Path):
    engine, url, path = _legacy_database(tmp_path)
    with engine.begin() as conn:
        old_sql = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' "
                 "AND name='mode_switch_events'")
        ).scalar()
        new_sql = re.sub(
            r",\s*CONSTRAINT\s+ck_mode_switch_events_reviewable_count\s+"
            r"CHECK\s*\([^)]*\)",
            "",
            old_sql,
            flags=re.IGNORECASE,
        )
        assert "ck_mode_switch_events_reviewable_count" not in new_sql
        new_sql = new_sql.replace(
            "CREATE TABLE mode_switch_events",
            "CREATE TABLE mode_switch_events_old",
            1,
        )
        conn.execute(text(new_sql))
        columns = [
            row[1] for row in conn.execute(
                text("PRAGMA table_info(mode_switch_events)")
            )
        ]
        quoted = ", ".join(f'"{c}"' for c in columns)
        conn.execute(text(
            f'INSERT INTO mode_switch_events_old ({quoted}) '
            f'SELECT {quoted} FROM mode_switch_events'
        ))
        conn.execute(text("DROP TABLE mode_switch_events"))
        conn.execute(text(
            "ALTER TABLE mode_switch_events_old RENAME TO mode_switch_events"
        ))
    assert inspect_session_schema(url) == SCHEMA_MAINTENANCE
    return engine, url, path


def test_known_legacy_mode_check_is_repaired_with_version(tmp_path):
    engine, url, path = _legacy_without_mode_check(tmp_path)
    _seed(engine)
    backup = tmp_path / "mode-backup.db"
    result = migrate_session_id_schema(url, backup)
    assert result.migrated is True
    assert inspect_session_schema(url) == SCHEMA_CURRENT
    conn = sqlite3.connect(path)
    try:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='mode_switch_events'"
        ).fetchone()[0]
        assert "reviewable_user_message_count" in ddl
        versions = {
            row[0] for row in conn.execute(
                "SELECT version FROM schema_migrations"
            )
        }
        assert MODE_SWITCH_CHECK_VERSION in versions
        assert SESSION_ID_SCHEMA_VERSION in versions
        assert conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall() == []
    finally:
        conn.close()
    engine.dispose()


def test_negative_reviewable_count_fails_before_backup(tmp_path):
    engine, url, path = _legacy_without_mode_check(tmp_path)
    _seed(engine)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE mode_switch_events SET reviewable_user_message_count=-1"
        )
        conn.commit()
    finally:
        conn.close()
    backup = tmp_path / "must-not-be-created.db"
    with pytest.raises(SessionIdMigrationError) as exc_info:
        migrate_session_id_schema(url, backup)
    assert exc_info.value.stage == "validate_mode_count"
    assert not backup.exists()
    engine.dispose()


def test_mode_rebuild_failure_rolls_back_both_tables(tmp_path):
    engine, url, path = _legacy_without_mode_check(tmp_path)
    _seed(engine)
    before = _rows(path)
    backup = tmp_path / "mode-failure.db"

    def fail(stage):
        if stage == "after_mode_copy":
            raise RuntimeError("injected mode copy")

    with pytest.raises(SessionIdMigrationError) as exc_info:
        migrate_session_id_schema(
            url, backup, failure_hook=fail
        )
    assert exc_info.value.committed is False
    assert inspect_session_schema(url) == SCHEMA_MAINTENANCE
    assert _rows(path) == before
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version=?",
            (MODE_SWITCH_CHECK_VERSION,),
        ).fetchone()[0] == 0
    finally:
        conn.close()
    engine.dispose()


def _messages_with_index(tmp_path: Path, index_sql: str):
    engine, url, path = _legacy_without_mode_check(tmp_path)
    engine.dispose()
    from sqlalchemy.schema import CreateTable

    raw = str(
        CreateTable(Message.__table__).compile(dialect=sqlite_dialect.dialect())
    )
    raw = re.sub(
        r",\s*CONSTRAINT\s+uq_messages_id_session\s+"
        r"UNIQUE\s*\(id,\s*session_id\)",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = raw.replace(
        "CREATE TABLE messages", "CREATE TABLE messages_old", 1
    )
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(raw)
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(messages)")
        ]
        quoted = ", ".join(f'"{c}"' for c in columns)
        conn.execute(
            f'INSERT INTO messages_old ({quoted}) '
            f'SELECT {quoted} FROM messages'
        )
        conn.execute("DROP TABLE messages")
        conn.execute(
            "ALTER TABLE messages_old RENAME TO messages"
        )
        conn.execute(index_sql)
        conn.commit()
    finally:
        conn.close()
    return url, path


@pytest.mark.parametrize(
    "index_sql",
    [
        "CREATE UNIQUE INDEX uq_messages_id_session "
        "ON messages(id, session_id) WHERE id > 0",
        "CREATE UNIQUE INDEX uq_messages_id_session "
        "ON messages(session_id)",
        "CREATE INDEX uq_messages_id_session "
        "ON messages(id, session_id)",
    ],
)
def test_wrong_unique_index_shapes_are_unknown(tmp_path, index_sql):
    url, _path = _messages_with_index(tmp_path, index_sql)
    assert inspect_session_schema(url) == SCHEMA_UNKNOWN


def test_confirmed_standalone_composite_unique_index_is_accepted(tmp_path):
    url, _path = _messages_with_index(
        tmp_path,
        "CREATE UNIQUE INDEX uq_messages_id_session "
        "ON messages(id, session_id)",
    )
    assert inspect_session_schema(url) == SCHEMA_MAINTENANCE


def _current_without_mode_check(tmp_path: Path):
    engine = create_database_engine(f"sqlite:///{tmp_path / 'mode-only.db'}")
    create_tables(engine)
    path = tmp_path / "mode-only.db"
    conn = sqlite3.connect(path)
    try:
        raw = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='mode_switch_events'"
        ).fetchone()[0]
        raw = re.sub(
            r",\s*CONSTRAINT\s+ck_mode_switch_events_reviewable_count\s+"
            r"CHECK\s*\([^)]*\)",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        raw = raw.replace(
            "CREATE TABLE mode_switch_events",
            "CREATE TABLE mode_switch_events_old",
            1,
        )
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(raw)
        columns = [
            row[1] for row in conn.execute(
                "PRAGMA table_info(mode_switch_events)"
            )
        ]
        quoted = ", ".join(f'"{c}"' for c in columns)
        conn.execute(
            f'INSERT INTO mode_switch_events_old ({quoted}) '
            f'SELECT {quoted} FROM mode_switch_events'
        )
        conn.execute("DROP TABLE mode_switch_events")
        conn.execute(
            "ALTER TABLE mode_switch_events_old RENAME TO mode_switch_events"
        )
        conn.commit()
    finally:
        conn.close()
    return engine, f"sqlite:///{path}", path


def test_mode_only_repair_does_not_rebuild_autoincrement_table(tmp_path):
    engine, url, path = _current_without_mode_check(tmp_path)
    Local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Local() as db:
        session = ChatSession(title="existing")
        db.add(session)
        db.commit()
        db.refresh(session)
        old_id = session.id
    before_ddl = sqlite3.connect(path).execute(
        "SELECT sql FROM sqlite_master WHERE type='table' "
        "AND name='chat_sessions'"
    ).fetchone()[0]
    assert inspect_session_schema(url) == SCHEMA_MAINTENANCE
    migrate_session_id_schema(url, tmp_path / "mode-only-backup.db")
    assert inspect_session_schema(url) == SCHEMA_CURRENT
    conn = sqlite3.connect(path)
    try:
        after_ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='chat_sessions'"
        ).fetchone()[0]
        assert after_ddl == before_ddl
        assert conn.execute(
            "SELECT version FROM schema_migrations WHERE version=?",
            (MODE_SWITCH_CHECK_VERSION,),
        ).fetchone() is not None
        # Existing AUTOINCREMENT high-water must not be reset by a mode-only
        # maintenance pass.
        conn.execute(
            "INSERT INTO chat_sessions(title, created_at, updated_at) "
            "VALUES ('next', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
        new_id = conn.execute(
            "SELECT id FROM chat_sessions WHERE title='next'"
        ).fetchone()[0]
        assert new_id > old_id
    finally:
        conn.close()
    engine.dispose()


def _comment_autoincrement_database(tmp_path: Path):
    engine, url, path = _legacy_database(tmp_path)
    conn = sqlite3.connect(path)
    try:
        old_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='chat_sessions'"
        ).fetchone()[0]
        commented = old_sql.replace(
            "id INTEGER NOT NULL",
            "id INTEGER NOT NULL /* AUTOINCREMENT */",
            1,
        )
        assert "/* AUTOINCREMENT */" in commented
        commented = commented.replace(
            "CREATE TABLE chat_sessions",
            "CREATE TABLE chat_sessions_old",
            1,
        )
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(chat_sessions)")
        ]
        quoted = ", ".join(f'"{c}"' for c in columns)
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(commented)
        conn.execute(
            f'INSERT INTO chat_sessions_old ({quoted}) '
            f'SELECT {quoted} FROM chat_sessions'
        )
        conn.execute("DROP TABLE chat_sessions")
        conn.execute(
            "ALTER TABLE chat_sessions_old RENAME TO chat_sessions"
        )
        conn.execute(
            "CREATE INDEX ix_chat_sessions_title_test "
            "ON chat_sessions(title)"
        )
        conn.commit()
    finally:
        conn.close()
    return engine, url, path


def test_comment_autoincrement_is_not_current_and_migrates(tmp_path):
    from backend.session_id_migration import _has_effective_autoincrement

    assert not _has_effective_autoincrement(
        "CREATE TABLE chat_sessions (\n"
        " id INTEGER /* AUTOINCREMENT */,\n"
        " PRIMARY KEY (id)\n"
        ")"
    )
    assert not _has_effective_autoincrement(
        'CREATE TABLE chat_sessions (id INTEGER DEFAULT "AUTOINCREMENT", '
        "PRIMARY KEY (id))"
    )
    assert not _has_effective_autoincrement(
        "CREATE TABLE chat_sessions (id INTEGER DEFAULT 'AUTOINCREMENT', "
        "PRIMARY KEY (id))"
    )
    assert _has_effective_autoincrement(
        "CREATE TABLE chat_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT)"
    )

    engine, url, path = _comment_autoincrement_database(tmp_path)
    assert inspect_session_schema(url) == SCHEMA_MAINTENANCE
    migrate_session_id_schema(url, tmp_path / "comment-backup.db")
    assert inspect_session_schema(url) == SCHEMA_CURRENT
    conn = sqlite3.connect(path)
    try:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='chat_sessions'"
        ).fetchone()[0]
        assert "AUTOINCREMENT" in ddl.upper()
        assert "/*" not in ddl
        assert conn.execute(
            "SELECT version FROM schema_migrations WHERE version=?",
            (SESSION_ID_SCHEMA_VERSION,),
        ).fetchone() is not None
    finally:
        conn.close()
    engine.dispose()
