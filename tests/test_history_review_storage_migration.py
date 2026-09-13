"""Iteration 2B storage migration tests."""

import sqlite3

import pytest
from sqlalchemy import text as sa_text

from backend.database import (
    create_database_engine,
    create_tables,
    run_migrations,
)

_PRIOR_MIGRATIONS = (
    "title_is_manual_v1",
    "llm_profile_v1",
    "message_llm_snapshot_v1",
    "interaction_mode_v1",
    "interaction_mode_constraints_v1",
    "history_review_boundary_v1",
)


def _make_iteration_2a_db(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute("PRAGMA foreign_keys = ON")
    raw.execute(
        "CREATE TABLE chat_sessions ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
        "  created_at DATETIME NOT NULL,"
        "  updated_at DATETIME NOT NULL,"
        "  title_is_manual INTEGER NOT NULL DEFAULT 0,"
        "  llm_profile_id VARCHAR(50) NOT NULL DEFAULT 'default',"
        "  llm_model_snapshot VARCHAR(255),"
        "  interaction_mode VARCHAR(30) NOT NULL "
        "DEFAULT 'receive_teaching'"
        ")"
    )
    raw.execute(
        "CREATE TABLE messages ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id INTEGER NOT NULL REFERENCES chat_sessions(id) "
        "ON DELETE CASCADE,"
        "  role VARCHAR(20) NOT NULL,"
        "  content TEXT NOT NULL,"
        "  created_at DATETIME NOT NULL,"
        "  llm_profile_id_snapshot VARCHAR(50),"
        "  llm_profile_kind_snapshot VARCHAR(20),"
        "  llm_model_snapshot VARCHAR(255),"
        "  interaction_mode_snapshot VARCHAR(30),"
        "  prompt_version_snapshot VARCHAR(50)"
        ")"
    )
    raw.execute(
        "CREATE TABLE mode_switch_events ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id INTEGER NOT NULL REFERENCES chat_sessions(id) "
        "ON DELETE CASCADE,"
        "  from_mode VARCHAR(30) NOT NULL,"
        "  to_mode VARCHAR(30) NOT NULL,"
        "  created_at DATETIME NOT NULL,"
        "  history_through_message_id INTEGER,"
        "  history_boundary_version VARCHAR(50),"
        "  reviewable_user_message_count INTEGER"
        ")"
    )
    raw.execute(
        "CREATE TABLE schema_migrations ("
        "  version VARCHAR(255) PRIMARY KEY,"
        "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    for version in _PRIOR_MIGRATIONS:
        raw.execute(
            "INSERT INTO schema_migrations (version) VALUES (?)",
            (version,),
        )
    raw.execute(
        "INSERT INTO chat_sessions "
        "(title, created_at, updated_at) "
        "VALUES ('Existing', '2026-01-01', '2026-01-01')"
    )
    raw.execute(
        "INSERT INTO messages "
        "(session_id, role, content, created_at, "
        "interaction_mode_snapshot) "
        "VALUES (1, 'user', 'existing teaching', '2026-01-02', "
        "'receive_teaching')"
    )
    raw.execute(
        "INSERT INTO mode_switch_events "
        "(session_id, from_mode, to_mode, created_at, "
        "history_through_message_id, history_boundary_version) "
        "VALUES (1, 'receive_teaching', 'corrective', '2026-01-03', "
        "1, 'history-boundary-v1')"
    )
    raw.commit()
    raw.close()


def _table_names(engine):
    with engine.begin() as conn:
        return {
            row[0]
            for row in conn.execute(
                sa_text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }


def _record_count(engine, version):
    with engine.begin() as conn:
        return conn.execute(
            sa_text(
                "SELECT COUNT(*) FROM schema_migrations "
                "WHERE version = :version"
            ),
            {"version": version},
        ).scalar()


class TestHistoryReviewStorageMigration:
    def test_upgrade_preserves_old_data_and_creates_schema(self, tmp_path):
        db_path = tmp_path / "old.db"
        _make_iteration_2a_db(db_path)

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(engine)
            assert {
                "history_reviews",
                "history_review_sources",
                "history_review_findings",
            }.issubset(_table_names(engine))
            with engine.begin() as conn:
                assert conn.execute(
                    sa_text(
                        "SELECT COUNT(*) FROM chat_sessions"
                    )
                ).scalar() == 1
                assert conn.execute(
                    sa_text(
                        "SELECT COUNT(*) FROM messages"
                    )
                ).scalar() == 1
                assert conn.execute(
                    sa_text(
                        "SELECT COUNT(*) FROM mode_switch_events"
                    )
                ).scalar() == 1
                indexes = {
                    row[0]
                    for row in conn.execute(
                        sa_text(
                            "SELECT name FROM sqlite_master "
                            "WHERE type = 'index'"
                        )
                    ).fetchall()
                }
                assert "uq_messages_id_session" in indexes
                assert "uq_mode_switch_events_id_session" in indexes
                assert conn.execute(
                    sa_text("PRAGMA foreign_key_check")
                ).fetchall() == []
            assert _record_count(engine, "history_review_storage_v1") == 1
        finally:
            engine.dispose()

    def test_migration_is_idempotent(self, tmp_path):
        db_path = tmp_path / "idempotent.db"
        _make_iteration_2a_db(db_path)

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(engine)
            run_migrations(engine)
            assert _record_count(engine, "history_review_storage_v1") == 1
        finally:
            engine.dispose()

    def test_create_tables_then_migrate(self, tmp_path):
        engine = create_database_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
        try:
            create_tables(bind=engine)
            run_migrations(engine)
            run_migrations(engine)
            assert {
                "history_reviews",
                "history_review_sources",
                "history_review_findings",
            }.issubset(_table_names(engine))
            assert _record_count(engine, "history_review_storage_v1") == 1
        finally:
            engine.dispose()

    def test_record_insert_failure_then_retry_converges(self, tmp_path):
        db_path = tmp_path / "retry.db"
        _make_iteration_2a_db(db_path)

        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "CREATE TRIGGER fail_storage_migration "
            "BEFORE INSERT ON schema_migrations "
            "WHEN NEW.version = 'history_review_storage_v1' "
            "BEGIN SELECT RAISE(FAIL, 'simulated failure'); END"
        )
        raw.commit()
        raw.close()

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            with pytest.raises(Exception, match="simulated failure"):
                run_migrations(engine)

            raw2 = sqlite3.connect(str(db_path))
            raw2.execute("DROP TRIGGER IF EXISTS fail_storage_migration")
            raw2.commit()
            raw2.close()

            run_migrations(engine)
            assert _record_count(engine, "history_review_storage_v1") == 1
        finally:
            engine.dispose()

    def test_incomplete_existing_table_fails_closed(self, tmp_path):
        db_path = tmp_path / "incomplete.db"
        _make_iteration_2a_db(db_path)

        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "CREATE TABLE history_reviews ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id INTEGER NOT NULL"
            ")"
        )
        raw.commit()
        raw.close()

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            with pytest.raises(RuntimeError, match="incomplete"):
                run_migrations(engine)
            assert _record_count(engine, "history_review_storage_v1") == 0
            with pytest.raises(RuntimeError, match="incomplete"):
                run_migrations(engine)
        finally:
            engine.dispose()

    def test_migrated_old_db_rejects_invalid_values(self, tmp_path):
        db_path = tmp_path / "constraints.db"
        _make_iteration_2a_db(db_path)

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(engine)
            with engine.begin() as conn:
                conn.execute(sa_text(
                    "INSERT INTO history_reviews ("
                    "  session_id, mode_switch_event_id, status, "
                    "  lower_bound_kind, eligible_message_count, "
                    "  eligible_char_count, source_message_count, "
                    "  source_char_count, truncated, "
                    "  reviewer_llm_profile_id_snapshot, "
                    "  reviewer_llm_profile_kind_snapshot, "
                    "  reviewer_llm_model_snapshot, "
                    "  prompt_version_snapshot, "
                    "  budget_version_snapshot, "
                    "  selection_policy_version, created_at, updated_at"
                    ") VALUES ("
                    "  1, 1, 'pending', 'session_start', 1, 5, 1, 5, 0, "
                    "  'default', 'fake', 'fake', 'history-review-v1', "
                    "  'history-review-budget-v1', "
                    "  'history-review-selection-v1', "
                    "  '2026-01-04', '2026-01-04'"
                    ")"
                ))
                review_id = conn.execute(
                    sa_text("SELECT last_insert_rowid()")
                ).scalar()
            assert review_id == 1

            invalid_updates = [
                "UPDATE history_reviews SET status = 'bogus' WHERE id = 1",
                "UPDATE history_reviews SET truncated = 2 WHERE id = 1",
                "UPDATE history_reviews SET eligible_message_count = 0 "
                "WHERE id = 1",
                "UPDATE history_reviews SET eligible_char_count = -1 "
                "WHERE id = 1",
            ]
            for statement in invalid_updates:
                with pytest.raises(Exception):
                    with engine.begin() as conn:
                        conn.execute(sa_text(statement))
        finally:
            engine.dispose()

    def test_migrated_old_db_cascade_delete(self, tmp_path):
        db_path = tmp_path / "cascade.db"
        _make_iteration_2a_db(db_path)

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(engine)
            with engine.begin() as conn:
                conn.execute(sa_text(
                    "INSERT INTO history_reviews ("
                    "  id, session_id, mode_switch_event_id, status, "
                    "  lower_bound_kind, eligible_message_count, "
                    "  eligible_char_count, source_message_count, "
                    "  source_char_count, truncated, "
                    "  reviewer_llm_profile_id_snapshot, "
                    "  reviewer_llm_profile_kind_snapshot, "
                    "  reviewer_llm_model_snapshot, "
                    "  prompt_version_snapshot, "
                    "  budget_version_snapshot, "
                    "  selection_policy_version, created_at, updated_at"
                    ") VALUES ("
                    "  1, 1, 1, 'pending', 'session_start', 1, 18, 1, 18, "
                    "  0, 'default', 'fake', 'fake', 'history-review-v1', "
                    "  'history-review-budget-v1', "
                    "  'history-review-selection-v1', "
                    "  '2026-01-04', '2026-01-04'"
                    ")"
                ))
                conn.execute(sa_text(
                    "INSERT INTO history_review_sources ("
                    "id, review_id, session_id, message_id, seq, "
                    "content_char_count, created_at) "
                    "VALUES (1, 1, 1, 1, 1, 18, '2026-01-04')"
                ))
                conn.execute(sa_text(
                    "INSERT INTO history_review_findings ("
                    "id, review_id, seq, verdict, claim_text, "
                    "source_message_id, created_at) "
                    "VALUES (1, 1, 1, 'correct', 'claim', 1, '2026-01-04')"
                ))
                conn.execute(
                    sa_text("DELETE FROM chat_sessions WHERE id = 1")
                )
                for table in (
                    "history_reviews",
                    "history_review_sources",
                    "history_review_findings",
                ):
                    count = conn.execute(
                        sa_text(f"SELECT COUNT(*) FROM {table}")
                    ).scalar()
                    assert count == 0
        finally:
            engine.dispose()


class TestIncompleteStorageTables:
    @pytest.mark.parametrize(
        "table_name",
        [
            "history_reviews",
            "history_review_sources",
            "history_review_findings",
        ],
    )
    def test_each_incomplete_same_name_table_fails_closed(
        self, tmp_path, table_name,
    ):
        db_path = tmp_path / f"incomplete-{table_name}.db"
        _make_iteration_2a_db(db_path)

        raw = sqlite3.connect(str(db_path))
        raw.execute(
            f"CREATE TABLE {table_name} ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT"
            ")"
        )
        raw.commit()
        raw.close()

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            with pytest.raises(RuntimeError, match="incomplete"):
                run_migrations(engine)
            assert _record_count(engine, "history_review_storage_v1") == 0
        finally:
            engine.dispose()

def _template_storage_table_sql(tmp_path, table_name, slug):
    template_db = tmp_path / f"template-{slug}.db"
    _make_iteration_2a_db(template_db)
    engine = create_database_engine(f"sqlite:///{template_db}")
    try:
        run_migrations(engine)
        with engine.begin() as conn:
            sql = conn.execute(
                sa_text(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type = 'table' AND name = :name"
                ),
                {"name": table_name},
            ).scalar()
        assert sql is not None
        return sql
    finally:
        engine.dispose()


def _malformed_storage_table_db(tmp_path, table_name, slug, mutation):
    target_db = tmp_path / f"malformed-{slug}.db"
    _make_iteration_2a_db(target_db)
    original = _template_storage_table_sql(tmp_path, table_name, slug)
    mutated = mutation(original)
    assert mutated != original

    raw = sqlite3.connect(str(target_db))
    raw.execute(mutated)
    raw.commit()
    raw.close()
    return target_db


def _assert_malformed_table_fails_closed(
    tmp_path, table_name, slug, mutation,
):
    db_path = _malformed_storage_table_db(
        tmp_path, table_name, slug, mutation,
    )
    engine = create_database_engine(f"sqlite:///{db_path}")
    try:
        with pytest.raises(RuntimeError):
            run_migrations(engine)
        assert _record_count(engine, "history_review_storage_v1") == 0
    finally:
        engine.dispose()


def _replace_once(source, old, new):
    assert old in source
    return source.replace(old, new, 1)



class TestMalformedStorageSchema:
    def test_composite_fk_from_column_order_wrong(self, tmp_path):
        _assert_malformed_table_fails_closed(
            tmp_path,
            "history_review_sources",
            "fk-from-order",
            lambda sql: _replace_once(
                sql,
                "FOREIGN KEY (review_id, session_id)",
                "FOREIGN KEY (session_id, review_id)",
            ),
        )

    def test_composite_fk_referenced_column_order_wrong(self, tmp_path):
        _assert_malformed_table_fails_closed(
            tmp_path,
            "history_review_findings",
            "fk-ref-order",
            lambda sql: _replace_once(
                sql,
                "REFERENCES history_review_sources "
                "(review_id, message_id)",
                "REFERENCES history_review_sources "
                "(message_id, review_id)",
            ),
        )

    def test_foreign_key_on_delete_not_cascade(self, tmp_path):
        _assert_malformed_table_fails_closed(
            tmp_path,
            "history_reviews",
            "fk-on-delete",
            lambda sql: _replace_once(
                sql,
                "ON DELETE CASCADE",
                "ON DELETE NO ACTION",
            ),
        )

    def test_required_column_wrongly_nullable(self, tmp_path):
        _assert_malformed_table_fails_closed(
            tmp_path,
            "history_review_sources",
            "nullable-required",
            lambda sql: _replace_once(
                sql,
                "session_id INTEGER NOT NULL",
                "session_id INTEGER",
            ),
        )

    def test_id_is_not_primary_key(self, tmp_path):
        _assert_malformed_table_fails_closed(
            tmp_path,
            "history_review_findings",
            "id-not-pk",
            lambda sql: _replace_once(
                sql,
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "id INTEGER",
            ),
        )

    def test_required_default_missing(self, tmp_path):
        _assert_malformed_table_fails_closed(
            tmp_path,
            "history_reviews",
            "default-missing",
            lambda sql: _replace_once(
                sql,
                "truncated BOOLEAN NOT NULL DEFAULT '0'",
                "truncated BOOLEAN NOT NULL",
            ),
        )

    def test_required_default_wrong(self, tmp_path):
        _assert_malformed_table_fails_closed(
            tmp_path,
            "history_reviews",
            "default-wrong",
            lambda sql: _replace_once(
                sql,
                "raw_output_truncated BOOLEAN NOT NULL DEFAULT '0'",
                "raw_output_truncated BOOLEAN NOT NULL DEFAULT '1'",
            ),
        )

    def test_parent_named_index_wrong_column_order(self, tmp_path):
        db_path = tmp_path / "parent-index-wrong-cols.db"
        _make_iteration_2a_db(db_path)
        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "CREATE UNIQUE INDEX uq_messages_id_session "
            "ON messages(session_id, id)"
        )
        raw.commit()
        raw.close()

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            with pytest.raises(RuntimeError, match="wrong column order"):
                run_migrations(engine)
            assert _record_count(engine, "history_review_storage_v1") == 0
        finally:
            engine.dispose()

    def test_parent_named_index_not_unique(self, tmp_path):
        db_path = tmp_path / "parent-index-not-unique.db"
        _make_iteration_2a_db(db_path)
        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "CREATE INDEX uq_messages_id_session "
            "ON messages(id, session_id)"
        )
        raw.commit()
        raw.close()

        engine = create_database_engine(f"sqlite:///{db_path}")
        try:
            with pytest.raises(RuntimeError, match="must be UNIQUE"):
                run_migrations(engine)
            assert _record_count(engine, "history_review_storage_v1") == 0
        finally:
            engine.dispose()

    def test_wrong_unique_column_order(self, tmp_path):
        _assert_malformed_table_fails_closed(
            tmp_path,
            "history_review_sources",
            "unique-order",
            lambda sql: _replace_once(
                sql,
                "UNIQUE (review_id, message_id)",
                "UNIQUE (message_id, review_id)",
            ),
        )
