"""SQLAlchemy engine, session factory, and table-creation helper."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


def _build_default_url() -> str:
    """Return the default SQLite URL, creating the data directory."""
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir / 'chatbot.db'}"


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def _is_memory_database(database_url: str) -> bool:
    """Return ``True`` when *database_url* targets an in-memory SQLite db.

    Uses SQLAlchemy's URL parser so all recognised ``:memory:`` variants
    are detected: ``sqlite:///:memory:``, ``sqlite://``,
    ``sqlite+pysqlite:///:memory:``, etc.
    """
    parsed = make_url(database_url)
    return parsed.get_backend_name() == "sqlite" and (
        parsed.database == ":memory:" or parsed.database is None
    )


def create_database_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine with SQLite foreign keys enabled.

    Every caller — production or test — must use this factory so the
    ``PRAGMA foreign_keys = ON`` listener is registered in one place.

    In-memory databases use ``StaticPool`` so every connection sees the
    same shared data.  ``check_same_thread=False`` is kept for **all**
    SQLite URLs because FastAPI's sync routes and lifespan handlers may
    run on different threads — without it, the same connection fails
    with ``sqlite3.ProgrammingError``.
    """
    connect_args: dict = {"check_same_thread": False}
    kwargs: dict = {}

    if _is_memory_database(database_url):
        kwargs["poolclass"] = StaticPool

    engine = create_engine(
        database_url,
        connect_args=connect_args,
        **kwargs,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return engine


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def run_migrations(bind: Engine) -> None:
    """Apply pending schema migrations idempotently.

    Uses SQLAlchemy's connection/transaction machinery — no raw
    ``sqlite3`` calls.  Each migration runs inside a single
    transaction; failure rolls back and the application fails to
    start (no silent continuation).
    """
    from sqlalchemy import text as sa_text

    with bind.begin() as conn:
        # ── Migration: title_is_manual_v1 ──────────────────────────
        row = conn.execute(
            sa_text(
                "SELECT version FROM schema_migrations "
                "WHERE version = 'title_is_manual_v1'"
            )
        ).fetchone()

        if row is None:
            # Migration not yet recorded — apply it.
            cols = [
                r[1]
                for r in conn.execute(
                    sa_text("PRAGMA table_info('chat_sessions')")
                ).fetchall()
            ]
            if "title_is_manual" not in cols:
                conn.execute(
                    sa_text(
                        "ALTER TABLE chat_sessions "
                        "ADD COLUMN title_is_manual INTEGER NOT NULL DEFAULT 0"
                    )
                )

            # Idempotent backfill — safe to re-run.
            conn.execute(
                sa_text(
                    "UPDATE chat_sessions "
                    "SET title_is_manual = 1 "
                    "WHERE title != 'New Chat' AND title_is_manual = 0"
                )
            )

            # Record completion.
            conn.execute(
                sa_text(
                    "INSERT INTO schema_migrations (version) "
                    "VALUES ('title_is_manual_v1')"
                )
            )

        # ── Migration: llm_profile_v1 ───────────────────────────────
        row = conn.execute(
            sa_text(
                "SELECT version FROM schema_migrations "
                "WHERE version = 'llm_profile_v1'"
            )
        ).fetchone()

        if row is None:
            cols = [
                r[1]
                for r in conn.execute(
                    sa_text("PRAGMA table_info('chat_sessions')")
                ).fetchall()
            ]

            if "llm_profile_id" not in cols:
                conn.execute(
                    sa_text(
                        "ALTER TABLE chat_sessions "
                        "ADD COLUMN llm_profile_id VARCHAR(50) "
                        "NOT NULL DEFAULT 'default'"
                    )
                )

            if "llm_model_snapshot" not in cols:
                conn.execute(
                    sa_text(
                        "ALTER TABLE chat_sessions "
                        "ADD COLUMN llm_model_snapshot VARCHAR(255)"
                    )
                )

            # Backfill rows whose llm_profile_id is NULL, empty, or
            # whitespace-only.  SQLite's default TRIM only strips
            # spaces, so an explicit whitespace set is used: TAB(9),
            # LF(10), VT(11), FF(12), CR(13), SPACE(32).
            #
            # Rows with a non-blank profile id are left untouched —
            # even ids unknown to the current registry, so they
            # resolve to profile_unavailable.  llm_model_snapshot is
            # intentionally not touched: existing rows keep NULL so
            # they resolve to legacy_unknown.
            conn.execute(
                sa_text(
                    "UPDATE chat_sessions "
                    "SET llm_profile_id = 'default' "
                    "WHERE llm_profile_id IS NULL "
                    "   OR length("
                    "       trim("
                    "           llm_profile_id,"
                    "           char(9) || char(10) || char(11) ||"
                    "           char(12) || char(13) || char(32)"
                    "       )"
                    "   ) = 0"
                )
            )

            # Record completion.
            conn.execute(
                sa_text(
                    "INSERT INTO schema_migrations (version) "
                    "VALUES ('llm_profile_v1')"
                )
            )

        # ── Migration: message_llm_snapshot_v1 ──────────────────────
        row = conn.execute(
            sa_text(
                "SELECT version FROM schema_migrations "
                "WHERE version = 'message_llm_snapshot_v1'"
            )
        ).fetchone()

        if row is None:
            # Fail closed when the messages table itself does not
            # exist: silently recording the version would mark an
            # incomplete schema as migrated.  Normal startup is
            # unaffected because the lifespan runs create_tables()
            # before run_migrations().
            has_messages = conn.execute(
                sa_text(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'messages'"
                )
            ).fetchone()

            if has_messages is None:
                raise RuntimeError(
                    "Cannot apply message_llm_snapshot_v1: "
                    "the 'messages' table is missing"
                )

            cols = [
                r[1]
                for r in conn.execute(
                    sa_text("PRAGMA table_info('messages')")
                ).fetchall()
            ]

            # Add only the missing columns.  All are nullable with
            # no database default — existing messages keep NULL
            # snapshots; their model source is never backfilled or
            # fabricated.
            if "llm_profile_id_snapshot" not in cols:
                conn.execute(
                    sa_text(
                        "ALTER TABLE messages "
                        "ADD COLUMN llm_profile_id_snapshot VARCHAR(50)"
                    )
                )

            if "llm_profile_kind_snapshot" not in cols:
                conn.execute(
                    sa_text(
                        "ALTER TABLE messages "
                        "ADD COLUMN llm_profile_kind_snapshot VARCHAR(20)"
                    )
                )

            if "llm_model_snapshot" not in cols:
                conn.execute(
                    sa_text(
                        "ALTER TABLE messages "
                        "ADD COLUMN llm_model_snapshot VARCHAR(255)"
                    )
                )

            # Record completion LAST.  If the record insert fails the
            # version stays unrecorded and the next run re-enters the
            # block — the per-column checks make the retry converge
            # even when SQLite kept some or all of the ALTER results.
            conn.execute(
                sa_text(
                    "INSERT INTO schema_migrations (version) "
                    "VALUES ('message_llm_snapshot_v1')"
                )
            )

        _run_interaction_mode_migration(conn)
        _run_interaction_mode_constraints_migration(conn)
        _run_history_review_boundary_migration(conn)
        _run_history_review_storage_migration(conn)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

configured_database_url = config.DATABASE_URL.strip()

database_url = (
    configured_database_url
    if configured_database_url
    else _build_default_url()
)

engine = create_database_engine(database_url)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------


def create_tables(bind: Engine | None = None) -> None:
    """Create all tables via ``Base.metadata.create_all``.

    Parameters
    ----------
    bind:
        Engine to use.  When ``None`` the module-level *engine* is
        used.  Tests should pass the engine returned by
        ``create_database_engine()``.
    """
    from backend.models import Base  # noqa: PLC0415 — avoid import cycle

    target_engine = bind if bind is not None else engine
    Base.metadata.create_all(bind=target_engine)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yield a per-request database session.

    The session is closed in ``finally`` so connections always return to
    the pool even when the route raises an exception.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_interaction_mode_migration(conn) -> None:
    """Apply interaction_mode_v1 idempotently."""
    from sqlalchemy import text as sa_text

    row = conn.execute(
        sa_text(
            "SELECT version FROM schema_migrations "
            "WHERE version = 'interaction_mode_v1'"
        )
    ).fetchone()
    if row is not None:
        return

    conn.execute(
        sa_text(
            "CREATE TABLE IF NOT EXISTS mode_switch_events ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id INTEGER NOT NULL,"
            "  from_mode VARCHAR(30) NOT NULL,"
            "  to_mode VARCHAR(30) NOT NULL,"
            "  created_at DATETIME NOT NULL,"
            "  CONSTRAINT fk_mode_switch_events_session_id "
            "      FOREIGN KEY(session_id) "
            "      REFERENCES chat_sessions(id) ON DELETE CASCADE,"
            "  CONSTRAINT ck_mode_switch_events_from_mode "
            "      CHECK (from_mode IN ('receive_teaching', 'corrective')),"
            "  CONSTRAINT ck_mode_switch_events_to_mode "
            "      CHECK (to_mode IN ('receive_teaching', 'corrective'))"
            ")"
        )
    )

    session_cols = [
        r[1]
        for r in conn.execute(
            sa_text("PRAGMA table_info('chat_sessions')")
        ).fetchall()
    ]
    if "interaction_mode" not in session_cols:
        conn.execute(
            sa_text(
                "ALTER TABLE chat_sessions "
                "ADD COLUMN interaction_mode VARCHAR(30) "
                "NOT NULL DEFAULT 'receive_teaching'"
            )
        )

    message_cols = [
        r[1]
        for r in conn.execute(
            sa_text("PRAGMA table_info('messages')")
        ).fetchall()
    ]
    if "interaction_mode_snapshot" not in message_cols:
        conn.execute(
            sa_text(
                "ALTER TABLE messages "
                "ADD COLUMN interaction_mode_snapshot VARCHAR(30)"
            )
        )

    if "prompt_version_snapshot" not in message_cols:
        conn.execute(
            sa_text(
                "ALTER TABLE messages "
                "ADD COLUMN prompt_version_snapshot VARCHAR(50)"
            )
        )

    conn.execute(
        sa_text(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('interaction_mode_v1')"
        )
    )


def _run_interaction_mode_constraints_migration(conn) -> None:
    """Add idempotent trigger guards for migrated SQLite databases.

    SQLite cannot add CHECK constraints to existing tables without a
    table rebuild.  These triggers provide the same rejection semantics
    for pre-existing databases while remaining safe and idempotent.
    """
    from sqlalchemy import text as sa_text

    row = conn.execute(
        sa_text(
            "SELECT version FROM schema_migrations "
            "WHERE version = 'interaction_mode_constraints_v1'"
        )
    ).fetchone()
    if row is not None:
        return

    conn.execute(
        sa_text(
            "CREATE TRIGGER IF NOT EXISTS "
            "trg_chat_sessions_interaction_mode_insert "
            "BEFORE INSERT ON chat_sessions "
            "FOR EACH ROW "
            "WHEN NEW.interaction_mode NOT IN "
            "('receive_teaching', 'corrective') "
            "BEGIN "
            "  SELECT RAISE(ABORT, 'invalid interaction_mode'); "
            "END"
        )
    )
    conn.execute(
        sa_text(
            "CREATE TRIGGER IF NOT EXISTS "
            "trg_chat_sessions_interaction_mode_update "
            "BEFORE UPDATE OF interaction_mode ON chat_sessions "
            "FOR EACH ROW "
            "WHEN NEW.interaction_mode NOT IN "
            "('receive_teaching', 'corrective') "
            "BEGIN "
            "  SELECT RAISE(ABORT, 'invalid interaction_mode'); "
            "END"
        )
    )
    conn.execute(
        sa_text(
            "CREATE TRIGGER IF NOT EXISTS "
            "trg_messages_interaction_mode_snapshot_insert "
            "BEFORE INSERT ON messages "
            "FOR EACH ROW "
            "WHEN NEW.interaction_mode_snapshot IS NOT NULL "
            "  AND NEW.interaction_mode_snapshot NOT IN "
            "('receive_teaching', 'corrective') "
            "BEGIN "
            "  SELECT RAISE(ABORT, 'invalid interaction_mode_snapshot'); "
            "END"
        )
    )
    conn.execute(
        sa_text(
            "CREATE TRIGGER IF NOT EXISTS "
            "trg_messages_interaction_mode_snapshot_update "
            "BEFORE UPDATE OF interaction_mode_snapshot ON messages "
            "FOR EACH ROW "
            "WHEN NEW.interaction_mode_snapshot IS NOT NULL "
            "  AND NEW.interaction_mode_snapshot NOT IN "
            "('receive_teaching', 'corrective') "
            "BEGIN "
            "  SELECT RAISE(ABORT, 'invalid interaction_mode_snapshot'); "
            "END"
        )
    )

    conn.execute(
        sa_text(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('interaction_mode_constraints_v1')"
        )
    )


def _run_history_review_boundary_migration(conn) -> None:
    """Apply history_review_boundary_v1 idempotently.

    Adds the immutable history-boundary columns to
    ``mode_switch_events`` and installs a SQLite trigger that rejects
    negative ``reviewable_user_message_count`` values on databases that
    were created before the ORM-level CHECK constraint existed.
    Existing events are never backfilled.
    """
    from sqlalchemy import text as sa_text

    row = conn.execute(
        sa_text(
            "SELECT version FROM schema_migrations "
            "WHERE version = 'history_review_boundary_v1'"
        )
    ).fetchone()
    if row is not None:
        return

    cols = [
        r[1]
        for r in conn.execute(
            sa_text("PRAGMA table_info('mode_switch_events')")
        ).fetchall()
    ]

    if "history_through_message_id" not in cols:
        conn.execute(
            sa_text(
                "ALTER TABLE mode_switch_events "
                "ADD COLUMN history_through_message_id INTEGER"
            )
        )

    if "history_boundary_version" not in cols:
        conn.execute(
            sa_text(
                "ALTER TABLE mode_switch_events "
                "ADD COLUMN history_boundary_version VARCHAR(50)"
            )
        )

    if "reviewable_user_message_count" not in cols:
        conn.execute(
            sa_text(
                "ALTER TABLE mode_switch_events "
                "ADD COLUMN reviewable_user_message_count INTEGER"
            )
        )

    conn.execute(
        sa_text(
            "CREATE TRIGGER IF NOT EXISTS "
            "trg_mode_switch_events_reviewable_count_insert "
            "BEFORE INSERT ON mode_switch_events "
            "FOR EACH ROW "
            "WHEN NEW.reviewable_user_message_count IS NOT NULL "
            "  AND NEW.reviewable_user_message_count < 0 "
            "BEGIN "
            "  SELECT RAISE(ABORT, "
            "    'invalid reviewable_user_message_count'); "
            "END"
        )
    )
    conn.execute(
        sa_text(
            "CREATE TRIGGER IF NOT EXISTS "
            "trg_mode_switch_events_reviewable_count_update "
            "BEFORE UPDATE OF reviewable_user_message_count "
            "ON mode_switch_events "
            "FOR EACH ROW "
            "WHEN NEW.reviewable_user_message_count IS NOT NULL "
            "  AND NEW.reviewable_user_message_count < 0 "
            "BEGIN "
            "  SELECT RAISE(ABORT, "
            "    'invalid reviewable_user_message_count'); "
            "END"
        )
    )

    conn.execute(
        sa_text(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('history_review_boundary_v1')"
        )
    )


_HISTORY_REVIEW_STORAGE_TABLES = {
    "history_reviews": {
        "columns": (
            ("id", "INTEGER", False, True, None),
            ("session_id", "INTEGER", True, False, None),
            ("mode_switch_event_id", "INTEGER", True, False, None),
            ("status", "TEXT", True, False, None),
            ("lower_bound_kind", "TEXT", True, False, None),
            ("lower_bound_message_id", "INTEGER", False, False, None),
            ("upper_bound_message_id", "INTEGER", False, False, None),
            ("eligible_from_message_id", "INTEGER", False, False, None),
            ("eligible_through_message_id", "INTEGER", False, False, None),
            ("eligible_message_count", "INTEGER", True, False, None),
            ("eligible_char_count", "INTEGER", True, False, None),
            ("source_from_message_id", "INTEGER", False, False, None),
            ("source_through_message_id", "INTEGER", False, False, None),
            ("source_message_count", "INTEGER", True, False, None),
            ("source_char_count", "INTEGER", True, False, None),
            ("truncated", "NUMERIC", True, False, 0),
            ("reviewer_llm_profile_id_snapshot", "TEXT", True, False, None),
            ("reviewer_llm_profile_kind_snapshot", "TEXT", True, False, None),
            ("reviewer_llm_model_snapshot", "TEXT", True, False, None),
            ("prompt_version_snapshot", "TEXT", True, False, None),
            ("budget_version_snapshot", "TEXT", True, False, None),
            ("selection_policy_version", "TEXT", True, False, None),
            ("summary", "TEXT", False, False, None),
            ("coverage_note", "TEXT", False, False, None),
            ("raw_output", "TEXT", False, False, None),
            ("raw_output_truncated", "NUMERIC", True, False, 0),
            ("error_code", "TEXT", False, False, None),
            ("error_message", "TEXT", False, False, None),
            ("attempt_count", "INTEGER", True, False, 1),
            ("remote_history_acknowledged", "NUMERIC", True, False, 0),
            ("remote_history_acknowledged_at", "NUMERIC", False, False, None),
            ("remote_history_ack_message_count", "INTEGER", False, False, None),
            ("created_at", "NUMERIC", True, False, None),
            ("started_at", "NUMERIC", False, False, None),
            ("completed_at", "NUMERIC", False, False, None),
            ("updated_at", "NUMERIC", True, False, None),
        ),
        "uniques": (
            ("session_id", "mode_switch_event_id"),
            ("id", "session_id"),
        ),
        "foreign_keys": (
            (
                ("mode_switch_event_id", "session_id"),
                "mode_switch_events",
                ("id", "session_id"),
                "CASCADE",
            ),
            (
                ("session_id",),
                "chat_sessions",
                ("id",),
                "CASCADE",
            ),
        ),
    },
    "history_review_sources": {
        "columns": (
            ("id", "INTEGER", False, True, None),
            ("review_id", "INTEGER", True, False, None),
            ("session_id", "INTEGER", True, False, None),
            ("message_id", "INTEGER", True, False, None),
            ("seq", "INTEGER", True, False, None),
            ("content_char_count", "INTEGER", True, False, None),
            ("created_at", "NUMERIC", True, False, None),
        ),
        "uniques": (
            ("review_id", "message_id"),
            ("review_id", "seq"),
        ),
        "foreign_keys": (
            (
                ("review_id", "session_id"),
                "history_reviews",
                ("id", "session_id"),
                "CASCADE",
            ),
            (
                ("message_id", "session_id"),
                "messages",
                ("id", "session_id"),
                "CASCADE",
            ),
        ),
    },
    "history_review_findings": {
        "columns": (
            ("id", "INTEGER", False, True, None),
            ("review_id", "INTEGER", True, False, None),
            ("seq", "INTEGER", True, False, None),
            ("verdict", "TEXT", True, False, None),
            ("claim_text", "TEXT", True, False, None),
            ("correction_text", "TEXT", False, False, None),
            ("explanation_text", "TEXT", False, False, None),
            ("source_message_id", "INTEGER", True, False, None),
            ("created_at", "NUMERIC", True, False, None),
        ),
        "uniques": (
            ("review_id", "seq"),
        ),
        "foreign_keys": (
            (
                ("review_id", "source_message_id"),
                "history_review_sources",
                ("review_id", "message_id"),
                "CASCADE",
            ),
        ),
    },
}

_HISTORY_REVIEW_PARENT_UNIQUE_INDEXES = (
    ("messages", "uq_messages_id_session", ("id", "session_id")),
    (
        "mode_switch_events",
        "uq_mode_switch_events_id_session",
        ("id", "session_id"),
    ),
)


def _sqlite_table_exists(conn, table: str) -> bool:
    from sqlalchemy import text as sa_text

    row = conn.execute(
        sa_text(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = :name"
        ),
        {"name": table},
    ).fetchone()
    return row is not None


def _sqlite_table_column_metadata(conn, table: str) -> list[dict]:
    from sqlalchemy import text as sa_text

    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2] or "",
            "notnull": bool(row[3]),
            "dflt_value": row[4],
            "pk": bool(row[5]),
        }
        for row in conn.execute(
            sa_text(f"PRAGMA table_info('{table}')")
        ).fetchall()
    ]


def _sqlite_type_affinity(declared_type: str) -> str:
    """Return SQLite's documented type affinity for *declared_type*."""
    text = (declared_type or "").upper()
    if "INT" in text:
        return "INTEGER"
    if any(token in text for token in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if "BLOB" in text or text == "":
        return "BLOB"
    if any(token in text for token in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


def _sqlite_indexes(conn, table: str) -> list[dict]:
    from sqlalchemy import text as sa_text

    indexes: list[dict] = []
    for row in conn.execute(
        sa_text(f"PRAGMA index_list('{table}')")
    ).fetchall():
        columns = [
            info[2]
            for info in conn.execute(
                sa_text(f"PRAGMA index_info('{row[1]}')")
            ).fetchall()
        ]
        indexes.append({
            "name": row[1],
            "unique": bool(row[2]),
            "origin": row[3],
            "partial": bool(row[4]),
            "columns": tuple(columns),
        })
    return indexes


def _sqlite_unique_column_tuples(
    conn, table: str,
) -> list[tuple[str, ...]]:
    return [
        index["columns"]
        for index in _sqlite_indexes(conn, table)
        if index["unique"] and not index["partial"]
    ]


def _sqlite_foreign_keys(
    conn, table: str,
) -> list[tuple[tuple[str, ...], str, tuple[str, ...], str]]:
    from sqlalchemy import text as sa_text

    grouped: dict[int, dict] = {}
    rows = conn.execute(
        sa_text(f"PRAGMA foreign_key_list('{table}')")
    ).fetchall()
    for row in rows:
        fk_id = row[0]
        grouped.setdefault(fk_id, {
            "table": row[2],
            "from": [],
            "to": [],
            "seq": [],
            "on_delete": row[6],
        })
        grouped[fk_id]["from"].append(row[3])
        grouped[fk_id]["to"].append(row[4])
        grouped[fk_id]["seq"].append(row[1])

    result: list[tuple[tuple[str, ...], str, tuple[str, ...], str]] = []
    for fk in grouped.values():
        order = sorted(
            range(len(fk["seq"])), key=lambda i: fk["seq"][i]
        )
        result.append((
            tuple(fk["from"][i] for i in order),
            fk["table"],
            tuple(fk["to"][i] for i in order),
            str(fk["on_delete"]).upper(),
        ))
    return result


def _sqlite_default_matches(value, expected: int) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    while len(text) >= 2 and text[0] == "(" and text[-1] == ")":
        text = text[1:-1].strip()
    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in ("'", '"')
    ):
        text = text[1:-1].strip()
    try:
        return int(text) == expected
    except ValueError:
        try:
            return float(text) == float(expected)
        except ValueError:
            return False


def _verify_history_review_parent_unique_indexes(conn) -> None:
    for table, index_name, expected_columns in (
        _HISTORY_REVIEW_PARENT_UNIQUE_INDEXES
    ):
        indexes = _sqlite_indexes(conn, table)
        matching = [
            index for index in indexes if index["name"] == index_name
        ]
        if not matching:
            raise RuntimeError(
                f"history_review_storage_v1: missing required index "
                f"'{index_name}' on table '{table}'"
            )
        index = matching[0]
        if not index["unique"]:
            raise RuntimeError(
                f"history_review_storage_v1: index '{index_name}' on "
                f"'{table}' must be UNIQUE"
            )
        if index["partial"]:
            raise RuntimeError(
                f"history_review_storage_v1: index '{index_name}' on "
                f"'{table}' must not be partial"
            )
        if tuple(index["columns"]) != tuple(expected_columns):
            raise RuntimeError(
                f"history_review_storage_v1: index '{index_name}' on "
                f"'{table}' has wrong column order: "
                f"{tuple(index['columns'])} != {tuple(expected_columns)}"
            )


def _verify_history_review_storage_table(conn, table: str) -> None:
    expected = _HISTORY_REVIEW_STORAGE_TABLES[table]
    if not _sqlite_table_exists(conn, table):
        raise RuntimeError(
            f"history_review_storage_v1: table '{table}' is missing"
        )

    actual_columns = {
        column["name"]: column
        for column in _sqlite_table_column_metadata(conn, table)
    }

    for (
        name,
        expected_affinity,
        expected_notnull,
        expected_pk,
        expected_default,
    ) in expected["columns"]:
        actual = actual_columns.get(name)
        if actual is None:
            raise RuntimeError(
                f"history_review_storage_v1: table '{table}' is "
                f"incomplete; missing column '{name}'"
            )

        actual_affinity = _sqlite_type_affinity(actual["type"])
        if actual_affinity != expected_affinity:
            raise RuntimeError(
                f"history_review_storage_v1: table '{table}' column "
                f"'{name}' has affinity {actual_affinity}, expected "
                f"{expected_affinity}"
            )

        if expected_pk:
            if not actual["pk"]:
                raise RuntimeError(
                    f"history_review_storage_v1: table '{table}' column "
                    f"'{name}' must be the primary key"
                )
        else:
            if actual["pk"]:
                raise RuntimeError(
                    f"history_review_storage_v1: table '{table}' column "
                    f"'{name}' must not be part of the primary key"
                )
            if expected_notnull and not actual["notnull"]:
                raise RuntimeError(
                    f"history_review_storage_v1: table '{table}' column "
                    f"'{name}' must be NOT NULL"
                )
            if not expected_notnull and actual["notnull"]:
                raise RuntimeError(
                    f"history_review_storage_v1: table '{table}' column "
                    f"'{name}' must be nullable"
                )

        if expected_default is not None:
            if not _sqlite_default_matches(
                actual["dflt_value"], expected_default
            ):
                raise RuntimeError(
                    f"history_review_storage_v1: table '{table}' column "
                    f"'{name}' has wrong default "
                    f"{actual['dflt_value']}; expected {expected_default}"
                )

    actual_uniques = [
        tuple(columns)
        for columns in _sqlite_unique_column_tuples(conn, table)
    ]
    for expected_columns in expected["uniques"]:
        if tuple(expected_columns) not in actual_uniques:
            raise RuntimeError(
                f"history_review_storage_v1: table '{table}' is "
                f"incomplete; missing unique constraint "
                f"{tuple(expected_columns)}"
            )

    actual_foreign_keys = _sqlite_foreign_keys(conn, table)
    for (
        from_columns,
        referred_table,
        referred_columns,
        on_delete,
    ) in expected["foreign_keys"]:
        expected_fk = (
            tuple(from_columns),
            referred_table,
            tuple(referred_columns),
            on_delete.upper(),
        )
        if expected_fk not in actual_foreign_keys:
            raise RuntimeError(
                f"history_review_storage_v1: table '{table}' is "
                f"incomplete; missing foreign key {expected_fk}"
            )


def _create_history_review_storage_table(conn, table: str) -> None:
    from sqlalchemy import text as sa_text

    if table == "history_reviews":
        conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS history_reviews (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id INTEGER NOT NULL,
              mode_switch_event_id INTEGER NOT NULL,
              status VARCHAR(20) NOT NULL,
              lower_bound_kind VARCHAR(30) NOT NULL,
              lower_bound_message_id INTEGER,
              upper_bound_message_id INTEGER,
              eligible_from_message_id INTEGER,
              eligible_through_message_id INTEGER,
              eligible_message_count INTEGER NOT NULL,
              eligible_char_count INTEGER NOT NULL,
              source_from_message_id INTEGER,
              source_through_message_id INTEGER,
              source_message_count INTEGER NOT NULL,
              source_char_count INTEGER NOT NULL,
              truncated BOOLEAN NOT NULL DEFAULT '0',
              reviewer_llm_profile_id_snapshot VARCHAR(50) NOT NULL,
              reviewer_llm_profile_kind_snapshot VARCHAR(20) NOT NULL,
              reviewer_llm_model_snapshot VARCHAR(255) NOT NULL,
              prompt_version_snapshot VARCHAR(50) NOT NULL,
              budget_version_snapshot VARCHAR(50) NOT NULL,
              selection_policy_version VARCHAR(50) NOT NULL,
              summary TEXT,
              coverage_note TEXT,
              raw_output TEXT,
              raw_output_truncated BOOLEAN NOT NULL DEFAULT '0',
              error_code VARCHAR(50),
              error_message TEXT,
              attempt_count INTEGER NOT NULL DEFAULT '1',
              remote_history_acknowledged BOOLEAN NOT NULL DEFAULT '0',
              remote_history_acknowledged_at DATETIME,
              remote_history_ack_message_count INTEGER,
              created_at DATETIME NOT NULL,
              started_at DATETIME,
              completed_at DATETIME,
              updated_at DATETIME NOT NULL,
              CONSTRAINT uq_history_reviews_session_event
                UNIQUE (session_id, mode_switch_event_id),
              CONSTRAINT uq_history_reviews_id_session
                UNIQUE (id, session_id),
              CONSTRAINT fk_history_reviews_event_session
                FOREIGN KEY (mode_switch_event_id, session_id)
                REFERENCES mode_switch_events (id, session_id)
                ON DELETE CASCADE,
              CONSTRAINT fk_history_reviews_session
                FOREIGN KEY (session_id)
                REFERENCES chat_sessions (id)
                ON DELETE CASCADE,
              CONSTRAINT ck_history_reviews_status
                CHECK (status IN
                  ('pending', 'running', 'completed', 'failed')),
              CONSTRAINT ck_history_reviews_lower_bound_kind
                CHECK (lower_bound_kind IN
                  ('session_start', 'corrective_switch')),
              CONSTRAINT ck_history_reviews_reviewer_kind
                CHECK (reviewer_llm_profile_kind_snapshot IN
                  ('fake', 'api', 'local')),
              CONSTRAINT ck_history_reviews_eligible_count
                CHECK (eligible_message_count >= 1),
              CONSTRAINT ck_history_reviews_source_count
                CHECK (source_message_count >= 1),
              CONSTRAINT ck_history_reviews_count_order
                CHECK (eligible_message_count >= source_message_count),
              CONSTRAINT ck_history_reviews_eligible_chars
                CHECK (eligible_char_count >= 0),
              CONSTRAINT ck_history_reviews_source_chars
                CHECK (source_char_count >= 0),
              CONSTRAINT ck_history_reviews_char_order
                CHECK (eligible_char_count >= source_char_count),
              CONSTRAINT ck_history_reviews_attempt_count
                CHECK (attempt_count >= 1),
              CONSTRAINT ck_history_reviews_ack_count
                CHECK (remote_history_ack_message_count IS NULL
                  OR remote_history_ack_message_count >= 0),
              CONSTRAINT ck_history_reviews_truncated
                CHECK (truncated IN (0, 1)),
              CONSTRAINT ck_history_reviews_raw_output_truncated
                CHECK (raw_output_truncated IN (0, 1)),
              CONSTRAINT ck_history_reviews_remote_ack
                CHECK (remote_history_acknowledged IN (0, 1))
            )
        """))
        return

    if table == "history_review_sources":
        conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS history_review_sources (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              review_id INTEGER NOT NULL,
              session_id INTEGER NOT NULL,
              message_id INTEGER NOT NULL,
              seq INTEGER NOT NULL,
              content_char_count INTEGER NOT NULL,
              created_at DATETIME NOT NULL,
              CONSTRAINT uq_history_review_sources_review_message
                UNIQUE (review_id, message_id),
              CONSTRAINT uq_history_review_sources_review_seq
                UNIQUE (review_id, seq),
              CONSTRAINT fk_history_review_sources_review_session
                FOREIGN KEY (review_id, session_id)
                REFERENCES history_reviews (id, session_id)
                ON DELETE CASCADE,
              CONSTRAINT fk_history_review_sources_message_session
                FOREIGN KEY (message_id, session_id)
                REFERENCES messages (id, session_id)
                ON DELETE CASCADE,
              CONSTRAINT ck_history_review_sources_seq
                CHECK (seq >= 1),
              CONSTRAINT ck_history_review_sources_char_count
                CHECK (content_char_count >= 0)
            )
        """))
        return

    if table == "history_review_findings":
        conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS history_review_findings (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              review_id INTEGER NOT NULL,
              seq INTEGER NOT NULL,
              verdict VARCHAR(20) NOT NULL,
              claim_text TEXT NOT NULL,
              correction_text TEXT,
              explanation_text TEXT,
              source_message_id INTEGER NOT NULL,
              created_at DATETIME NOT NULL,
              CONSTRAINT uq_history_review_findings_review_seq
                UNIQUE (review_id, seq),
              CONSTRAINT fk_history_review_findings_review_source
                FOREIGN KEY (review_id, source_message_id)
                REFERENCES history_review_sources (review_id, message_id)
                ON DELETE CASCADE,
              CONSTRAINT ck_history_review_findings_seq
                CHECK (seq >= 1),
              CONSTRAINT ck_history_review_findings_verdict
                CHECK (verdict IN
                  ('correct', 'incorrect', 'uncertain', 'not_a_claim')),
              CONSTRAINT ck_history_review_findings_claim_not_blank
                CHECK (length(trim(claim_text,
                  char(9) || char(10) || char(13) || char(32))) > 0)
            )
        """))
        return

    raise ValueError(f"Unknown history review table: {table}")


def _run_history_review_storage_migration(conn) -> None:
    """Apply history_review_storage_v1 idempotently.

    Creates the history-review persistence tables and the parent unique
    indexes required by their composite foreign keys.  Existing table
    definitions are verified instead of silently accepted when
    incomplete.
    """
    from sqlalchemy import text as sa_text

    row = conn.execute(
        sa_text(
            "SELECT version FROM schema_migrations "
            "WHERE version = 'history_review_storage_v1'"
        )
    ).fetchone()
    if row is not None:
        return

    conn.execute(
        sa_text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_id_session "
            "ON messages(id, session_id)"
        )
    )
    conn.execute(
        sa_text(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "uq_mode_switch_events_id_session "
            "ON mode_switch_events(id, session_id)"
        )
    )
    _verify_history_review_parent_unique_indexes(conn)

    for table in _HISTORY_REVIEW_STORAGE_TABLES:
        if _sqlite_table_exists(conn, table):
            _verify_history_review_storage_table(conn, table)
        else:
            _create_history_review_storage_table(conn, table)

    for table in _HISTORY_REVIEW_STORAGE_TABLES:
        _verify_history_review_storage_table(conn, table)

    conn.execute(sa_text(
        "CREATE INDEX IF NOT EXISTS "
        "ix_history_reviews_session_created "
        "ON history_reviews(session_id, created_at)"
    ))
    conn.execute(sa_text(
        "CREATE INDEX IF NOT EXISTS "
        "ix_history_reviews_mode_switch_event_id "
        "ON history_reviews(mode_switch_event_id)"
    ))
    conn.execute(sa_text(
        "CREATE INDEX IF NOT EXISTS "
        "ix_history_reviews_status "
        "ON history_reviews(status)"
    ))
    conn.execute(sa_text(
        "CREATE INDEX IF NOT EXISTS "
        "ix_history_review_sources_review_seq "
        "ON history_review_sources(review_id, seq)"
    ))
    conn.execute(sa_text(
        "CREATE INDEX IF NOT EXISTS "
        "ix_history_review_findings_review_seq "
        "ON history_review_findings(review_id, seq)"
    ))

    conn.execute(sa_text("""
        CREATE TRIGGER IF NOT EXISTS trg_history_reviews_checks_insert
        BEFORE INSERT ON history_reviews
        FOR EACH ROW
        WHEN NEW.status NOT IN
               ('pending', 'running', 'completed', 'failed')
          OR NEW.lower_bound_kind NOT IN
               ('session_start', 'corrective_switch')
          OR NEW.reviewer_llm_profile_kind_snapshot NOT IN
               ('fake', 'api', 'local')
          OR NEW.eligible_message_count < 1
          OR NEW.source_message_count < 1
          OR NEW.eligible_message_count < NEW.source_message_count
          OR NEW.eligible_char_count < 0
          OR NEW.source_char_count < 0
          OR NEW.eligible_char_count < NEW.source_char_count
          OR NEW.attempt_count < 1
          OR (NEW.remote_history_ack_message_count IS NOT NULL
              AND NEW.remote_history_ack_message_count < 0)
          OR NEW.truncated NOT IN (0, 1)
          OR NEW.raw_output_truncated NOT IN (0, 1)
          OR NEW.remote_history_acknowledged NOT IN (0, 1)
        BEGIN
          SELECT RAISE(ABORT, 'invalid history_reviews row');
        END
    """))
    conn.execute(sa_text("""
        CREATE TRIGGER IF NOT EXISTS trg_history_reviews_checks_update
        BEFORE UPDATE ON history_reviews
        FOR EACH ROW
        WHEN NEW.status NOT IN
               ('pending', 'running', 'completed', 'failed')
          OR NEW.lower_bound_kind NOT IN
               ('session_start', 'corrective_switch')
          OR NEW.reviewer_llm_profile_kind_snapshot NOT IN
               ('fake', 'api', 'local')
          OR NEW.eligible_message_count < 1
          OR NEW.source_message_count < 1
          OR NEW.eligible_message_count < NEW.source_message_count
          OR NEW.eligible_char_count < 0
          OR NEW.source_char_count < 0
          OR NEW.eligible_char_count < NEW.source_char_count
          OR NEW.attempt_count < 1
          OR (NEW.remote_history_ack_message_count IS NOT NULL
              AND NEW.remote_history_ack_message_count < 0)
          OR NEW.truncated NOT IN (0, 1)
          OR NEW.raw_output_truncated NOT IN (0, 1)
          OR NEW.remote_history_acknowledged NOT IN (0, 1)
        BEGIN
          SELECT RAISE(ABORT, 'invalid history_reviews row');
        END
    """))
    conn.execute(sa_text("""
        CREATE TRIGGER IF NOT EXISTS trg_history_review_sources_checks_insert
        BEFORE INSERT ON history_review_sources
        FOR EACH ROW
        WHEN NEW.seq < 1 OR NEW.content_char_count < 0
        BEGIN
          SELECT RAISE(ABORT, 'invalid history_review_sources row');
        END
    """))
    conn.execute(sa_text("""
        CREATE TRIGGER IF NOT EXISTS trg_history_review_sources_checks_update
        BEFORE UPDATE ON history_review_sources
        FOR EACH ROW
        WHEN NEW.seq < 1 OR NEW.content_char_count < 0
        BEGIN
          SELECT RAISE(ABORT, 'invalid history_review_sources row');
        END
    """))
    conn.execute(sa_text("""
        CREATE TRIGGER IF NOT EXISTS trg_history_review_findings_checks_insert
        BEFORE INSERT ON history_review_findings
        FOR EACH ROW
        WHEN NEW.seq < 1
          OR NEW.verdict NOT IN
               ('correct', 'incorrect', 'uncertain', 'not_a_claim')
          OR length(trim(NEW.claim_text,
                 char(9) || char(10) || char(13) || char(32))) = 0
        BEGIN
          SELECT RAISE(ABORT, 'invalid history_review_findings row');
        END
    """))
    conn.execute(sa_text("""
        CREATE TRIGGER IF NOT EXISTS trg_history_review_findings_checks_update
        BEFORE UPDATE ON history_review_findings
        FOR EACH ROW
        WHEN NEW.seq < 1
          OR NEW.verdict NOT IN
               ('correct', 'incorrect', 'uncertain', 'not_a_claim')
          OR length(trim(NEW.claim_text,
                 char(9) || char(10) || char(13) || char(32))) = 0
        BEGIN
          SELECT RAISE(ABORT, 'invalid history_review_findings row');
        END
    """))

    conn.execute(sa_text(
        "INSERT INTO schema_migrations (version) "
        "VALUES ('history_review_storage_v1')"
    ))
