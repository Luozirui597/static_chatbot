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
