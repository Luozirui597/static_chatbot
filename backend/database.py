"""SQLAlchemy engine, session factory, and table-creation helper."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import sessionmaker

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


def create_database_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine with SQLite foreign keys enabled.

    Every caller — production or test — must use this factory so the
    ``PRAGMA foreign_keys = ON`` listener is registered in one place.
    """
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return engine


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
