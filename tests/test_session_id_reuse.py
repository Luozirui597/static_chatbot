"""Regression: session ids are durable public identifiers, not rowid slots."""

from sqlalchemy.orm import sessionmaker

from backend.database import create_database_engine, create_tables
from backend.models import ChatSession


def test_deleted_highest_session_id_is_not_reused(tmp_path):
    engine = create_database_engine(f"sqlite:///{tmp_path / 'ids.db'}")
    create_tables(engine)
    Local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    db = Local()
    try:
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
    finally:
        db.close()
        engine.dispose()
