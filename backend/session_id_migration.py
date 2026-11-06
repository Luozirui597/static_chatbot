"""Durable SQLite session-id migration and startup schema preflight.

SQLite's plain ``INTEGER PRIMARY KEY`` reuses the largest deleted rowid.
Session ids are externally visible in URLs, so the durable schema must use
``AUTOINCREMENT``.  ``create_all`` cannot add that to an existing table; the
rebuild here is deliberately a separate, explicitly-invoked maintenance step.

Nothing in this module runs automatically against a real database.  The
application startup path only performs a read-only preflight; a legacy table
makes startup fail closed until the maintenance command is run.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from sqlalchemy import MetaData
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateTable

SESSION_ID_SCHEMA_VERSION = "chat_session_autoincrement_v1"
MODE_SWITCH_CHECK_VERSION = "mode_switch_reviewable_check_v1"

SCHEMA_NEW = "new"
SCHEMA_CURRENT = "current"
SCHEMA_MAINTENANCE = "maintenance"
SCHEMA_UNKNOWN = "unknown"

APP_TABLES = {
    "chat_sessions",
    "messages",
    "mode_switch_events",
    "history_reviews",
    "history_review_sources",
    "history_review_findings",
    "schema_migrations",
}
REQUIRED_TABLES = (
    "chat_sessions",
    "messages",
    "schema_migrations",
    "mode_switch_events",
    "history_reviews",
    "history_review_sources",
    "history_review_findings",
)
OPTIONAL_TABLES = ("session_teaching_strategies", "message_teaching_strategies")
COMMIT_NOT_COMMITTED = "not_committed"
COMMIT_UNKNOWN = "commit_unknown"
COMMIT_COMMITTED = "committed"
ROLLBACK_NOT_ATTEMPTED = "not_attempted"
ROLLBACK_SUCCEEDED = "rollback_succeeded"
ROLLBACK_FAILED = "rollback_failed"
ROLLBACK_UNKNOWN = "rollback_unknown"

_REPO_ROOT = Path(__file__).resolve().parent.parent


class SessionIdMigrationError(RuntimeError):
    """A migration failed; transaction state is explicit and safe to log."""

    def __init__(
        self,
        stage: str,
        *,
        committed: bool | None,
        cleanup_error: str | None = None,
        commit_state: str | None = None,
        rollback_state: str = ROLLBACK_NOT_ATTEMPTED,
    ) -> None:
        self.stage = stage
        self.committed = committed
        self.commit_state = commit_state or (
            COMMIT_COMMITTED if committed is True
            else COMMIT_UNKNOWN if committed is None
            else COMMIT_NOT_COMMITTED
        )
        self.rollback_state = rollback_state
        self.cleanup_error = cleanup_error
        suffix = f"; cleanup failed: {cleanup_error}" if cleanup_error else ""
        super().__init__(
            f"session id migration failed at {stage}; "
            f"commit_state={self.commit_state!r}; "
            f"rollback_state={self.rollback_state!r}{suffix}"
        )


class SessionIdMigrationRequired(SessionIdMigrationError):
    """The database needs the explicit maintenance rebuild before startup."""

    def __str__(self) -> str:
        return (
            "database requires explicit maintenance for a known legacy "
            "session/mode schema; stop the service, make a consistent backup "
            "outside the repository, then run: "
            "python -m backend.session_id_migration "
            "--database-url <database-url> --backup-path <backup-path>; "
            "the service must not be started against the legacy schema"
        )


class SessionSchemaUnknown(SessionIdMigrationError):
    """The database schema could not be identified safely."""

    def __str__(self) -> str:
        return (
            "database schema is not recognized as a canonical project schema; "
            "stop and restore/inspect a backup before any write"
        )


class BackupError(RuntimeError):
    """A consistent backup could not be made or verified."""


@dataclass(frozen=True)
class MigrationResult:
    migrated: bool
    backup_path: Path | None
    schema_version: str


def _sqlite_path(database_url: str) -> Path | None:
    parsed = make_url(database_url)
    if parsed.get_backend_name() != "sqlite":
        raise SessionSchemaUnknown(
            "database schema", committed=False
        )
    if parsed.database in (None, "", ":memory:"):
        return None
    path = Path(parsed.database).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _normalise_sql(sql: str) -> str:
    return " ".join(sql.replace(chr(34), "").strip().split()).rstrip(";")


def _expected_table_ddl(table_name: str, *, autoincrement: bool) -> str:
    from backend.models import Base

    table = Base.metadata.tables[table_name]
    if table_name == "chat_sessions":
        table = table.to_metadata(MetaData(), name=table_name)
        table.dialect_options["sqlite"]["autoincrement"] = autoincrement
    ddl = str(CreateTable(table).compile(dialect=sqlite_dialect.dialect()))
    return _normalise_sql(ddl)



@dataclass(frozen=True)
class _TableShape:
    columns: dict[str, tuple[str, bool, str | None, int]]
    uniques: frozenset[tuple[str, ...]]
    foreign_keys: frozenset[tuple[tuple[str, str, str, str], ...]]
    checks: frozenset[str]
    autoincrement: bool
    valid: bool = True


def _normalise_default(value: object) -> str | None:
    if value is None:
        return None
    return _normalise_sql(str(value)).replace("'", "").lower()


def _extract_check_expressions(sql: str) -> set[str]:
    expressions: set[str] = set()
    for match in re.finditer(r"\bCHECK\s*\(", sql, flags=re.IGNORECASE):
        start = match.end()
        depth = 1
        index = start
        while index < len(sql) and depth:
            char = sql[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        if depth:
            return {"__invalid_check__"}
        expression = sql[start:index - 1]
        expressions.add(
            re.sub(r"\s+", "", expression.replace('"', "").replace("'", "")).lower()
        )
    return expressions


def _sql_without_literals(sql: str) -> str:
    """Remove comments and replace string/identifier literals."""
    output: list[str] = []
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        nxt = sql[index + 1] if index + 1 < length else ""
        if char == "-" and nxt == "-":
            index += 2
            while index < length and sql[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and nxt == "*":
            end = sql.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            while index < length:
                if sql[index] == quote:
                    if index + 1 < length and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            output.append("IDENT" if quote != "'" else "STR")
            continue
        if char == "[":
            end = sql.find("]", index + 1)
            index = length if end < 0 else end + 1
            output.append("IDENT")
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _has_effective_autoincrement(sql: str) -> bool:
    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|[(),]",
        _sql_without_literals(sql).upper(),
    )
    for index, token in enumerate(tokens):
        if token != "PRIMARY" or index + 1 >= len(tokens):
            continue
        if tokens[index + 1] != "KEY":
            continue
        cursor = index + 2
        if cursor < len(tokens) and tokens[cursor] == "AUTOINCREMENT":
            return True
        if cursor >= len(tokens) or tokens[cursor] != "(":
            continue
        cursor += 1
        # Skip the first primary-key column name; AUTOINCREMENT, when
        # present, follows that name and is not a standalone column name.
        if cursor < len(tokens) and re.fullmatch(r"[A-Z_][A-Z0-9_]*", tokens[cursor]):
            cursor += 1
        if cursor < len(tokens) and tokens[cursor] == "AUTOINCREMENT":
            return True
    return False


def _table_shape(conn: sqlite3.Connection, table_name: str) -> _TableShape | None:
    table_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    if table_row is None:
        return None
    table_sql = table_row[0] or ""
    columns = {
        str(name): (
            str(type_name).upper(),
            bool(not_null),
            _normalise_default(default),
            int(primary_key),
        )
        for _cid, name, type_name, not_null, default, primary_key
        in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    }
    valid = True
    uniques: set[tuple[str, ...]] = set()
    indexes = conn.execute(f'PRAGMA index_list("{table_name}")').fetchall()
    for _seq, index_name, unique, _origin, partial in indexes:
        if not unique:
            continue
        if partial:
            valid = False
            continue
        info = conn.execute(
            f'PRAGMA index_xinfo("{index_name}")'
        ).fetchall()
        key_columns = []
        for _seqno, cid, name, desc, collation, key in info:
            if not key:
                continue
            if cid < 0 or name is None or desc or collation.upper() != "BINARY":
                valid = False
                break
            key_columns.append(str(name))
        if valid:
            uniques.add(tuple(key_columns))
    foreign_key_rows = conn.execute(
        f'PRAGMA foreign_key_list("{table_name}")'
    ).fetchall()
    grouped: dict[int, list[tuple[int, tuple[str, str, str, str]]]] = {}
    for row in foreign_key_rows:
        fk_id, seq, parent, local, parent_col, on_update, on_delete = row[:7]
        grouped.setdefault(int(fk_id), []).append(
            (
                int(seq),
                (
                    str(local),
                    str(parent),
                    str(parent_col),
                    str(on_update),
                    str(on_delete),
                ),
            )
        )
    foreign_keys = frozenset(
        tuple(part for _seq, part in sorted(parts))
        for parts in grouped.values()
    )
    return _TableShape(
        columns=columns,
        uniques=frozenset(uniques),
        foreign_keys=foreign_keys,
        checks=frozenset(_extract_check_expressions(table_sql)),
        autoincrement=_has_effective_autoincrement(table_sql),
        valid=valid,
    )


def _expected_shapes() -> dict[str, _TableShape]:
    from backend.models import Base

    conn = sqlite3.connect(":memory:")
    try:
        for table in Base.metadata.sorted_tables:
            conn.execute(
                str(CreateTable(table).compile(dialect=sqlite_dialect.dialect()))
            )
        return {
            name: _table_shape(conn, name)
            for name in REQUIRED_TABLES + OPTIONAL_TABLES
            if _table_shape(conn, name) is not None
        }
    finally:
        conn.close()


def _check_versions(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }


def _classify_connection(conn: sqlite3.Connection) -> tuple[str, set[str]]:
    expected = _expected_shapes()
    actual: dict[str, _TableShape | None] = {}
    for name in REQUIRED_TABLES + tuple(
        name for name in OPTIONAL_TABLES if _table_shape(conn, name) is not None
    ):
        shape = _table_shape(conn, name)
        if shape is None or not shape.valid:
            return SCHEMA_UNKNOWN, set()
        actual[name] = shape
    versions = _check_versions(conn)
    if "teaching_strategies_v1" in versions and any(name not in actual for name in OPTIONAL_TABLES):
        return SCHEMA_UNKNOWN, set()
    needs: set[str] = set()
    for name in actual:
        exp = expected[name]
        got = actual[name]
        if got.columns != exp.columns:
            return SCHEMA_UNKNOWN, set()
        if got.uniques != exp.uniques:
            return SCHEMA_UNKNOWN, set()
        if got.foreign_keys != exp.foreign_keys:
            return SCHEMA_UNKNOWN, set()
        if name == "chat_sessions":
            if got.autoincrement != exp.autoincrement:
                if not got.autoincrement:
                    needs.add("autoincrement")
                else:
                    return SCHEMA_UNKNOWN, set()
        elif got.autoincrement != exp.autoincrement:
            return SCHEMA_UNKNOWN, set()
        if got.checks != exp.checks:
            missing = exp.checks - got.checks
            extra = got.checks - exp.checks
            if extra:
                return SCHEMA_UNKNOWN, set()
            if (
                name == "mode_switch_events"
                and len(missing) == 1
                and "reviewable_user_message_count" in next(iter(missing))
            ):
                needs.add("mode_check")
            else:
                return SCHEMA_UNKNOWN, set()
    if "autoincrement" in needs and SESSION_ID_SCHEMA_VERSION in versions:
        return SCHEMA_UNKNOWN, set()
    if "mode_check" in needs and MODE_SWITCH_CHECK_VERSION in versions:
        return SCHEMA_UNKNOWN, set()
    return (SCHEMA_MAINTENANCE if needs else SCHEMA_CURRENT), needs

def inspect_session_schema(database_url: str) -> str:
    """Read-only, strict classification used by startup and maintenance.

    A file with no user objects is NEW.  Any unrelated table, view, missing
    core table or structural drift is UNKNOWN.  A canonical old schema is
    MAINTENANCE; a canonical current schema is CURRENT.  Canonical means the
    exact SQLAlchemy table DDL, so column types, defaults, keys and table
    constraints are all part of the signature.
    """
    try:
        path = _sqlite_path(database_url)
    except SessionSchemaUnknown:
        return SCHEMA_UNKNOWN
    if path is None:
        return SCHEMA_NEW
    if not path.exists():
        return SCHEMA_NEW
    if not path.is_file():
        return SCHEMA_UNKNOWN
    try:
        if path.stat().st_size == 0:
            return SCHEMA_NEW
        conn = _readonly_connection(path)
    except (OSError, sqlite3.Error):
        return SCHEMA_UNKNOWN
    try:
        quick = conn.execute("PRAGMA quick_check").fetchone()
        if not quick or str(quick[0]).lower() != "ok":
            return SCHEMA_UNKNOWN
        rows = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE substr(name, 1, 7) <> 'sqlite_'"
        ).fetchall()
        if not rows:
            return SCHEMA_NEW
        if any(row[0] == "view" for row in rows):
            return SCHEMA_UNKNOWN
        tables = {
            row[1]: (row[3] or "") for row in rows if row[0] == "table"
        }
        if not APP_TABLES.issubset(tables) or not set(tables).issubset(APP_TABLES | set(OPTIONAL_TABLES)):
            return SCHEMA_UNKNOWN
        if not all(tables.get(name, "").strip() for name in REQUIRED_TABLES):
            return SCHEMA_UNKNOWN
        if any(row[0] not in {"table", "index", "trigger"} for row in rows):
            return SCHEMA_UNKNOWN

        state, _needs = _classify_connection(conn)
        return state
    except (sqlite3.Error, ValueError):
        return SCHEMA_UNKNOWN
    finally:
        conn.close()

def _integrity_and_counts(conn: sqlite3.Connection) -> tuple[int, int]:
    integrity = conn.execute("PRAGMA integrity_check").fetchone()
    if not integrity or str(integrity[0]).lower() != "ok":
        raise sqlite3.DatabaseError("backup failed SQLite integrity_check")
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chat_sessions"
    ).fetchone()
    return int(row[0]), int(row[1])


def assert_startup_schema(database_url: str) -> str:
    """Fail closed before startup writes if the database needs maintenance."""
    state = inspect_session_schema(database_url)
    if state == SCHEMA_MAINTENANCE:
        raise SessionIdMigrationRequired(
            "legacy chat_sessions schema", committed=False
        )
    if state == SCHEMA_UNKNOWN:
        raise SessionSchemaUnknown("unknown database schema", committed=False)
    return state


def backup_database(database_url: str, backup_path: str | Path) -> Path:
    """Copy a consistent SQLite snapshot, including WAL contents.

    The target must be outside the repository and must not already exist.
    The function raises before modifying the target if those constraints are
    not met.  A partially-written target is removed on failure.
    """
    source_path = _sqlite_path(database_url)
    if source_path is None or not source_path.is_file():
        raise BackupError("backup requires an existing on-disk SQLite database")

    target = Path(backup_path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target

    # Check the final path component with lexists before resolve().  This
    # catches both an existing target and a dangling symlink, either of which
    # could otherwise be used to alias a protected path.  Parent symlinks are
    # covered by the resolved-path check below.
    if os.path.lexists(target):
        raise BackupError(f"backup target already exists: {target}")
    target = target.resolve(strict=False)
    if target.is_relative_to(_REPO_ROOT.resolve()):
        raise BackupError("backup target must be outside the repository")
    if os.path.lexists(target):
        raise BackupError(f"backup target already exists: {target}")
    if not target.parent.is_dir():
        raise BackupError(f"backup parent directory does not exist: {target.parent}")

    # Reserve the target exclusively before opening it as SQLite.
    try:
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except OSError as exc:
        raise BackupError(f"cannot reserve backup target: {exc}") from exc

    source = None
    destination = None
    try:
        source = sqlite3.connect(str(source_path))
        destination = sqlite3.connect(str(target))
        source.backup(destination)
        destination.execute("PRAGMA synchronous = FULL")
        counts = _integrity_and_counts(destination)
        source_counts = _integrity_and_counts(source)
        if counts != source_counts:
            raise BackupError("backup row count/high-water differs from source")
        return target
    except Exception:
        try:
            Path(target).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()


def _expected_columns(table_name: str = "chat_sessions") -> list[str]:
    from backend.models import Base

    return [
        column.name
        for column in Base.metadata.tables[table_name].columns
    ]


def _rebuild_table_sql(table_name: str, new_name: str) -> str:
    # Compile the canonical current table directly from the shared metadata.
    # Copying one table into a fresh MetaData would try to resolve its
    # foreign keys against a missing parent table in SQLAlchemy 2.
    ddl = _expected_table_ddl(table_name, autoincrement=True)
    prefix = f"CREATE TABLE {table_name}"
    if not ddl.startswith(prefix):
        raise RuntimeError("unexpected canonical DDL shape")
    return f"CREATE TABLE {new_name}" + ddl[len(prefix):]

def _validate_mode_counts(conn: sqlite3.Connection) -> None:
    invalid = conn.execute(
        "SELECT COUNT(*) FROM mode_switch_events "
        "WHERE reviewable_user_message_count IS NOT NULL "
        "AND (typeof(reviewable_user_message_count) <> 'integer' "
        "OR reviewable_user_message_count < 0)"
    ).fetchone()[0]
    if int(invalid):
        raise SessionIdMigrationError(
            "validate_mode_count", committed=False
        )


def _read_needs(path: Path) -> set[str]:
    conn = _readonly_connection(path)
    try:
        state, needs = _classify_connection(conn)
        if state != SCHEMA_MAINTENANCE:
            if state == SCHEMA_CURRENT:
                return set()
            raise SessionSchemaUnknown("unknown schema", committed=False)
        return needs
    finally:
        conn.close()


def _restore_foreign_keys(conn: sqlite3.Connection) -> tuple[bool, str | None]:
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        value = conn.execute("PRAGMA foreign_keys").fetchone()
        if value is None or int(value[0]) != 1:
            return False, "PRAGMA foreign_keys did not read back as 1"
        return True, None
    except Exception as exc:  # noqa: BLE001 - report cleanup failure
        return False, f"{type(exc).__name__}: {exc}"


def _rollback_verified(
    conn: sqlite3.Connection,
    before_ddl: str | None,
    before_count: int | None,
    before_max: int | None,
) -> bool:
    if before_ddl is None or before_count is None or before_max is None:
        return False
    try:
        actual = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='chat_sessions'"
        ).fetchone()
        if actual is None or _normalise_sql(actual[0] or "") != _normalise_sql(before_ddl):
            return False
        count, high = _integrity_and_counts(conn)
        if (count, high) != (before_count, before_max):
            return False
        return not conn.execute("PRAGMA foreign_key_check").fetchall()
    except sqlite3.Error:
        return False


def _table_objects(conn: sqlite3.Connection, table_name: str):
    return conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE tbl_name = ? AND type IN ('index', 'trigger') AND sql IS NOT NULL",
        (table_name,),
    ).fetchall()


def _unique_index_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f'PRAGMA index_list("{table_name}")').fetchall()
        if int(row[2])
    }


def _rebuild_one_table(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    failure_hook: Callable[[str], None] | None,
    mode_steps: bool,
) -> tuple[int, int]:
    expected = _expected_columns(table_name)
    actual = [
        row[1]
        for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    ]
    if set(actual) != set(expected):
        raise SessionIdMigrationError("validate_columns", committed=False)
    before_count, before_max = _integrity_and_counts_for(conn, table_name)
    objects = _table_objects(conn, table_name)
    unique_names = _unique_index_names(conn, table_name) if table_name == "mode_switch_events" else set()
    new_name = f"{table_name}_session_id_rebuild"
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (new_name,),
    ).fetchone():
        raise SessionIdMigrationError("create_table", committed=False)
    conn.execute(_rebuild_table_sql(table_name, new_name))
    if failure_hook:
        failure_hook("after_mode_create_table" if mode_steps else "after_create_table")
    quoted = ", ".join(f'"{name}"' for name in expected)
    conn.execute(
        f'INSERT INTO "{new_name}" ({quoted}) '
        f'SELECT {quoted} FROM "{table_name}"'
    )
    if failure_hook:
        failure_hook("after_mode_copy" if mode_steps else "after_copy")
    conn.execute(f'DROP TABLE "{table_name}"')
    if failure_hook:
        failure_hook("after_mode_drop" if mode_steps else "after_drop")
    conn.execute(f'ALTER TABLE "{new_name}" RENAME TO "{table_name}"')
    if failure_hook:
        failure_hook("after_mode_rename" if mode_steps else "after_rename")
    restored = 0
    for object_type, name, sql in objects:
        if object_type == "index" and name in unique_names:
            # The new table has the canonical inline UNIQUE constraint; the
            # old standalone equivalent must not be duplicated.
            continue
        conn.execute(sql)
        restored += 1
    if failure_hook:
        failure_hook(
            "after_mode_restore_objects" if mode_steps
            else "after_restore_objects"
        )
    after_count, after_max = _integrity_and_counts_for(conn, table_name)
    if (after_count, after_max) != (before_count, before_max):
        raise SessionIdMigrationError("compare_counts", committed=False)
    return after_count, after_max


def _integrity_and_counts_for(
    conn: sqlite3.Connection, table_name: str
) -> tuple[int, int]:
    integrity = conn.execute("PRAGMA integrity_check").fetchone()
    if not integrity or str(integrity[0]).lower() != "ok":
        raise sqlite3.DatabaseError("SQLite integrity_check failed")
    row = conn.execute(
        f'SELECT COUNT(*), COALESCE(MAX(id), 0) FROM "{table_name}"'
    ).fetchone()
    return int(row[0]), int(row[1])


def migrate_session_id_schema(
    database_url: str,
    backup_path: str | Path,
    *,
    backup_already_done: bool = False,
    failure_hook: Callable[[str], None] | None = None,
    connection_factory: Callable[[Path], sqlite3.Connection] | None = None,
) -> MigrationResult:
    """Repair known legacy session/mode schemas in one transaction."""
    state = inspect_session_schema(database_url)
    if state == SCHEMA_CURRENT:
        return MigrationResult(False, None, SESSION_ID_SCHEMA_VERSION)
    if state == SCHEMA_NEW:
        raise SessionIdMigrationRequired("new database", committed=False)
    if state != SCHEMA_MAINTENANCE:
        raise SessionSchemaUnknown("unknown schema", committed=False)

    path = _sqlite_path(database_url)
    if path is None:
        raise BackupError("maintenance requires an on-disk SQLite database")
    needs = _read_needs(path)
    if "mode_check" in needs:
        check_conn = _readonly_connection(path)
        try:
            _validate_mode_counts(check_conn)
        finally:
            check_conn.close()

    target_backup: Path | None = None
    if not backup_already_done:
        target_backup = backup_database(database_url, backup_path)
    elif Path(backup_path).exists():
        target_backup = Path(backup_path)

    conn: sqlite3.Connection | None = None
    failure: BaseException | None = None
    cleanup_error: str | None = None
    commit_state = COMMIT_NOT_COMMITTED
    rollback_state = ROLLBACK_NOT_ATTEMPTED
    commit_attempted = False
    before_ddl: str | None = None
    before_count: int | None = None
    before_max: int | None = None
    try:
        conn = (
            connection_factory(path)
            if connection_factory is not None
            else sqlite3.connect(str(path), isolation_level=None)
        )
        conn.execute("PRAGMA foreign_keys = OFF")
        value = conn.execute("PRAGMA foreign_keys").fetchone()
        if value is None or int(value[0]) != 0:
            raise SessionIdMigrationError("disable_foreign_keys", committed=False)
        if failure_hook:
            failure_hook("after_begin_setup")
        conn.execute("BEGIN IMMEDIATE")
        state, needs = _classify_connection(conn)
        if state != SCHEMA_MAINTENANCE:
            raise SessionIdMigrationError("revalidate_schema", committed=False)
        _validate_mode_counts(conn)
        if failure_hook:
            failure_hook("after_validate")

        old_ddl_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_sessions'"
        ).fetchone()
        before_ddl = old_ddl_row[0] if old_ddl_row else None
        before_count, before_max = _integrity_and_counts_for(conn, "chat_sessions")

        rebuilt = []
        if "autoincrement" in needs:
            rebuilt.append(("chat_sessions", False))
        if "mode_check" in needs:
            rebuilt.append(("mode_switch_events", True))
        for table_name, mode_steps in rebuilt:
            _rebuild_one_table(
                conn,
                table_name,
                failure_hook=failure_hook,
                mode_steps=mode_steps,
            )

        # Optional additive settings must exist before the current ORM can
        # read a repaired pre-feature database. Parent rows are preserved.
        from backend.models import Base
        for name in OPTIONAL_TABLES:
            if _table_shape(conn, name) is None:
                conn.execute(str(CreateTable(Base.metadata.tables[name]).compile(
                    dialect=sqlite_dialect.dialect()
                )))

        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise SessionIdMigrationError("foreign_key_check", committed=False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version VARCHAR(255) NOT NULL PRIMARY KEY, "
            "applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        if "autoincrement" in needs:
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (SESSION_ID_SCHEMA_VERSION,),
            )
        if "mode_check" in needs:
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (MODE_SWITCH_CHECK_VERSION,),
            )
        if failure_hook:
            failure_hook("after_version")
        commit_attempted = True
        conn.execute("COMMIT")
        commit_state = COMMIT_COMMITTED
    except BaseException as exc:  # noqa: BLE001
        failure = exc
        if commit_attempted and commit_state == COMMIT_NOT_COMMITTED:
            commit_state = COMMIT_UNKNOWN
        if conn is not None and conn.in_transaction:
            try:
                conn.execute("ROLLBACK")
            except Exception as rollback_exc:  # noqa: BLE001
                rollback_state = ROLLBACK_FAILED
                cleanup_error = f"rollback failed: {type(rollback_exc).__name__}"
            else:
                rollback_state = (
                    ROLLBACK_SUCCEEDED
                    if _rollback_verified(
                        conn, before_ddl, before_count, before_max
                    )
                    else ROLLBACK_UNKNOWN
                )
    finally:
        if conn is not None:
            try:
                ok, restore_error = _restore_foreign_keys(conn)
                if not ok:
                    message = restore_error or "foreign_keys restore failed"
                    cleanup_error = f"{cleanup_error}; {message}" if cleanup_error else message
            except Exception as restore_exc:  # noqa: BLE001
                message = f"foreign_keys restore raised {type(restore_exc).__name__}"
                cleanup_error = f"{cleanup_error}; {message}" if cleanup_error else message
            try:
                conn.close()
            except Exception as close_exc:  # noqa: BLE001
                message = f"connection close failed {type(close_exc).__name__}"
                cleanup_error = f"{cleanup_error}; {message}" if cleanup_error else message

    if failure is not None:
        committed: bool | None = (
            True if commit_state == COMMIT_COMMITTED
            else None if commit_state == COMMIT_UNKNOWN
            else False
        )
        stage = failure.stage if isinstance(failure, SessionIdMigrationError) else "unexpected"
        raise SessionIdMigrationError(
            stage,
            committed=committed,
            cleanup_error=cleanup_error,
            commit_state=commit_state,
            rollback_state=rollback_state,
        ) from failure
    if cleanup_error:
        raise SessionIdMigrationError(
            "restore_foreign_keys",
            committed=True,
            cleanup_error=cleanup_error,
            commit_state=COMMIT_COMMITTED,
            rollback_state=ROLLBACK_NOT_ATTEMPTED,
        )
    return MigrationResult(True, target_backup, SESSION_ID_SCHEMA_VERSION)
def maintenance_command(database_url: str, backup_path: str) -> int:
    """Explicit maintenance entry point; never called by app startup."""
    state = inspect_session_schema(database_url)
    if state == SCHEMA_NEW:
        print("database is new; start the application normally")
        return 0
    if state == SCHEMA_UNKNOWN:
        print("refusing unknown database schema", file=sys.stderr)
        return 2
    if state == SCHEMA_CURRENT:
        print("session id schema is already current")
        return 0

    migrate_session_id_schema(database_url, backup_path)
    print("session id migration completed")
    return 0


def _safe_stage(stage: str) -> str:
    return stage if re.fullmatch(r"[a-z0-9_]{1,64}", stage) else "unknown"


def _failure_hint(
    commit_state: str, rollback_state: str, cleanup_error: str | None
) -> str:
    if commit_state == COMMIT_COMMITTED:
        if cleanup_error:
            return (
                "migration committed, but cleanup failed; do not start the "
                "service until a human verifies schema and data"
            )
        return "migration committed and verified"
    if commit_state == COMMIT_UNKNOWN:
        return (
            "commit outcome is unknown; do not start the service; inspect the "
            "database and restore the explicit backup if needed"
        )
    if rollback_state == ROLLBACK_SUCCEEDED:
        return (
            "migration was not committed; rollback passed the limited checks "
            "(chat_sessions and mode_switch_events DDL, row counts, maximum "
            "ids, foreign_key_check); associated data was not independently "
            "audited"
        )
    if rollback_state in (ROLLBACK_FAILED, ROLLBACK_UNKNOWN):
        return (
            "migration was not committed, but rollback is not confirmed; do "
            "not start the service; restore the explicit backup after inspection"
        )
    return (
        "migration was not committed and no transaction rollback was "
        "attempted; inspect before starting the service"
    )


def _print_failure(exc: BaseException) -> None:
    if isinstance(exc, SessionIdMigrationRequired):
        print(str(exc), file=sys.stderr)
        return
    if isinstance(exc, SessionSchemaUnknown):
        print(str(exc), file=sys.stderr)
        return
    if isinstance(exc, SessionIdMigrationError):
        stage = _safe_stage(exc.stage)
        cleanup = "yes" if exc.cleanup_error else "no"
        print(
            f"maintenance failed: stage={stage} "
            f"commit_state={exc.commit_state} "
            f"rollback_state={exc.rollback_state} "
            f"cleanup_failed={cleanup}",
            file=sys.stderr,
        )
        print(
            "next: " + _failure_hint(
                exc.commit_state, exc.rollback_state, exc.cleanup_error
            ),
            file=sys.stderr,
        )
        return
    if isinstance(exc, BackupError):
        print(
            "maintenance failed: backup failed before migration; "
            "database was not modified",
            file=sys.stderr,
        )
        return
    print(
        "maintenance failed: unexpected error; inspect logs and the backup",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--backup-path", required=True)
    args = parser.parse_args(argv)
    try:
        return maintenance_command(args.database_url, args.backup_path)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _print_failure(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


def record_session_schema_version(engine) -> None:
    """Record the current session-id invariant after a verified schema."""
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version VARCHAR(255) NOT NULL PRIMARY KEY, "
            "applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.exec_driver_sql(
            "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
            (SESSION_ID_SCHEMA_VERSION,),
        )
