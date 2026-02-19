"""
lukezoom.episodic.schema — SQLite schema creation, FTS triggers, and migrations.

Manages the episodic store's table definitions and schema versioning
using PRAGMA user_version. Each migration is a numbered SQL function
that runs within an exclusive transaction.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import List, Callable

log = logging.getLogger(__name__)

# Current schema version — increment when adding migrations.
SCHEMA_VERSION = 2


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all episodic store tables, indexes, FTS tables, and triggers."""
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          TEXT PRIMARY KEY,
            person      TEXT,
            speaker     TEXT,
            content     TEXT,
            source      TEXT,
            timestamp   TEXT,
            salience    REAL DEFAULT 0.5,
            signal      JSON,
            metadata    JSON
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id              TEXT PRIMARY KEY,
            content         TEXT,
            created         TEXT,
            kind            TEXT,
            tags            JSON,
            salience        REAL DEFAULT 0.5,
            tokens          INTEGER,
            access_count    INTEGER DEFAULT 0,
            last_accessed   TEXT,
            metadata        JSON
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          TEXT PRIMARY KEY,
            type        TEXT,
            description TEXT,
            person      TEXT,
            timestamp   TEXT,
            salience    REAL DEFAULT 0.5,
            metadata    JSON
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            person      TEXT,
            started     TEXT,
            ended       TEXT,
            message_count INTEGER DEFAULT 0,
            summary     TEXT,
            metadata    JSON
        )
    """)

    # Indexes
    for stmt in [
        "CREATE INDEX IF NOT EXISTS idx_messages_person    ON messages(person)",
        "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_messages_salience  ON messages(salience)",
        "CREATE INDEX IF NOT EXISTS idx_traces_salience    ON traces(salience)",
        "CREATE INDEX IF NOT EXISTS idx_traces_kind        ON traces(kind)",
        "CREATE INDEX IF NOT EXISTS idx_events_type        ON events(type)",
        "CREATE INDEX IF NOT EXISTS idx_events_timestamp   ON events(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_person    ON sessions(person)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_started   ON sessions(started)",
    ]:
        c.execute(stmt)

    # FTS5 full-text search
    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
        USING fts5(content, content=messages, content_rowid=rowid)
    """)
    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS traces_fts
        USING fts5(content, content=traces, content_rowid=rowid)
    """)

    # FTS sync triggers
    for table in ("messages", "traces"):
        fts = f"{table}_fts"
        c.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_ai AFTER INSERT ON {table}
            BEGIN
                INSERT INTO {fts}(rowid, content)
                VALUES (new.rowid, new.content);
            END
        """)
        c.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_ad AFTER DELETE ON {table}
            BEGIN
                INSERT INTO {fts}({fts}, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END
        """)
        c.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_au AFTER UPDATE ON {table}
            BEGIN
                INSERT INTO {fts}({fts}, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO {fts}(rowid, content)
                VALUES (new.rowid, new.content);
            END
        """)

    conn.commit()


# ── Migration system using PRAGMA user_version ───────────────────────

def _migration_v1(conn: sqlite3.Connection) -> None:
    """v0 → v1: Initial schema — tables already exist via create_tables()."""
    pass  # create_tables() handles the initial schema


def _migration_v2(conn: sqlite3.Connection) -> None:
    """v1 → v2: Add composite index for common consolidation queries."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_traces_kind_salience "
        "ON traces(kind, salience DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_traces_consolidated "
        "ON traces(kind) WHERE json_extract(metadata, '$.consolidated_into') IS NOT NULL"
    )


# Ordered list of migrations. Index = target version.
_MIGRATIONS: List[Callable[[sqlite3.Connection], None]] = [
    lambda _: None,  # v0: placeholder (unused)
    _migration_v1,   # v0 → v1
    _migration_v2,   # v1 → v2
]


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Read the current schema version from PRAGMA user_version."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return row[0] if row else 0


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Set the schema version in PRAGMA user_version."""
    conn.execute(f"PRAGMA user_version = {int(version)}")


def run_migrations(conn: sqlite3.Connection) -> int:
    """Run any pending schema migrations.

    Returns the number of migrations applied. Each migration runs
    in its own transaction for safety.
    """
    current = get_schema_version(conn)
    applied = 0

    for target_version in range(current + 1, SCHEMA_VERSION + 1):
        if target_version >= len(_MIGRATIONS):
            break
        migration_fn = _MIGRATIONS[target_version]

        log.info("Applying migration v%d → v%d", target_version - 1, target_version)
        try:
            migration_fn(conn)
            set_schema_version(conn, target_version)
            conn.commit()
            applied += 1
        except Exception as exc:
            log.error("Migration to v%d failed: %s", target_version, exc)
            conn.rollback()
            raise

    return applied


def initialize(conn: sqlite3.Connection) -> None:
    """Create tables and run migrations. Safe to call on every startup."""
    create_tables(conn)

    current = get_schema_version(conn)
    if current == 0:
        # Fresh database — set to current version
        set_schema_version(conn, SCHEMA_VERSION)
        conn.commit()
    elif current < SCHEMA_VERSION:
        run_migrations(conn)
