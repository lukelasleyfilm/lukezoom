"""
lukezoom.episodic.integrity — Memory health diagnostics and repair tools.

Provides rebuild_fts(), cleanup_orphaned_consolidations(), memory_health(),
and deep_purge() — all operating on the EpisodicStore's SQLite connection.

These are implemented as standalone functions taking a sqlite3.Connection
rather than methods, to keep the main store class focused on CRUD.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

log = logging.getLogger(__name__)


def rebuild_fts(conn: sqlite3.Connection) -> Dict[str, int]:
    """Rebuild FTS5 indexes from source tables.

    Fixes desynchronization caused by crashes or interrupted writes.
    Returns counts of indexed rows per table.
    """
    counts = {}
    for table in ("messages", "traces"):
        fts = f"{table}_fts"
        conn.execute(f"INSERT INTO {fts}({fts}) VALUES('rebuild')")
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = row[0] if row else 0
    conn.commit()
    return counts


def cleanup_orphaned_consolidations(conn: sqlite3.Connection) -> Dict[str, int]:
    """Repair orphaned consolidation references.

    Fixes three types of orphans:
    1. Dead children: threads/arcs referencing traces that were pruned.
    2. Orphaned marks: traces with consolidated_into pointing to deleted parents.
    3. Stale sessions: active sessions with 0 messages older than 24h.

    Returns counts of repairs per category.
    """
    counts = {"dead_children": 0, "orphaned_marks": 0, "stale_sessions": 0}

    # 1. Dead children — threads/arcs with child_ids pointing to deleted traces
    consolidation_rows = conn.execute(
        "SELECT id, metadata FROM traces WHERE kind IN ('thread', 'arc')"
    ).fetchall()
    all_trace_ids = set(
        r[0] for r in conn.execute("SELECT id FROM traces").fetchall()
    )

    for row in consolidation_rows:
        trace_id, meta_str = row[0], row[1]
        if not meta_str:
            continue
        try:
            meta = json.loads(meta_str)
        except (json.JSONDecodeError, TypeError):
            continue
        child_ids = meta.get("child_ids", [])
        if not child_ids:
            continue
        live_ids = [cid for cid in child_ids if cid in all_trace_ids]
        if len(live_ids) < len(child_ids):
            dead_count = len(child_ids) - len(live_ids)
            meta["child_ids"] = live_ids
            conn.execute(
                "UPDATE traces SET metadata = ? WHERE id = ?",
                (json.dumps(meta), trace_id),
            )
            counts["dead_children"] += dead_count

    # 2. Orphaned marks — traces with consolidated_into pointing to deleted parents
    marked_rows = conn.execute(
        """SELECT id, metadata FROM traces
           WHERE json_extract(metadata, '$.consolidated_into') IS NOT NULL
             AND json_extract(metadata, '$.consolidated_into') != ''"""
    ).fetchall()

    for row in marked_rows:
        trace_id, meta_str = row[0], row[1]
        try:
            meta = json.loads(meta_str)
        except (json.JSONDecodeError, TypeError):
            continue
        parent_id = meta.get("consolidated_into", "")
        if parent_id and parent_id not in all_trace_ids:
            del meta["consolidated_into"]
            conn.execute(
                "UPDATE traces SET metadata = ? WHERE id = ?",
                (json.dumps(meta), trace_id),
            )
            counts["orphaned_marks"] += 1

    # 3. Stale sessions — active sessions with 0 messages older than 24h
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=24)
    ).isoformat().replace("+00:00", "Z")
    cur = conn.execute(
        """DELETE FROM sessions
           WHERE ended IS NULL AND message_count = 0 AND started < ?""",
        (cutoff,),
    )
    counts["stale_sessions"] = cur.rowcount if hasattr(cur, "rowcount") else 0

    conn.commit()
    return counts


def memory_health(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Comprehensive memory health diagnostic.

    Returns fragmentation score (0.0 = healthy, 1.0 = critical),
    FTS sync status, consolidation coverage, salience distribution,
    and session state.
    """
    # Counts
    total_traces = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    low_salience = 0
    unconsolidated = 0

    if total_traces > 0:
        row = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE salience < 0.1"
        ).fetchone()
        low_salience = row[0] if row else 0

        row = conn.execute(
            """SELECT COUNT(*) FROM traces
               WHERE kind IN ('episode', 'summary')
                 AND COALESCE(json_extract(metadata, '$.consolidated_into'), '') = ''"""
        ).fetchone()
        unconsolidated = row[0] if row else 0

    low_salience_ratio = low_salience / max(total_traces, 1)
    unconsolidated_ratio = unconsolidated / max(total_traces, 1)

    # FTS sync check — compare counts
    fts_desync = 0.0
    for table in ("messages", "traces"):
        fts = f"{table}_fts"
        try:
            source_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            fts_count = conn.execute(f"SELECT COUNT(*) FROM {fts}").fetchone()[0]
            if source_count > 0 and abs(source_count - fts_count) > 0:
                fts_desync = max(
                    fts_desync, abs(source_count - fts_count) / source_count
                )
        except Exception:
            fts_desync = 1.0

    # Composite fragmentation score
    fragmentation = (
        low_salience_ratio * 0.5
        + unconsolidated_ratio * 0.3
        + fts_desync * 0.2
    )

    # Session state
    active_sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE ended IS NULL"
    ).fetchone()[0]

    # Consolidation coverage
    consolidated = 0
    if total_traces > 0:
        row = conn.execute(
            """SELECT COUNT(*) FROM traces
               WHERE json_extract(metadata, '$.consolidated_into') IS NOT NULL
                 AND json_extract(metadata, '$.consolidated_into') != ''"""
        ).fetchone()
        consolidated = row[0] if row else 0

    return {
        "fragmentation": round(min(1.0, fragmentation), 4),
        "total_traces": total_traces,
        "total_messages": total_messages,
        "low_salience_traces": low_salience,
        "unconsolidated_traces": unconsolidated,
        "consolidated_traces": consolidated,
        "fts_desync_score": round(fts_desync, 4),
        "active_sessions": active_sessions,
        "status": (
            "critical"
            if fragmentation > 0.7
            else "warning"
            if fragmentation > 0.3
            else "healthy"
        ),
    }


def deep_purge(
    conn: sqlite3.Connection,
    person: str,
    purge_fn,
) -> Dict[str, int]:
    """Deep purge: delete all data AND redact consolidation references.

    Extends purge_person() by scanning thread/arc content for person
    name references and redacting them. Addresses the GDPR consolidation
    erasure problem where derived knowledge persists after source deletion.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    person : str
        Person name to purge.
    purge_fn : callable
        The store's purge_person method for standard deletion.

    Returns counts of deleted/redacted rows per category.
    """
    # Standard purge first
    counts = purge_fn(person)

    # Scan remaining consolidation traces for person references
    pattern = re.compile(re.escape(person), re.IGNORECASE)
    consolidation_rows = conn.execute(
        "SELECT id, content, metadata FROM traces "
        "WHERE kind IN ('summary', 'thread', 'arc')"
    ).fetchall()

    redacted = 0
    deleted_consolidations = 0
    for row in consolidation_rows:
        trace_id, content, meta_str = row[0], row[1], row[2]
        if content and pattern.search(content):
            new_content = pattern.sub("[REDACTED]", content)
            if (
                new_content.count("[REDACTED]") > 2
                or len(new_content.replace("[REDACTED]", "").strip()) < 20
            ):
                conn.execute("DELETE FROM traces WHERE id = ?", (trace_id,))
                deleted_consolidations += 1
            else:
                conn.execute(
                    "UPDATE traces SET content = ? WHERE id = ?",
                    (new_content, trace_id),
                )
                redacted += 1

    counts["consolidation_redacted"] = redacted
    counts["consolidation_deleted"] = deleted_consolidations
    conn.commit()
    return counts
