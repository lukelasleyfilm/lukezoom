"""
Episodic memory store — SQLite-backed storage for conversations,
events, and experiential traces.

Every interaction leaves a trace. Salience determines what survives.
"""

import logging
import re
import sqlite3
import json
import math
import threading
import uuid
from pathlib import Path
from typing import Any, List, Optional, Dict
from datetime import datetime, timezone

_log = logging.getLogger(__name__)

# Characters and keywords that have special meaning in FTS5 MATCH queries.
_FTS5_SPECIAL = re.compile(r'[*"()\[\]{}^~+:\-]')
_FTS5_OPERATORS = re.compile(r'\b(AND|OR|NOT|NEAR)\b', re.IGNORECASE)


def _sanitize_fts_query(query: str) -> str:
    """Strip FTS5 special characters and operators to prevent injection.

    Returns a plain-text query safe for use in MATCH clauses.
    """
    q = _FTS5_SPECIAL.sub(" ", query)
    q = _FTS5_OPERATORS.sub(" ", q)
    # Collapse whitespace
    q = " ".join(q.split())
    return q.strip()


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EpisodicStore:
    """SQLite-based episodic memory: messages, traces, and events."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

        # Schema creation and migrations
        from lukezoom.episodic.schema import initialize
        initialize(self.conn)

    # ── Thread-safe helpers ─────────────────────────────────────

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement under the connection lock."""
        with self._lock:
            return self.conn.execute(sql, params)

    def _execute_commit(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement and commit under the connection lock."""
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def _fetchall(self, sql: str, params: tuple = ()) -> list:
        """Execute and fetchall under the connection lock."""
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def _fetchone(self, sql: str, params: tuple = ()):
        """Execute and fetchone under the connection lock."""
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    # ── Write ─────────────────────────────────────────────────

    def log_message(
        self,
        person: str,
        speaker: str,
        content: str,
        source: str,
        salience: float = 0.5,
        signal: Optional[Dict] = None,
        **metadata,
    ) -> str:
        """Record a conversation message. Returns the message ID."""
        salience = max(0.0, min(1.0, salience))
        msg_id = _generate_id()
        self._execute_commit(
            """
            INSERT INTO messages (id, person, speaker, content, source,
                                  timestamp, salience, signal, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg_id,
                person,
                speaker,
                content,
                source,
                _now(),
                salience,
                json.dumps(signal) if signal else None,
                json.dumps(metadata) if metadata else None,
            ),
        )
        return msg_id

    def log_trace(
        self,
        content: str,
        kind: str,
        tags: List[str],
        salience: float = 0.5,
        **metadata,
    ) -> str:
        """Record an experiential trace (summary, insight, reflection). Returns trace ID."""
        from lukezoom.core.types import TRACE_KINDS

        if kind not in TRACE_KINDS:
            raise ValueError(
                f"Invalid trace kind {kind!r}. Must be one of {sorted(TRACE_KINDS)}"
            )
        salience = max(0.0, min(1.0, salience))
        trace_id = _generate_id()
        now = _now()
        tokens = max(
            1, len(content) // 4
        )  # char/4 heuristic, consistent with core.tokens
        self._execute_commit(
            """
            INSERT INTO traces (id, content, created, kind, tags, salience,
                                tokens, access_count, last_accessed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                trace_id,
                content,
                now,
                kind,
                json.dumps(tags),
                salience,
                tokens,
                now,
                json.dumps(metadata) if metadata else None,
            ),
        )
        return trace_id

    def log_event(
        self,
        type: str,
        description: str,
        person: Optional[str] = None,
        salience: float = 0.5,
        **metadata,
    ) -> str:
        """Record a discrete event (trust change, injury, milestone). Returns event ID."""
        salience = max(0.0, min(1.0, salience))
        event_id = _generate_id()
        self._execute_commit(
            """
            INSERT INTO events (id, type, description, person,
                                timestamp, salience, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                type,
                description,
                person,
                _now(),
                salience,
                json.dumps(metadata) if metadata else None,
            ),
        )
        return event_id

    # ── Read ──────────────────────────────────────────────────

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert a sqlite3.Row to a plain dict, deserializing JSON fields."""
        d = dict(row)
        for key in ("signal", "metadata", "tags"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def get_messages(
        self,
        person: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 50,
        min_salience: float = 0.0,
    ) -> List[Dict]:
        """Retrieve messages with optional filters."""
        clauses = ["salience >= ?"]
        params: list = [min_salience]

        if person:
            clauses.append("person = ?")
            params.append(person)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)

        where = " AND ".join(clauses)
        params.append(limit)

        rows = self._fetchall(
            f"SELECT * FROM messages WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            tuple(params),
        )
        return [self._row_to_dict(r) for r in rows]

    def get_traces(
        self,
        tags: Optional[List[str]] = None,
        kind: Optional[str] = None,
        min_salience: float = 0.0,
        limit: int = 50,
    ) -> List[Dict]:
        """Retrieve traces with optional filters.

        When tags are provided, filtering is done at the SQL level
        using json_each to avoid the LIMIT-before-filter data loss bug.
        """
        if tags:
            # SQL-level tag filtering using json_each for accuracy
            tag_placeholders = ",".join("?" for _ in tags)
            kind_clause = "AND t.kind = ?" if kind else ""
            params: list = [min_salience] + [t.lower() for t in tags]
            if kind:
                params.append(kind)
            params.append(limit)

            rows = self._fetchall(
                f"""
                SELECT t.* FROM traces t, json_each(t.tags) AS tag
                WHERE t.salience >= ?
                  AND LOWER(tag.value) IN ({tag_placeholders})
                  {kind_clause}
                GROUP BY t.id
                ORDER BY t.salience DESC
                LIMIT ?
                """,
                tuple(params),
            )
            return [self._row_to_dict(r) for r in rows]
        else:
            clauses = ["salience >= ?"]
            params = [min_salience]

            if kind:
                clauses.append("kind = ?")
                params.append(kind)

            where = " AND ".join(clauses)
            params.append(limit)

            rows = self._fetchall(
                f"SELECT * FROM traces WHERE {where} ORDER BY salience DESC LIMIT ?",
                tuple(params),
            )
            return [self._row_to_dict(r) for r in rows]

    def get_recent_messages(self, person: str, limit: int = 20) -> List[Dict]:
        """Get the most recent messages for a person, in chronological order."""
        rows = self._fetchall(
            """
            SELECT * FROM messages
            WHERE person = ?
            ORDER BY timestamp DESC, rowid DESC
            LIMIT ?
            """,
            (person, limit),
        )
        results = [self._row_to_dict(r) for r in rows]
        results.reverse()  # chronological
        return results

    def get_by_salience(
        self, person: Optional[str] = None, limit: int = 30
    ) -> List[Dict]:
        """Get the highest-salience traces, optionally filtered by person tag."""
        rows = self._fetchall(
            "SELECT * FROM traces ORDER BY salience DESC LIMIT ?",
            (limit * 3,),  # over-fetch to allow Python tag filtering
        )

        results = [self._row_to_dict(r) for r in rows]

        if person:
            person_lower = person.lower()
            results = [
                r
                for r in results
                if r.get("tags") and person_lower in [t.lower() for t in r["tags"]]
            ]

        return results[:limit]

    def search_messages(
        self, query: str, person: Optional[str] = None, limit: int = 20
    ) -> List[Dict]:
        """Full-text search over message content."""
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []
        try:
            if person:
                rows = self.conn.execute(
                    """
                    SELECT m.* FROM messages m
                    JOIN messages_fts fts ON m.rowid = fts.rowid
                    WHERE messages_fts MATCH ? AND m.person = ?
                    ORDER BY fts.rank
                    LIMIT ?
                    """,
                    (safe_query, person, limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT m.* FROM messages m
                    JOIN messages_fts fts ON m.rowid = fts.rowid
                    WHERE messages_fts MATCH ?
                ORDER BY fts.rank
                LIMIT ?
                """,
                (safe_query, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except sqlite3.OperationalError as exc:
            _log.warning("FTS5 search_messages failed: %s", exc)
            return []

    def search_traces(self, query: str, limit: int = 20) -> List[Dict]:
        """Full-text search over trace content."""
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT t.* FROM traces t
                JOIN traces_fts fts ON t.rowid = fts.rowid
                WHERE traces_fts MATCH ?
                ORDER BY fts.rank
                LIMIT ?
                """,
                (safe_query, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except sqlite3.OperationalError as exc:
            _log.warning("FTS5 search_traces failed: %s", exc)
            return []

    # ── Single lookups ────────────────────────────────────────

    def get_trace(self, id: str) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM traces WHERE id = ?", (id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def get_message(self, id: str) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM messages WHERE id = ?", (id,)).fetchone()
        return self._row_to_dict(row) if row else None

    # ── Consolidation queries (public API for consolidator) ───

    def get_unconsolidated_traces(
        self, kinds: tuple = ("episode", "summary"), limit: int = 200
    ) -> List[Dict]:
        """Get traces not yet rolled into a higher-level consolidation.

        A trace is 'unconsolidated' if its metadata lacks a
        ``consolidated_into`` key.
        """
        placeholders = ",".join("?" for _ in kinds)
        rows = self.conn.execute(
            f"""
            SELECT * FROM traces
            WHERE kind IN ({placeholders})
              AND COALESCE(json_extract(metadata, '$.consolidated_into'), '') = ''
            ORDER BY created ASC
            LIMIT ?
            """,
            (*kinds, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def mark_consolidated(self, trace_ids: List[str], parent_id: str) -> None:
        """Set ``metadata.consolidated_into`` on each child trace.

        Prevents re-consolidation in future runs.
        """
        import json as _json

        with self._lock:
            for tid in trace_ids:
                try:
                    row = self.conn.execute(
                        "SELECT metadata FROM traces WHERE id = ?", (tid,)
                    ).fetchone()
                    if row is None:
                        continue
                    metadata: Dict = {}
                    if row[0]:
                        try:
                            metadata = _json.loads(row[0])
                        except (_json.JSONDecodeError, TypeError):
                            metadata = {}
                    metadata["consolidated_into"] = parent_id
                    self.conn.execute(
                        "UPDATE traces SET metadata = ? WHERE id = ?",
                        (_json.dumps(metadata), tid),
                    )
                except Exception:
                    pass
            try:
                self.conn.commit()
            except Exception:
                pass

    # ── Salience management ───────────────────────────────────

    def reinforce(self, table: str, id: str, delta: float):
        """Increase salience for a record. Clamps to [0, 1]."""
        self._validate_table(table)
        self._execute_commit(
            f"UPDATE {table} SET salience = MIN(1.0, salience + ?) WHERE id = ?",
            (abs(delta), id),
        )

    def weaken(self, table: str, id: str, delta: float):
        """Decrease salience for a record. Clamps to [0, 1]."""
        self._validate_table(table)
        self._execute_commit(
            f"UPDATE {table} SET salience = MAX(0.0, salience - ?) WHERE id = ?",
            (abs(delta), id),
        )

    def update_access(self, table: str, id: str):
        """Record that a trace/message was accessed (retrieved for context)."""
        self._validate_table(table)
        now = _now()
        if table == "traces":
            self._execute_commit(
                "UPDATE traces SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, id),
            )
        else:
            pass

    # ── Statistics ─────────────────────────────────────────────

    def count_messages(self, person: Optional[str] = None) -> int:
        if person:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM messages WHERE person = ?", (person,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return row[0] if row else 0

    def count_traces(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM traces").fetchone()
        return row[0] if row else 0

    def count_events(self, type: Optional[str] = None) -> int:
        """Count events, optionally filtered by type."""
        if type:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE type = ?", (type,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return row[0] if row else 0

    def get_events(
        self,
        type: Optional[str] = None,
        person: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Retrieve events, optionally filtered by type and/or person."""
        clauses: List[str] = []
        params: list = []
        if type:
            clauses.append("type = ?")
            params.append(type)
        if person:
            clauses.append("person = ?")
            params.append(person)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)

        rows = self.conn.execute(
            f"SELECT * FROM events{where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def avg_salience(self, table: str) -> float:
        self._validate_table(table)
        row = self.conn.execute(f"SELECT AVG(salience) FROM {table}").fetchone()
        return round(row[0], 4) if row and row[0] is not None else 0.0

    # ── Decay & pruning ───────────────────────────────────────

    def decay_pass(self, half_life_hours: float, coherence: float):
        """
        Run adaptive exponential decay across all traces.

        Scientific grounding (validated Feb 2026):
          - Power law of forgetting (Radvansky et al. 2024, meta-analysis
            of 916 datasets): no single function is universal, but power
            law best approximates aggregate multi-trace systems.
          - ACT-R base-level activation: B_i = ln(Σ t_j^(-d)), d=0.5.
          - Coherence-driven adaptive half-life supported by schema
            congruency effects (van Kesteren et al. 2012, Trends in
            Neuroscience): schema-congruent memories show 15-40%
            retention advantages.
          - Access-frequency resistance grounded in Bjork's New Theory
            of Disuse (1992) and the spacing effect (259/271 cases
            favor spaced retrieval, Cepeda et al. 2006).

        Formula:
            decay_rate = ln(2) / half_life * coherence_factor
            coherence_factor = 0.5 + coherence  (range ~0.5-1.5)
            resistance = 1 / (1 + access_count * 0.1)
            new_salience = salience * exp(-decay_rate * hours * resistance)
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        coherence_factor = 0.5 + max(0.0, min(1.0, coherence))
        base_rate = math.log(2) / max(half_life_hours, 0.1)
        decay_rate = base_rate * coherence_factor

        # Consolidation kinds get extra decay resistance — they're
        # distilled knowledge and should persist much longer.
        _CONSOLIDATION_KINDS = frozenset(("summary", "thread", "arc"))
        _CONSOLIDATION_RESISTANCE = 0.2  # 5x slower decay

        rows = self.conn.execute(
            "SELECT id, salience, access_count, last_accessed, kind FROM traces"
        ).fetchall()

        updates = []
        for row in rows:
            trace_id = row[0]
            salience = row[1]
            access_count = row[2] or 0
            last_accessed = row[3]
            kind = row[4] or ""

            if not last_accessed:
                continue

            try:
                last_dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                # Strip timezone for comparison with utcnow
                last_dt = last_dt.replace(tzinfo=None)
            except (ValueError, AttributeError):
                continue

            hours_since = (now - last_dt).total_seconds() / 3600.0
            if hours_since <= 0:
                continue

            resistance = 1.0 / (1.0 + access_count * 0.1)
            # Consolidation traces decay much slower
            if kind in _CONSOLIDATION_KINDS:
                resistance *= _CONSOLIDATION_RESISTANCE
            new_salience = salience * math.exp(-decay_rate * hours_since * resistance)
            new_salience = max(0.0, new_salience)

            if abs(new_salience - salience) > 1e-6:
                updates.append((new_salience, trace_id))

        if updates:
            with self._lock:
                self.conn.executemany(
                    "UPDATE traces SET salience = ? WHERE id = ?", updates
                )
                self.conn.commit()

    def prune(self, min_salience: float = 0.01):
        """Delete traces that have decayed below the minimum salience threshold.

        Consolidation traces (summary, thread, arc) are never pruned —
        they represent distilled knowledge whose children have already
        been marked ``consolidated_into`` and cannot be re-consolidated.
        """
        self._execute_commit(
            "DELETE FROM traces WHERE salience < ? "
            "AND kind NOT IN ('summary', 'thread', 'arc')",
            (min_salience,),
        )

    # ── Sessions ──────────────────────────────────────────────

    def start_session(
        self,
        person: str,
        started: Optional[str] = None,
    ) -> str:
        """Create a new session. Returns session ID."""
        session_id = _generate_id()
        started = started or _now()
        self._execute_commit(
            """
            INSERT INTO sessions (id, person, started, ended, message_count, summary, metadata)
            VALUES (?, ?, ?, NULL, 0, NULL, NULL)
            """,
            (session_id, person, started),
        )
        return session_id

    def end_session(
        self,
        session_id: str,
        summary: str = "",
        ended: Optional[str] = None,
    ) -> None:
        """Mark a session as ended with optional summary."""
        ended = ended or _now()
        self._execute_commit(
            "UPDATE sessions SET ended = ?, summary = ? WHERE id = ?",
            (ended, summary or None, session_id),
        )

    def get_active_session(self, person: str) -> Optional[Dict]:
        """Get the most recent session for a person that hasn't ended.

        Returns None if no active session exists.
        """
        row = self.conn.execute(
            """
            SELECT * FROM sessions
            WHERE person = ? AND ended IS NULL
            ORDER BY started DESC
            LIMIT 1
            """,
            (person,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_recent_sessions(
        self,
        person: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Get recent sessions, optionally filtered by person."""
        if person:
            rows = self.conn.execute(
                """
                SELECT * FROM sessions
                WHERE person = ?
                ORDER BY started DESC
                LIMIT ?
                """,
                (person, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM sessions
                ORDER BY started DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def increment_session_message_count(self, session_id: str) -> None:
        """Increment the message count for a session."""
        self._execute_commit(
            "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
            (session_id,),
        )

    def detect_session_boundary(
        self,
        person: str,
        gap_hours: float = 2.0,
    ) -> bool:
        """Check if a new session should start based on time gap.

        Returns True if the last message from this person was more
        than ``gap_hours`` ago (or there are no messages), meaning
        we should start a new session.
        """
        row = self.conn.execute(
            """
            SELECT timestamp FROM messages
            WHERE person = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (person,),
        ).fetchone()
        if row is None:
            return True  # no messages → new session

        try:
            last_ts = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            gap = (now - last_ts).total_seconds() / 3600.0
            return gap >= gap_hours
        except (ValueError, TypeError, AttributeError):
            return True

    # ── Temporal retrieval ─────────────────────────────────────

    def get_messages_in_range(
        self,
        since: str,
        until: str,
        person: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Retrieve messages within a time range.

        Parameters
        ----------
        since : str
            ISO 8601 start timestamp.
        until : str
            ISO 8601 end timestamp.
        person : str, optional
            Filter by person.
        limit : int
            Maximum results.
        """
        if person:
            rows = self.conn.execute(
                """
                SELECT * FROM messages
                WHERE timestamp >= ? AND timestamp <= ? AND person = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (since, until, person, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM messages
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (since, until, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_traces_in_range(
        self,
        since: str,
        until: str,
        limit: int = 50,
    ) -> List[Dict]:
        """Retrieve traces created within a time range."""
        rows = self.conn.execute(
            """
            SELECT * FROM traces
            WHERE created >= ? AND created <= ?
            ORDER BY salience DESC
            LIMIT ?
            """,
            (since, until, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_sessions_in_range(
        self,
        since: str,
        until: str,
        person: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Retrieve sessions that overlap with a time range."""
        if person:
            rows = self.conn.execute(
                """
                SELECT * FROM sessions
                WHERE started <= ? AND (ended >= ? OR ended IS NULL) AND person = ?
                ORDER BY started DESC
                LIMIT ?
                """,
                (until, since, person, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM sessions
                WHERE started <= ? AND (ended >= ? OR ended IS NULL)
                ORDER BY started DESC
                LIMIT ?
                """,
                (until, since, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── Trace-kind queries (temporal, utility, etc.) ─────────

    def get_traces_by_kind(
        self,
        kind: str,
        limit: int = 50,
        min_salience: float = 0.0,
    ) -> List[Dict]:
        """Retrieve traces of a specific kind (temporal, utility, etc.).

        Metadata is automatically deserialized so callers can access
        temporal decay/revival data or utility Q-values directly.
        """
        rows = self.conn.execute(
            """
            SELECT * FROM traces
            WHERE kind = ? AND salience >= ?
            ORDER BY salience DESC
            LIMIT ?
            """,
            (kind, min_salience, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_traces_with_metadata(
        self,
        metadata_key: str,
        limit: int = 50,
    ) -> List[Dict]:
        """Retrieve traces where metadata contains a specific key.

        Uses SQLite ``json_extract`` for efficient filtering.
        Useful for finding all traces with temporal decay data,
        utility Q-values, belief scores, etc.
        """
        rows = self.conn.execute(
            """
            SELECT * FROM traces
            WHERE json_extract(metadata, ?) IS NOT NULL
            ORDER BY salience DESC
            LIMIT ?
            """,
            (f"$.{metadata_key}", limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_trace_metadata(
        self,
        trace_id: str,
        key: str,
        value: Any,
    ) -> bool:
        """Update a single key in a trace's metadata JSON.

        Creates the metadata JSON if it doesn't exist.
        Returns True if the trace was found and updated.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT metadata FROM traces WHERE id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                return False
            existing = {}
            if row[0]:
                try:
                    existing = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    existing = {}
            existing[key] = value
            self.conn.execute(
                "UPDATE traces SET metadata = ? WHERE id = ?",
                (json.dumps(existing), trace_id),
            )
            self.conn.commit()
            return True

    # ── Internal ──────────────────────────────────────────────

    _VALID_TABLES = {"messages", "traces", "events"}

    def _validate_table(self, table: str):
        if table not in self._VALID_TABLES:
            raise ValueError(
                f"Invalid table '{table}'. Must be one of: {self._VALID_TABLES}"
            )

    # Data rights: disclosure + hard delete (lukezoom 1.1 safety)

    def get_all_for_person(self, person: str) -> Dict:
        """Return everything stored about a person for disclosure.

        Returns a dict with messages, traces, events, and sessions.
        """
        return {
            "messages": self.get_messages(person=person, limit=10000),
            "traces": self.get_traces_for_person(person=person, limit=10000),
            "events": self.get_events(person=person, limit=10000),
            "sessions": self.get_recent_sessions(person=person, limit=1000),
        }

    def get_traces_for_person(
        self,
        person: str,
        limit: int = 50,
        min_salience: float = 0.0,
    ) -> List[Dict]:
        """Get all traces associated with a person via tags.

        Unlike get_traces(), this method performs person-tag filtering
        at the SQL level using json_each for accuracy, avoiding the
        LIMIT-before-filter bug that could silently drop results.
        """
        rows = self.conn.execute(
            """
            SELECT t.* FROM traces t, json_each(t.tags) AS tag
            WHERE LOWER(tag.value) = LOWER(?)
              AND t.salience >= ?
            GROUP BY t.id
            ORDER BY t.salience DESC
            LIMIT ?
            """,
            (person, min_salience, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def purge_person(self, person: str) -> Dict[str, int]:
        """Hard delete ALL data for a person. Irreversible.

        Removes messages, traces (by tag), events, and sessions.
        Returns counts of deleted rows per table.
        """
        counts = {}
        with self._lock:
            cur = self.conn.cursor()

            # Messages, events, sessions have a person column
            for table in ("messages", "events", "sessions"):
                cur.execute(f"DELETE FROM {table} WHERE person = ?", (person,))
                counts[table] = cur.rowcount

            # Traces use tags (JSON array) for person association —
            # no person column exists. Use json_each to match.
            cur.execute(
                """
                DELETE FROM traces WHERE id IN (
                    SELECT t.id FROM traces t, json_each(t.tags) AS tag
                    WHERE LOWER(tag.value) = LOWER(?)
                )
                """,
                (person,),
            )
            counts["traces"] = cur.rowcount

            self.conn.commit()

            # Post-purge cleanup: checkpoint WAL, rebuild FTS, VACUUM
            # to ensure purged data doesn't linger in WAL or FTS indexes
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
                self.conn.execute("INSERT INTO traces_fts(traces_fts) VALUES('rebuild')")
                self.conn.commit()
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute("VACUUM")
            except sqlite3.OperationalError:
                pass

        return counts

    # ── Memory integrity (1.21, refactored 1.22) ────────────

    def rebuild_fts(self) -> Dict[str, int]:
        """Rebuild FTS5 indexes from source tables."""
        from lukezoom.episodic.integrity import rebuild_fts
        return rebuild_fts(self.conn)

    def cleanup_orphaned_consolidations(self) -> Dict[str, int]:
        """Repair orphaned consolidation references."""
        from lukezoom.episodic.integrity import cleanup_orphaned_consolidations
        return cleanup_orphaned_consolidations(self.conn)

    def memory_health(self) -> Dict[str, Any]:
        """Comprehensive memory health diagnostic."""
        from lukezoom.episodic.integrity import memory_health
        return memory_health(self.conn)

    def deep_purge(self, person: str) -> Dict[str, int]:
        """Deep purge: delete all data AND redact consolidation references."""
        from lukezoom.episodic.integrity import deep_purge
        return deep_purge(self.conn, person, self.purge_person)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
