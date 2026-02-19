"""
lukezoom.search.indexed — SQLite FTS5 keyword search.

Provides full-text search across messages and traces stored in the
episodic database.  Connects to the same SQLite DB that
``EpisodicStore`` manages — the FTS5 virtual tables and sync triggers
are created there.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


class IndexedSearch:
    """Keyword search over the episodic SQLite database using FTS5."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    # Public API

    def search(
        self,
        query: str,
        memory_type: Optional[str] = None,
        person: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Search messages and/or traces via FTS5.

        Parameters
        ----------
        query:
            FTS5 match expression (plain terms work fine).
        memory_type:
            ``"messages"`` or ``"traces"`` to restrict search.
            ``None`` searches both.
        person:
            Filter results to this person (messages only).
        limit:
            Maximum number of results per source type.

        Returns
        -------
        list[dict]
            Combined results sorted by FTS rank (lower is better).
            Each dict includes a ``"source"`` key (``"messages"`` or
            ``"traces"``).
        """
        if not query or not query.strip():
            return []

        # Sanitise query for FTS5 — wrap each word in quotes to avoid
        # syntax errors from special characters.
        safe_query = self._sanitise_query(query)

        results: List[Dict] = []

        if memory_type in (None, "messages"):
            results.extend(
                self._search_messages(safe_query, person=person, limit=limit)
            )

        if memory_type in (None, "traces"):
            results.extend(self._search_traces(safe_query, limit=limit))

        # Sort by rank (FTS5 rank is negative; more negative = better match)
        results.sort(key=lambda r: r.get("rank", 0))

        return results[:limit]

    # Internal search helpers

    def _search_messages(
        self,
        query: str,
        person: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        try:
            if person:
                rows = self.conn.execute(
                    """
                    SELECT m.*, fts.rank
                    FROM messages m
                    JOIN messages_fts fts ON m.rowid = fts.rowid
                    WHERE messages_fts MATCH ? AND m.person = ?
                    ORDER BY fts.rank
                    LIMIT ?
                    """,
                    (query, person, limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT m.*, fts.rank
                    FROM messages m
                    JOIN messages_fts fts ON m.rowid = fts.rowid
                    WHERE messages_fts MATCH ?
                    ORDER BY fts.rank
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            # FTS table may not exist yet or query may be malformed
            return []

        return [self._row_to_result(row, source="messages") for row in rows]

    def _search_traces(self, query: str, limit: int = 20) -> List[Dict]:
        try:
            rows = self.conn.execute(
                """
                SELECT t.*, fts.rank
                FROM traces t
                JOIN traces_fts fts ON t.rowid = fts.rowid
                WHERE traces_fts MATCH ?
                ORDER BY fts.rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        return [self._row_to_result(row, source="traces") for row in rows]

    # Helpers

    @staticmethod
    def _sanitise_query(query: str) -> str:
        """Wrap each term in double-quotes for safe FTS5 matching."""
        terms = query.strip().split()
        if not terms:
            return '""'
        # Quote each term, strip any existing quotes
        quoted = [f'"{t.strip(chr(34))}"' for t in terms if t.strip('"')]
        return " ".join(quoted)

    @staticmethod
    def _row_to_result(row: sqlite3.Row, source: str) -> Dict:
        """Convert a sqlite3.Row to a dict with JSON deserialization."""
        d = dict(row)
        for key in ("signal", "metadata", "tags"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        d["source"] = source
        return d

    # Lifecycle

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "IndexedSearch":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
