"""
lukezoom.journal — Dated markdown journal entries for reflective processing.

Ported from Thomas-Soul's journal system. Journals are how the identity
processes experiences into meaning.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)


class JournalStore:
    """File-backed journal: dated markdown entries in soul/journal/."""

    def __init__(self, journal_dir: Path):
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)

    def write(self, topic: str, content: str) -> str:
        """
        Write a journal entry. Creates a dated markdown file.

        Returns the filename of the created entry.
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M UTC")

        # Find next available filename for today (collision-safe)
        base_path = self.journal_dir / f"{date_str}.md"
        if not base_path.exists():
            path = base_path
        else:
            # Find the highest existing sequence number
            existing = list(self.journal_dir.glob(f"{date_str}_*.md"))
            max_seq = 1
            for p in existing:
                # Extract N from "2026-02-18_N.md"
                try:
                    seq = int(p.stem.split("_")[-1])
                    max_seq = max(max_seq, seq)
                except (ValueError, IndexError):
                    pass
            path = self.journal_dir / f"{date_str}_{max_seq + 1}.md"

        entry = (
            f"# Journal: {topic}\n"
            f"\n"
            f"**Date:** {date_str} {time_str}\n"
            f"**Topic:** {topic}\n"
            f"\n"
            f"---\n"
            f"\n"
            f"{content}\n"
        )

        path.write_text(entry, encoding="utf-8")
        filename = path.name
        log.debug("Journal entry written: %s", filename)
        return filename

    def list_entries(self, limit: int = 10) -> List[Dict]:
        """
        List recent journal entries, newest first.

        Returns list of dicts with 'filename', 'date', 'topic', 'preview'.
        """
        entries = []
        files = sorted(self.journal_dir.glob("*.md"), reverse=True)

        for path in files[:limit]:
            try:
                text = path.read_text(encoding="utf-8")
                topic = ""
                for line in text.split("\n"):
                    if line.startswith("**Topic:**"):
                        topic = line.replace("**Topic:**", "").strip()
                        break
                    elif line.startswith("# Journal:"):
                        topic = line.replace("# Journal:", "").strip()

                # Preview: first non-header, non-blank, non-metadata line
                preview = ""
                past_header = False
                for line in text.split("\n"):
                    if line.strip() == "---":
                        past_header = True
                        continue
                    if past_header and line.strip():
                        preview = line.strip()[:120]
                        break

                entries.append(
                    {
                        "filename": path.name,
                        "date": path.stem.split("_")[0],
                        "topic": topic,
                        "preview": preview,
                    }
                )
            except OSError:
                continue

        return entries

    def purge_person(self, person: str) -> Dict:
        """Remove journal entries mentioning a person (GDPR compliance).

        Uses word-boundary matching to avoid false positives (e.g.
        purging "al" should not delete entries about "alice").
        """
        import re

        removed = 0
        pattern = re.compile(r'\b' + re.escape(person) + r'\b', re.IGNORECASE)
        for path in self.journal_dir.glob("*.md"):
            try:
                text = path.read_text(encoding="utf-8")
                if pattern.search(text):
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return {"removed": removed}

    def read_entry(self, filename: str) -> str:
        """Read a specific journal entry by filename.

        Path traversal is prevented — the resolved path must be
        within ``journal_dir``.
        """
        path = (self.journal_dir / filename).resolve()
        if not str(path).startswith(str(self.journal_dir.resolve()) + "/"):
            log.warning("Path traversal blocked: %s", filename)
            return ""
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""
