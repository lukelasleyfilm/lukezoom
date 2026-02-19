"""
lukezoom.procedural.store — Procedural memory (skills & processes).

Skills are stored as individual Markdown files in a directory.
Search is keyword-based — no embeddings needed for the small
number of skills a single identity typically accumulates.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional


class ProceduralStore:
    """Filesystem-backed store for procedural skills.

    Each skill is a ``.md`` file under ``skills_dir``.  The filename
    (without extension) is the skill name; the first non-empty line is
    treated as a short description.
    """

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    # CRUD

    def list_skills(self) -> List[Dict]:
        """Return all skills with ``name`` and ``description``."""
        skills: List[Dict] = []
        for path in sorted(self.skills_dir.glob("*.md")):
            name = path.stem
            description = self._first_line(path)
            skills.append({"name": name, "description": description, "path": str(path)})
        return skills

    def get_skill(self, name: str) -> Optional[str]:
        """Read a skill file by name.  Returns ``None`` if not found."""
        safe_name = self._sanitize_name(name)
        path = self.skills_dir / f"{safe_name}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def add_skill(self, name: str, content: str) -> None:
        """Write (or overwrite) a skill file."""
        safe_name = self._sanitize_name(name)
        path = self.skills_dir / f"{safe_name}.md"
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Normalize a skill name to a safe, lowercase filename stem.

        Strips whitespace, lowercases, replaces non-word chars with
        underscores, and collapses consecutive underscores.
        """
        safe = re.sub(r"[^\w\-]", "_", name.strip().lower())
        safe = re.sub(r"_+", "_", safe).strip("_")
        return safe or "unnamed"

    # Search

    def search_skills(self, query: str) -> List[Dict]:
        """Keyword search across skill names and content.

        Returns matching skills sorted by relevance (number of keyword
        hits, descending).  Each result includes ``name``,
        ``description``, ``hits``, and ``path``.
        """
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        results: List[Dict] = []
        for path in self.skills_dir.glob("*.md"):
            name = path.stem
            try:
                content = path.read_text(encoding="utf-8").lower()
            except OSError:
                continue

            name_lower = name.lower()
            searchable = name_lower + " " + content
            hits = sum(searchable.count(kw) for kw in keywords)

            if hits > 0:
                results.append(
                    {
                        "name": name,
                        "description": self._first_line(path),
                        "hits": hits,
                        "path": str(path),
                    }
                )

        results.sort(key=lambda r: r["hits"], reverse=True)
        return results

    def match_context(self, message: str) -> List[str]:
        """Find skills relevant to a user message.

        Simple heuristic: check whether any skill name or a keyword
        from the skill's first line appears in the message.  Returns
        the full content of each matching skill.
        """
        msg_lower = message.lower()
        matched: List[str] = []

        for path in self.skills_dir.glob("*.md"):
            name_lower = path.stem.lower()
            description_lower = self._first_line(path).lower()

            # Match on skill name appearing in message
            if name_lower in msg_lower:
                content = self._safe_read(path)
                if content is not None:
                    matched.append(content)
                continue

            # Match on description keywords
            desc_keywords = self._extract_keywords(description_lower)
            for kw in desc_keywords:
                if len(kw) >= 4 and kw in msg_lower:
                    content = self._safe_read(path)
                    if content is not None:
                        matched.append(content)
                    break

        return matched

    # Helpers

    @staticmethod
    def _first_line(path: Path) -> str:
        """Return the first non-empty, non-heading line as a description."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip().lstrip("# ").strip()
                    if stripped:
                        return stripped
        except OSError:
            pass
        return ""

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Split text into lowercase keywords, filtering stop-words."""
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "about",
            "like",
            "through",
            "after",
            "over",
            "between",
            "out",
            "against",
            "during",
            "without",
            "before",
            "under",
            "around",
            "among",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "every",
            "all",
            "any",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "because",
            "if",
            "when",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
        }
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [w for w in words if w not in stop and len(w) >= 2]

    @staticmethod
    def _safe_read(path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None
