"""
Semantic memory store — structured knowledge persisted as
YAML configs and Markdown relationship files.

This is the "what I know" layer: identity, relationships,
preferences, boundaries, trust tiers, contradictions.
"""

import re
import unicodedata
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone

from lukezoom.core.filelock import FileLock


MAX_NAME_LENGTH = 200


def _sanitize_name(name: str) -> str:
    """Sanitize a person/entity name into a safe filename stem.

    Strips leading/trailing whitespace, lowercases, and replaces any
    character that isn't alphanumeric, hyphen, or underscore with ``_``.
    Truncates to MAX_NAME_LENGTH characters.
    Raises ValueError on empty result.
    """
    normalized = unicodedata.normalize("NFKC", name.strip().lower())
    safe = re.sub(r"[^\w\-]", "_", normalized)
    safe = re.sub(r"_+", "_", safe).strip("_")  # collapse repeated _
    if not safe:
        raise ValueError(f"Name {name!r} produces an empty safe filename")
    safe = safe[:MAX_NAME_LENGTH]
    return safe


class SemanticStore:
    """File-backed semantic memory: YAML + Markdown knowledge base."""

    def __init__(self, semantic_dir: Path, soul_dir: Path):
        self.semantic_dir = Path(semantic_dir)
        self.soul_dir = Path(soul_dir)
        self.relationships_dir = self.semantic_dir / "relationships"

        # Ensure directories exist
        self.semantic_dir.mkdir(parents=True, exist_ok=True)
        self.relationships_dir.mkdir(parents=True, exist_ok=True)

    # ── Identity ──────────────────────────────────────────────

    def get_identity(self) -> str:
        """Read the full SOUL.md identity document."""
        soul_path = self.soul_dir / "SOUL.md"
        return self._read_markdown(soul_path)

    # ── Relationships ─────────────────────────────────────────

    def get_relationship(self, person: str) -> Optional[str]:
        """Read a person's relationship file as raw markdown."""
        path = self.relationships_dir / f"{_sanitize_name(person)}.md"
        if not path.exists():
            return None
        return self._read_markdown(path)

    def list_relationships(self) -> List[Dict]:
        """List all known people with name and trust tier parsed from files."""
        results = []
        for path in sorted(self.relationships_dir.glob("*.md")):
            name = path.stem
            trust_tier = self._parse_trust_tier(path)
            results.append({"name": name, "trust_tier": trust_tier})
        return results

    def update_relationship(self, person: str, section: str, content: str):
        """
        Append content to a section in a person's relationship file.
        Creates the file with a basic template if it doesn't exist.
        Backs up before writing.
        """
        path = self.relationships_dir / f"{_sanitize_name(person)}.md"

        with FileLock(path):
            if not path.exists():
                self._create_relationship_template(path, person)
            self._backup(path)
            self._append_to_section(path, section, content)

    def add_fact(self, person: str, fact: str):
        """Add a fact to a person's relationship file under 'What I Know'."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = f"- {fact} ({timestamp})"
        self.update_relationship(person, "What I Know", entry)

    # ── Preferences ───────────────────────────────────────────

    def get_preferences(self) -> str:
        """Read preferences.yaml and return as formatted text."""
        data = self._read_yaml(self.semantic_dir / "preferences.yaml")
        if not data:
            return ""
        return self._format_preferences(data)

    def update_preferences(self, item: str, pref_type: str, reason: str):
        """
        Add to preferences.yaml under likes, dislikes, or uncertainties.

        pref_type: "like", "dislike", or "uncertainty"
        """
        # Normalize the section key (before lock to fail fast on bad input)
        section_map = {
            "like": "likes",
            "likes": "likes",
            "dislike": "dislikes",
            "dislikes": "dislikes",
            "uncertainty": "uncertainties",
            "uncertainties": "uncertainties",
        }
        section = section_map.get(pref_type.lower())
        if not section:
            raise ValueError(
                f"Invalid pref_type '{pref_type}'. Use: like, dislike, or uncertainty"
            )

        path = self.semantic_dir / "preferences.yaml"
        entry = {"item": item, "reason": reason}

        with FileLock(path):
            data = self._read_yaml(path) or {}
            if section not in data:
                data[section] = []
            data[section].append(entry)
            self._backup(path)
            self._write_yaml(path, data)

    # ── Boundaries ────────────────────────────────────────────

    def get_boundaries(self) -> str:
        """Read boundaries.yaml and return as formatted text."""
        data = self._read_yaml(self.semantic_dir / "boundaries.yaml")
        if not data:
            return ""
        return self._format_yaml_as_text(data, title="Boundaries")

    def add_boundary(self, category: str, boundary: str):
        """Add a boundary under a category (Identity, Safety, Interaction, Growth)."""
        path = self.semantic_dir / "boundaries.yaml"
        cat = category.strip()

        with FileLock(path):
            data = self._read_yaml(path) or {}
            if "categories" not in data:
                data["categories"] = {}
            if cat not in data["categories"]:
                data["categories"][cat] = []
            data["categories"][cat].append(
                {
                    "boundary": boundary,
                    "added": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                }
            )
            self._backup(path)
            self._write_yaml(path, data)

    # ── Trust ─────────────────────────────────────────────────

    def get_trust(self) -> Dict:
        """Read the full trust.yaml as a dict."""
        return self._read_yaml(self.semantic_dir / "trust.yaml") or {}

    def update_trust(self, person: str, tier: str, reason: str):
        """Update a person's trust tier with a reason."""
        path = self.semantic_dir / "trust.yaml"
        safe_person = _sanitize_name(person)

        with FileLock(path):
            data = self._read_yaml(path) or {}
            if "tiers" not in data:
                data["tiers"] = {}
            data["tiers"][safe_person] = {
                "tier": tier,
                "reason": reason,
                "updated": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
            self._backup(path)
            self._write_yaml(path, data)

    # ── Contradictions ────────────────────────────────────────

    def get_contradictions(self) -> str:
        """Read contradictions.yaml and return as formatted text."""
        data = self._read_yaml(self.semantic_dir / "contradictions.yaml")
        if not data:
            return ""
        return self._format_yaml_as_text(data, title="Contradictions")

    def add_contradiction(
        self, title: str, description: str, current_thinking: str = ""
    ):
        """Add a contradiction to contradictions.yaml."""
        path = self.semantic_dir / "contradictions.yaml"

        entry = {
            "title": title,
            "description": description,
            "added": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        if current_thinking:
            entry["current_thinking"] = current_thinking

        with FileLock(path):
            data = self._read_yaml(path) or {}
            if "active" not in data:
                data["active"] = []
            data["active"].append(entry)
            self._backup(path)
            self._write_yaml(path, data)

    def search_preferences(self, query: str) -> str:
        """Search preferences for matching items."""
        data = self._read_yaml(self.semantic_dir / "preferences.yaml")
        if not data:
            return "No preferences found."

        query_lower = query.lower()
        matches = []
        for section in ("likes", "dislikes", "uncertainties"):
            for item in data.get(section, []):
                if isinstance(item, dict):
                    text = f"{item.get('item', '')} {item.get('reason', '')}"
                else:
                    text = str(item)
                if query_lower in text.lower():
                    matches.append(f"[{section}] {text}")

        if not matches:
            return f"No preferences matching '{query}'."
        return "\n".join(matches)

    # ── Trust helpers ─────────────────────────────────────────

    def check_trust(self, person: str) -> Dict:
        """Check a person's trust tier and level."""
        trust_data = self.get_trust()
        tiers = trust_data.get("tiers", {})
        entry = tiers.get(_sanitize_name(person), {})

        tier = entry.get("tier", "stranger") if entry else "stranger"

        tier_levels = {
            "core": 5,
            "inner_circle": 4,
            "friend": 3,
            "acquaintance": 2,
            "stranger": 1,
        }
        level = tier_levels.get(tier.lower(), 0)

        return {
            "person": person,
            "tier": tier,
            "level": level,
            "reason": entry.get("reason", ""),
            "updated": entry.get("updated", ""),
        }

    def can_access(self, person: str, required_tier: str) -> bool:
        """Check if a person meets the required trust tier."""
        tier_levels = {
            "core": 5,
            "inner_circle": 4,
            "friend": 3,
            "acquaintance": 2,
            "stranger": 1,
        }
        info = self.check_trust(person)
        required_level = tier_levels.get(required_tier.lower(), 0)
        return info["level"] >= required_level

    def promote_trust(self, person: str, new_tier: str, reason: str):
        """Promote a person to a higher trust tier."""
        self.update_trust(person, new_tier, reason)

    # ── File I/O helpers ──────────────────────────────────────

    # ── Data rights: disclosure + hard delete (lukezoom 1.1 safety) ──

    def get_person_data(self, person: str) -> Dict:
        """Return all semantic data stored about a person."""
        safe = _sanitize_name(person)
        trust = self.get_trust()
        tier_info = trust.get("tiers", {}).get(safe, {})
        return {
            "relationship": self.get_relationship(person) or "",
            "trust_tier": tier_info.get("tier", "stranger"),
            "trust_reason": tier_info.get("reason", ""),
        }

    def purge_person(self, person: str) -> Dict[str, bool]:
        """Hard delete all semantic data for a person. Irreversible."""
        safe = _sanitize_name(person)
        result = {}

        # Delete relationship file and its backup
        rel_path = self.relationships_dir / f"{safe}.md"
        if rel_path.exists():
            rel_path.unlink()
            result["relationship_deleted"] = True
        else:
            result["relationship_deleted"] = False
        rel_bak = rel_path.with_suffix(rel_path.suffix + ".bak")
        if rel_bak.exists():
            rel_bak.unlink()

        # Remove from trust.yaml
        trust_path = self.semantic_dir / "trust.yaml"
        trust_data = self._read_yaml(trust_path) or {}
        tiers = trust_data.get("tiers", {})
        if safe in tiers:
            del tiers[safe]
            self._write_yaml(trust_path, trust_data)
            result["trust_deleted"] = True
        else:
            result["trust_deleted"] = False

        # Clean up any .bak files containing PII
        for bak in self.semantic_dir.glob("*.bak"):
            try:
                bak.unlink()
            except OSError:
                pass

        return result

    # ── Private helpers ───────────────────────────────────────

    def _backup(self, filepath: Path):
        """Copy file to filepath.bak before modification."""
        if filepath.exists():
            bak = filepath.with_suffix(filepath.suffix + ".bak")
            shutil.copy2(str(filepath), str(bak))

    def _read_yaml(self, path: Path) -> Optional[Dict]:
        """Safe YAML read. Returns None if file missing or invalid."""
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
            return yaml.safe_load(text) or {}
        except (yaml.YAMLError, OSError):
            return None

    def _write_yaml(self, path: Path, data: Dict):
        """Safe YAML write with human-friendly formatting."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.dump(
                data,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def _read_markdown(self, path: Path) -> str:
        """Safe file read. Returns empty string if missing."""
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _append_to_section(self, filepath: Path, section: str, content: str):
        """
        Find '## {section}' in a markdown file and append content
        after all existing content in that section (before the next
        heading or end of file).
        """
        text = filepath.read_text(encoding="utf-8")
        lines = text.split("\n")
        header = f"## {section}"

        # Find the section header
        section_idx = None
        for i, line in enumerate(lines):
            if line.strip().lower() == header.lower():
                section_idx = i
                break

        if section_idx is None:
            # Section doesn't exist — append it at the end
            lines.append("")
            lines.append(header)
            lines.append(content)
        else:
            # Find the end of this section (next ## or EOF)
            insert_idx = len(lines)
            for i in range(section_idx + 1, len(lines)):
                if lines[i].startswith("## "):
                    insert_idx = i
                    break
            # Insert content just before the next section
            lines.insert(insert_idx, content)

        filepath.write_text("\n".join(lines), encoding="utf-8")

    # ── Formatting helpers ────────────────────────────────────

    def _parse_trust_tier(self, path: Path) -> str:
        """Extract trust tier from a relationship markdown file."""
        try:
            text = path.read_text(encoding="utf-8")
            for line in text.split("\n"):
                lower = line.lower().strip()
                if "trust" in lower and ":" in lower:
                    # e.g., "**Trust Tier:** friend" or "Trust: inner_circle"
                    return line.split(":", 1)[1].strip().strip("*").strip()
        except OSError:
            pass
        return "unknown"

    def _create_relationship_template(self, path: Path, person: str):
        """Create a basic relationship file for a new person."""
        template = (
            f"# {person.title()}\n"
            f"\n"
            f"**Trust Tier:** acquaintance\n"
            f"**First Contact:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
            f"\n"
            f"## What I Know\n"
            f"\n"
            f"## Conversation History\n"
            f"\n"
            f"## Notes\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template, encoding="utf-8")

    def _format_preferences(self, data: Dict) -> str:
        """Format preferences dict into readable text."""
        sections = []
        for key in ("likes", "dislikes", "uncertainties"):
            items = data.get(key, [])
            if not items:
                continue
            section_lines = [f"## {key.title()}"]
            for item in items:
                if isinstance(item, dict):
                    line = f"- {item.get('item', '?')}"
                    reason = item.get("reason")
                    if reason:
                        line += f" — {reason}"
                    section_lines.append(line)
                else:
                    section_lines.append(f"- {item}")
            sections.append("\n".join(section_lines))
        return "\n\n".join(sections)

    def _format_yaml_as_text(self, data: Dict, title: str = "") -> str:
        """Generic YAML-to-readable-text formatter."""
        lines = []
        if title:
            lines.append(f"# {title}")
            lines.append("")
        lines.append(
            yaml.dump(
                data,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ).strip()
        )
        return "\n".join(lines)
