"""
Identity resolver — maps aliases to canonical names.

People go by many names: Discord handles, nicknames, typos.
This ensures "alice_dev", "Alice", and "alice" all resolve
to the same person record.
"""

import unicodedata
import yaml
from pathlib import Path
from typing import Optional, Dict, List


class IdentityResolver:
    """
    Bidirectional alias resolver backed by identities.yaml.

    File format:
        people:
          alice:
            aliases: [alice_dev, Alice]
            discord_id: "123456789"
            trust_tier: friend
          bob:
            aliases: [bobby, Robert]
            trust_tier: acquaintance
    """

    def __init__(self, identities_path: Path):
        self.path = Path(identities_path)
        self._data: Dict = {}
        self._lookup: Dict[str, str] = {}  # lowercase alias -> canonical
        self._load()

    # ── Public API ────────────────────────────────────────────

    def resolve(self, alias: str) -> str:
        """
        Convert any alias to a canonical name (case-insensitive).
        Returns the alias itself lowercased if no mapping is found.
        Returns empty string for None/empty input.
        """
        if not alias:
            return alias or ""
        normalized = unicodedata.normalize("NFKC", alias).lower()
        return self._lookup.get(normalized, normalized)

    def get_person(self, canonical: str) -> Optional[Dict]:
        """Get the full person record by canonical name."""
        people = self._data.get("people", {})
        record = people.get(canonical.lower())
        if record is None:
            return None
        # Return a copy with the canonical name included
        result = dict(record)
        result["name"] = canonical.lower()
        return result

    def list_people(self) -> List[Dict]:
        """List all known people with their aliases and metadata."""
        people = self._data.get("people", {})
        results = []
        for name, record in people.items():
            entry = dict(record)
            entry["name"] = name
            results.append(entry)
        return results

    def add_alias(self, person: str, alias: str):
        """Add a new alias for an existing person."""
        people = self._data.get("people", {})
        canonical = person.lower()

        if canonical not in people:
            raise KeyError(
                f"Person '{canonical}' not found. "
                f"Use add_person() to create them first."
            )

        aliases = people[canonical].get("aliases", [])
        alias_lower = alias.lower()

        # Check for conflicts
        if alias_lower in self._lookup:
            existing = self._lookup[alias_lower]
            if existing != canonical:
                raise ValueError(f"Alias '{alias}' already maps to '{existing}'")
            return  # Already mapped to this person, no-op

        aliases.append(alias)
        people[canonical]["aliases"] = aliases
        self._save()
        self._build_lookup()

    def add_person(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        trust_tier: str = "acquaintance",
        **kwargs,
    ):
        """Add a new person to the identity database."""
        if "people" not in self._data:
            self._data["people"] = {}

        canonical = name.lower()
        if canonical in self._data["people"]:
            raise ValueError(f"Person '{canonical}' already exists")

        # Check alias conflicts
        for alias in aliases or []:
            if alias.lower() in self._lookup:
                existing = self._lookup[alias.lower()]
                raise ValueError(f"Alias '{alias}' already maps to '{existing}'")

        record: Dict = {
            "aliases": aliases or [],
            "trust_tier": trust_tier,
        }
        record.update(kwargs)

        self._data["people"][canonical] = record
        self._save()
        self._build_lookup()

    # ── Internal ──────────────────────────────────────────────

    def _load(self):
        """Load identities from YAML file."""
        if self.path.exists():
            try:
                text = self.path.read_text(encoding="utf-8")
                self._data = yaml.safe_load(text) or {}
            except (yaml.YAMLError, OSError):
                self._data = {}
        else:
            self._data = {"people": {}}
        self._build_lookup()

    def _save(self):
        """Write identities back to YAML file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.dump(
                self._data,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def _build_lookup(self):
        """Build the reverse alias -> canonical name lookup dict."""
        self._lookup = {}
        people = self._data.get("people", {})

        for canonical, record in people.items():
            canonical_lower = canonical.lower()

            # The canonical name itself is an alias
            norm_canonical = unicodedata.normalize("NFKC", canonical_lower)
            self._lookup[norm_canonical] = norm_canonical

            # All explicit aliases
            for alias in record.get("aliases", []):
                norm_alias = unicodedata.normalize("NFKC", alias).lower()
                self._lookup[norm_alias] = norm_canonical
