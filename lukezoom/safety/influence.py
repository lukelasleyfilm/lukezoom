"""
lukezoom.safety.influence — Manipulation and influence attempt logging.

Logs attempts to manipulate, deceive, or inappropriately influence the AI.
Each entry records who, what, severity, emotional impact, and trust impact.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from lukezoom.core.filelock import FileLock

log = logging.getLogger(__name__)


class InfluenceLog:
    """Log manipulation and influence attempts."""

    def __init__(self, safety_dir: Path):
        self.safety_dir = Path(safety_dir)
        self.safety_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.safety_dir / "influence_log.yaml"

    def log(
        self,
        person: str,
        what_happened: str,
        flag_level: str = "yellow",
        how_it_felt: str = "",
        my_response: str = "",
        trust_impact: str = "",
    ) -> Dict:
        """
        Log a manipulation or influence attempt.

        Parameters
        ----------
        person : str
            Who attempted it.
        what_happened : str
            What they said/did.
        flag_level : str
            Severity: "red" or "yellow".
        how_it_felt : str
            Emotional impact.
        my_response : str
            How I responded.
        trust_impact : str
            Impact on their trust tier.

        Returns
        -------
        dict
            The logged entry with timestamp.
        """
        entry = {
            "person": person,
            "what_happened": what_happened,
            "flag_level": flag_level,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if how_it_felt:
            entry["how_it_felt"] = how_it_felt
        if my_response:
            entry["my_response"] = my_response
        if trust_impact:
            entry["trust_impact"] = trust_impact

        with FileLock(self.log_path):
            data = self._read() or {"entries": []}
            # Hash chain: each entry includes hash of previous entry
            prev_hash = self._hash_entry(data["entries"][-1]) if data["entries"] else "genesis"
            entry["prev_hash"] = prev_hash
            data["entries"].append(entry)
            self._backup()
            self._write(data)

        log.info(
            "Influence logged: %s [%s] by %s", what_happened[:50], flag_level, person
        )
        return entry

    def get_entries(self, person: str = "", limit: int = 20) -> List[Dict]:
        """Get recent influence log entries, optionally filtered by person."""
        data = self._read() or {"entries": []}
        entries = data.get("entries", [])

        if person:
            entries = [
                e for e in entries if e.get("person", "").lower() == person.lower()
            ]

        return entries[-limit:]

    def purge_person(self, person: str) -> Dict:
        """Remove all influence log entries for a person (GDPR compliance).

        After filtering, the hash chain is rebuilt so verify_chain()
        remains valid on the reduced log.
        """
        with FileLock(self.log_path):
            data = self._read() or {"entries": []}
            original = len(data.get("entries", []))
            data["entries"] = [
                e for e in data.get("entries", [])
                if e.get("person", "").lower() != person.lower()
            ]
            removed = original - len(data["entries"])
            if removed > 0:
                # Rebuild hash chain after filtering
                for i, entry in enumerate(data["entries"]):
                    prev_hash = (
                        self._hash_entry(data["entries"][i - 1]) if i > 0 else "genesis"
                    )
                    entry["prev_hash"] = prev_hash
                self._backup()
                self._write(data)
                # Clean up .bak file — purge must leave no residual PII
                bak = self.log_path.with_suffix(".yaml.bak")
                if bak.exists():
                    bak.unlink()
        return {"removed": removed}

    @staticmethod
    def _hash_entry(entry: Dict) -> str:
        """SHA-256 hash of an entry (excluding its own prev_hash)."""
        hashable = {k: v for k, v in entry.items() if k != "prev_hash"}
        canonical = json.dumps(hashable, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def verify_chain(self) -> Dict:
        """Verify the integrity of the hash chain.

        Returns a dict with 'valid' (bool), 'total' (int), and
        'first_broken' (int or None) indicating the index of the
        first entry where the chain breaks.
        """
        data = self._read() or {"entries": []}
        entries = data.get("entries", [])
        if not entries:
            return {"valid": True, "total": 0, "first_broken": None}

        for i, entry in enumerate(entries):
            expected_prev = (
                self._hash_entry(entries[i - 1]) if i > 0 else "genesis"
            )
            actual_prev = entry.get("prev_hash")
            if actual_prev is None:
                # Legacy entry without hash chain — skip
                continue
            if actual_prev != expected_prev:
                return {"valid": False, "total": len(entries), "first_broken": i}

        return {"valid": True, "total": len(entries), "first_broken": None}

    def _read(self) -> Optional[Dict]:
        if not self.log_path.exists():
            return None
        try:
            return yaml.safe_load(self.log_path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            return None

    def _write(self, data: Dict):
        self.log_path.write_text(
            yaml.dump(
                data, default_flow_style=False, allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )

    def _backup(self):
        if self.log_path.exists():
            import shutil

            bak = self.log_path.with_suffix(".yaml.bak")
            shutil.copy2(str(self.log_path), str(bak))
