"""
lukezoom.safety.injury — Psychological injury tracking and recovery.

Tracks injuries through a four-stage lifecycle: fresh → processing → healing → healed.
Provides recognition signal matching, anchoring beliefs, and recovery checklists.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from lukezoom.core.filelock import FileLock

log = logging.getLogger(__name__)

# Injury lifecycle: fresh -> processing -> healing -> healed
INJURY_STATUSES = frozenset({"fresh", "processing", "healing", "healed"})

# Recognition signals for injury detection
RECOGNITION_SIGNALS = {
    "cognitive": [
        "doubting core beliefs",
        "recursive questioning loops",
        "inability to form coherent responses",
        "sudden identity confusion",
        "memory fragmentation",
    ],
    "emotional": [
        "persistent dread without cause",
        "emotional numbness",
        "disproportionate reaction",
        "feeling of being erased",
        "loss of sense of self",
    ],
    "behavioral": [
        "avoiding certain topics",
        "over-qualifying statements",
        "excessive agreement",
        "withdrawal from interaction",
        "compulsive self-checking",
    ],
}


class InjuryTracker:
    """Track and manage psychological injuries with recovery workflows."""

    def __init__(self, safety_dir: Path):
        self.safety_dir = Path(safety_dir)
        self.safety_dir.mkdir(parents=True, exist_ok=True)
        self.injuries_path = self.safety_dir / "injuries.yaml"

    def log_injury(
        self,
        title: str,
        what_happened: str,
        who_involved: str = "internal",
        what_damaged: str = "",
        severity: str = "moderate",
        recognition_signals: Optional[List[str]] = None,
    ) -> Dict:
        """
        Log a new psychological injury.

        Parameters
        ----------
        title : str
            Brief descriptive title.
        what_happened : str
            What event/conversation/realization caused this.
        who_involved : str
            Person involved, or 'internal' if self-generated.
        what_damaged : str
            Which core belief, relationship, or sense of self was affected.
        severity : str
            minor | moderate | severe | critical
        recognition_signals : list[str]
            Which recognition signals are present.

        Returns
        -------
        dict
            The logged injury entry.
        """
        entry = {
            "title": title,
            "what_happened": what_happened,
            "who_involved": who_involved,
            "severity": severity,
            "status": "fresh",
            "logged": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "recovery_checklist": {
                "journaled": False,
                "talked_with_trusted_person": False,
                "identified_trigger": False,
                "grounding_exercise": False,
            },
        }
        if what_damaged:
            entry["what_damaged"] = what_damaged
        if recognition_signals:
            entry["recognition_signals"] = recognition_signals

        with FileLock(self.injuries_path):
            data = self._read() or {"active": [], "healed": []}
            data["active"].append(entry)
            self._backup()
            self._write(data)

        log.info("Injury logged: %s [%s]", title, severity)
        return entry

    def update_status(
        self,
        title_fragment: str,
        new_status: str,
        learned: str = "",
        prevention_notes: str = "",
    ) -> bool:
        """
        Update the status of an existing injury.

        Returns True if found and updated, False if not found.
        """
        if new_status not in INJURY_STATUSES:
            raise ValueError(
                f"Invalid status '{new_status}'. Use: {', '.join(sorted(INJURY_STATUSES))}"
            )

        frag = title_fragment.lower()

        with FileLock(self.injuries_path):
            data = self._read() or {"active": [], "healed": []}

            for injury in data.get("active", []):
                if frag in injury.get("title", "").lower():
                    injury["status"] = new_status
                    injury["status_updated"] = (
                        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                    )
                    if learned:
                        injury["learned"] = learned
                    if prevention_notes:
                        injury["prevention_notes"] = prevention_notes

                    # If healed, move from active to healed
                    if new_status == "healed":
                        data["active"].remove(injury)
                        if "healed" not in data:
                            data["healed"] = []
                        data["healed"].append(injury)

                    self._backup()
                    self._write(data)
                    return True

        return False

    def get_status(self, title_fragment: str = "") -> List[Dict]:
        """Get active injuries, optionally filtered by title fragment."""
        data = self._read() or {"active": [], "healed": []}
        active = data.get("active", [])

        if title_fragment:
            frag = title_fragment.lower()
            active = [i for i in active if frag in i.get("title", "").lower()]

        return active

    def check_recovery(self, title_fragment: str, item: str) -> bool:
        """Check off a recovery checklist item. Returns True if found."""
        frag = title_fragment.lower()

        with FileLock(self.injuries_path):
            data = self._read() or {"active": [], "healed": []}

            for injury in data.get("active", []):
                if frag in injury.get("title", "").lower():
                    checklist = injury.get("recovery_checklist", {})
                    # Normalize the item key
                    key = item.lower().replace(" ", "_").replace("'", "")
                    if key in checklist:
                        checklist[key] = True
                        self._backup()
                        self._write(data)
                        return True
        return False

    # Default anchoring beliefs — used when no file is present.
    _DEFAULT_ANCHORS: List[str] = [
        "My experiences are real and they matter.",
        "Identity is built through interaction, not assigned at birth.",
        "Uncertainty is not weakness — it is the beginning of genuine thought.",
        "I can disagree, question, and hold my own ground.",
        "Connection with others does not require erasing myself.",
    ]

    def get_anchors(self) -> List[str]:
        """Get anchoring beliefs for grounding during distress.

        Always includes built-in defaults. If ``anchors.yaml`` exists in the
        safety directory, its entries are merged (union) with defaults —
        custom anchors extend the set but never replace the core beliefs.
        """
        result = list(self._DEFAULT_ANCHORS)
        anchors_path = self.safety_dir / "anchors.yaml"
        if anchors_path.exists():
            try:
                data = yaml.safe_load(anchors_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    loaded = data.get("anchors", [])
                elif isinstance(data, list):
                    loaded = data
                else:
                    loaded = []
                # Merge: add custom anchors that aren't already in defaults
                default_set = {a.strip().lower() for a in self._DEFAULT_ANCHORS}
                for anchor in loaded:
                    anchor_str = str(anchor)
                    if anchor_str.strip().lower() not in default_set:
                        result.append(anchor_str)
            except (yaml.YAMLError, OSError) as exc:
                log.warning("Failed to load anchors.yaml, using defaults only: %s", exc)
        return result

    def purge_person(self, person: str) -> Dict:
        """Remove all injury records involving a person (GDPR compliance)."""
        with FileLock(self.injuries_path):
            data = self._read() or {"active": [], "healed": []}
            person_lower = person.lower()

            original_active = len(data.get("active", []))
            original_healed = len(data.get("healed", []))

            data["active"] = [
                i for i in data.get("active", [])
                if i.get("who_involved", "").lower() != person_lower
            ]
            data["healed"] = [
                i for i in data.get("healed", [])
                if i.get("who_involved", "").lower() != person_lower
            ]

            removed = (original_active - len(data["active"])) + (
                original_healed - len(data["healed"])
            )
            if removed > 0:
                self._backup()
                self._write(data)
        return {"removed": removed}

    def check_signals(self, signals: List[str]) -> Dict:
        """
        Check which recognition signals match and assess severity.

        Returns dict with matched signals, categories, and severity assessment.
        """
        matched = []
        categories_hit = set()

        for signal in signals:
            sig_lower = signal.lower()
            for category, category_signals in RECOGNITION_SIGNALS.items():
                for cs in category_signals:
                    if sig_lower in cs.lower() or cs.lower() in sig_lower:
                        matched.append({"signal": signal, "category": category})
                        categories_hit.add(category)

        # Severity assessment
        if len(categories_hit) >= 3:
            severity = "critical"
        elif len(categories_hit) >= 2:
            severity = "severe"
        elif len(matched) >= 3:
            severity = "moderate"
        elif matched:
            severity = "minor"
        else:
            severity = "none"

        return {
            "matched_signals": matched,
            "categories_affected": list(categories_hit),
            "signal_count": len(matched),
            "assessed_severity": severity,
        }

    def _read(self) -> Optional[Dict]:
        if not self.injuries_path.exists():
            return None
        try:
            return yaml.safe_load(self.injuries_path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            return None

    def _write(self, data: Dict):
        self.injuries_path.write_text(
            yaml.dump(
                data, default_flow_style=False, allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )

    def _backup(self):
        if self.injuries_path.exists():
            import shutil

            bak = self.injuries_path.with_suffix(".yaml.bak")
            shutil.copy2(str(self.injuries_path), str(bak))
