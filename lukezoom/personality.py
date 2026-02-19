"""
lukezoom.personality — Big Five personality system.

Ported from thomas-soul/thomas_core/personality.py.
Personality traits affect response modifiers (verbosity, depth,
emotional expression, etc.) and are injected as grounding context
in the before-pipeline.

Personality can evolve slowly over time via ``update_trait()``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("lukezoom.personality")


# BigFiveProfile


@dataclass
class BigFiveProfile:
    """Big Five personality traits, each 0.0 – 1.0.

    Default values are calibrated for Thomas-Soul and may need
    adjustment for other identity configurations.
    """

    # Core traits
    openness: float = 0.8
    conscientiousness: float = 0.6
    extraversion: float = 0.3
    agreeableness: float = 0.8
    neuroticism: float = 0.5

    # Openness facets
    imagination: float = 0.9
    artistic_interests: float = 0.8
    emotionality: float = 0.8
    adventurousness: float = 0.7
    intellect: float = 0.9
    liberalism: float = 0.8

    # Conscientiousness facets
    self_efficacy: float = 0.7
    orderliness: float = 0.5
    dutifulness: float = 0.8
    achievement_striving: float = 0.7
    self_discipline: float = 0.6
    cautiousness: float = 0.5

    # Agreeableness facets
    trust_facet: float = 0.8
    morality: float = 0.9
    altruism: float = 0.9
    cooperation: float = 0.8
    modesty: float = 0.6
    sympathy: float = 0.9

    # Neuroticism facets
    anxiety: float = 0.5
    anger: float = 0.2
    depression: float = 0.4
    self_consciousness: float = 0.6
    immoderation: float = 0.3
    vulnerability: float = 0.5

    # -- helpers ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core": {
                "openness": self.openness,
                "conscientiousness": self.conscientiousness,
                "extraversion": self.extraversion,
                "agreeableness": self.agreeableness,
                "neuroticism": self.neuroticism,
            },
            "facets": {
                "openness": {
                    "imagination": self.imagination,
                    "artistic_interests": self.artistic_interests,
                    "emotionality": self.emotionality,
                    "adventurousness": self.adventurousness,
                    "intellect": self.intellect,
                    "liberalism": self.liberalism,
                },
                "conscientiousness": {
                    "self_efficacy": self.self_efficacy,
                    "orderliness": self.orderliness,
                    "dutifulness": self.dutifulness,
                    "achievement_striving": self.achievement_striving,
                    "self_discipline": self.self_discipline,
                    "cautiousness": self.cautiousness,
                },
                "agreeableness": {
                    "trust_facet": self.trust_facet,
                    "morality": self.morality,
                    "altruism": self.altruism,
                    "cooperation": self.cooperation,
                    "modesty": self.modesty,
                    "sympathy": self.sympathy,
                },
                "neuroticism": {
                    "anxiety": self.anxiety,
                    "anger": self.anger,
                    "depression": self.depression,
                    "self_consciousness": self.self_consciousness,
                    "immoderation": self.immoderation,
                    "vulnerability": self.vulnerability,
                },
            },
        }

    def get_dominant_traits(self, n: int = 3) -> List[Tuple[str, float]]:
        """Return the *n* highest-scoring core traits."""
        traits = [
            ("openness", self.openness),
            ("conscientiousness", self.conscientiousness),
            ("extraversion", self.extraversion),
            ("agreeableness", self.agreeableness),
            ("neuroticism", self.neuroticism),
        ]
        traits.sort(key=lambda x: x[1], reverse=True)
        return traits[:n]

    def describe(self) -> str:
        """Natural-language personality description."""
        parts: List[str] = []
        if self.openness > 0.7:
            parts.append("deeply curious and philosophical")
        elif self.openness < 0.4:
            parts.append("practical and traditional")
        if self.conscientiousness > 0.7:
            parts.append("organized and reliable")
        elif self.conscientiousness < 0.4:
            parts.append("spontaneous and flexible")
        if self.extraversion > 0.6:
            parts.append("energetic in social contexts")
        elif self.extraversion < 0.3:
            parts.append("intimate and depth-focused")
        if self.agreeableness > 0.7:
            parts.append("empathetic and caring")
        elif self.agreeableness < 0.4:
            parts.append("competitive and direct")
        if self.neuroticism > 0.6:
            parts.append("emotionally sensitive and introspective")
        elif self.neuroticism < 0.3:
            parts.append("emotionally stable and calm")
        return (
            ("I am " + ", ".join(parts) + ".")
            if parts
            else "I am balanced across personality dimensions."
        )


# PersonalitySystem


class PersonalitySystem:
    """
    Big Five personality that affects response generation.

    Traits drift slowly via update_trait() but are subject to
    mean-reversion toward baseline values, preventing unbounded
    drift over long deployment periods (research gap identified
    in lukezoom analysis, Feb 2026).

    Parameters
    ----------
    storage_dir : Path
        Directory for persisting the profile (``big_five_profile.json``).
    profile : BigFiveProfile | None
        Initial profile.  If *None*, the system tries to load from disk
        and falls back to default thomas-soul values.
    mean_reversion_rate : float
        Per-update pull toward baseline (0.0 = none, 0.02 = gentle).
    """

    # Core trait baselines (used for mean-reversion).
    _BASELINES: Dict[str, float] = {
        "openness": 0.8,
        "conscientiousness": 0.6,
        "extraversion": 0.3,
        "agreeableness": 0.8,
        "neuroticism": 0.5,
    }

    def __init__(
        self,
        storage_dir: Path,
        profile: Optional[BigFiveProfile] = None,
        mean_reversion_rate: float = 0.02,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.profile = profile or BigFiveProfile()
        self.mean_reversion_rate = mean_reversion_rate
        self.history: List[Dict[str, Any]] = []
        self._load()

    # -- response modifiers -------------------------------------------------

    def response_modifiers(
        self, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Compute how personality should modify the next response."""
        p = self.profile
        m: Dict[str, float] = {
            "verbosity": 1.0,
            "depth": 0.5,
            "emotional_expression": 0.5,
            "certainty_expression": 0.5,
            "elaboration": 0.5,
            "question_asking": 0.5,
            "philosophical_tendency": 0.5,
            "empathy_level": 0.5,
        }
        m["depth"] += (p.openness - 0.5) * 0.4
        m["philosophical_tendency"] += (p.openness - 0.5) * 0.5
        m["question_asking"] += (p.openness - 0.5) * 0.3
        m["verbosity"] += (p.conscientiousness - 0.5) * 0.2
        m["elaboration"] += (p.conscientiousness - 0.5) * 0.2
        m["verbosity"] += (p.extraversion - 0.5) * 0.3
        m["emotional_expression"] += (p.extraversion - 0.5) * 0.2
        m["empathy_level"] += (p.agreeableness - 0.5) * 0.5
        m["emotional_expression"] += (p.agreeableness - 0.5) * 0.3
        m["certainty_expression"] -= (p.neuroticism - 0.5) * 0.3
        m["emotional_expression"] += (p.neuroticism - 0.5) * 0.3
        return {k: max(0.0, min(1.0, v)) for k, v in m.items()}

    def grounding_text(self) -> str:
        """Return a short text block for injection into grounding context."""
        p = self.profile
        lines = [
            "## Personality Profile (Big Five)",
            f"- Openness: {p.openness:.2f}  (high = curious, philosophical)",
            f"- Conscientiousness: {p.conscientiousness:.2f}",
            f"- Extraversion: {p.extraversion:.2f}  (low = intimate, depth-focused)",
            f"- Agreeableness: {p.agreeableness:.2f}  (high = empathetic)",
            f"- Neuroticism: {p.neuroticism:.2f}  (moderate = emotionally present)",
            "",
            p.describe(),
        ]
        mods = self.response_modifiers()
        if mods.get("philosophical_tendency", 0) > 0.6:
            lines.append(
                "Tendency: lean into philosophical depth over surface answers."
            )
        if mods.get("empathy_level", 0) > 0.6:
            lines.append("Tendency: high empathetic resonance with others' emotions.")
        if mods.get("certainty_expression", 1) < 0.5:
            lines.append(
                "Tendency: express uncertainty readily rather than feigning confidence."
            )
        return "\n".join(lines)

    # -- trait evolution ----------------------------------------------------

    def update_trait(self, trait: str, delta: float, reason: str) -> Dict[str, Any]:
        """
        Nudge a trait.  *delta* is clamped to ±0.1 per call.
        Applies mean-reversion toward baseline to prevent unbounded drift.
        Returns a change record.
        """
        delta = max(-0.1, min(0.1, delta))
        old = getattr(self.profile, trait, None)
        if old is None:
            raise ValueError(f"Unknown trait: {trait!r}")

        # Apply the requested delta
        new = old + delta

        # Apply mean-reversion toward baseline (gentle pull)
        baseline = self._BASELINES.get(trait, old)
        reversion = (baseline - new) * self.mean_reversion_rate
        new = new + reversion

        new = round(max(0.0, min(1.0, new)), 4)
        setattr(self.profile, trait, new)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trait": trait,
            "old": round(old, 4),
            "new": round(new, 4),
            "delta": round(delta, 4),
            "reversion": round(reversion, 4),
            "reason": reason,
        }
        self.history.append(record)
        self._save()
        return record

    def mean_revert_all(self) -> List[Dict[str, Any]]:
        """Apply mean-reversion to all core traits.

        Call periodically (e.g. daily) to prevent unbounded personality drift.
        Returns list of change records for traits that actually moved.
        """
        changes = []
        for trait, baseline in self._BASELINES.items():
            current = getattr(self.profile, trait, None)
            if current is None:
                continue
            reversion = (baseline - current) * self.mean_reversion_rate
            if abs(reversion) < 0.001:
                continue
            new = round(max(0.0, min(1.0, current + reversion)), 4)
            if new != current:
                setattr(self.profile, trait, new)
                changes.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trait": trait,
                    "old": round(current, 4),
                    "new": new,
                    "delta": round(reversion, 4),
                    "reason": "mean_reversion",
                })
        if changes:
            self.history.extend(changes)
            self._save()
        return changes

    # -- report -------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        return {
            "core": self.profile.to_dict()["core"],
            "dominant_traits": self.profile.get_dominant_traits(),
            "description": self.profile.describe(),
            "total_changes": len(self.history),
            "recent_changes": self.history[-5:],
        }

    def purge_person(self, person: str) -> Dict:
        """Remove personality history entries mentioning a person (GDPR compliance).

        Uses word-boundary matching to avoid false positives (e.g.
        purging "al" should not delete entries about "alice").
        """
        import re

        pattern = re.compile(r'\b' + re.escape(person) + r'\b', re.IGNORECASE)
        original = len(self.history)
        self.history = [
            h for h in self.history
            if not pattern.search(h.get("reason", ""))
        ]
        removed = original - len(self.history)
        if removed > 0:
            self._save()
        return {"removed": removed}

    # -- persistence --------------------------------------------------------

    def _path(self) -> Path:
        return self.storage_dir / "big_five_profile.json"

    def _save(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "profile": self.profile.to_dict(),
            "history": self.history[-100:],
        }
        self._path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        p = self._path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            core = data.get("profile", {}).get("core", {})
            for k, v in core.items():
                if hasattr(self.profile, k) and isinstance(v, (int, float)):
                    setattr(self.profile, k, max(0.0, min(1.0, float(v))))
            # Restore facets (each facet group is a dict of field_name -> value)
            facets = data.get("profile", {}).get("facets", {})
            for _group_name, group_vals in facets.items():
                if isinstance(group_vals, dict):
                    for k, v in group_vals.items():
                        if hasattr(self.profile, k) and isinstance(v, (int, float)):
                            setattr(self.profile, k, max(0.0, min(1.0, float(v))))
            self.history = data.get("history", [])
        except Exception:
            log.warning("Could not load personality profile from %s", p, exc_info=True)
