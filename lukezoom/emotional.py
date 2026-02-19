"""
lukezoom.emotional — VAD emotional continuity system.

Ported from thomas-soul/thomas_core/emotional_continuity.py.
Emotions persist between interactions via exponential decay toward
neutral baselines (valence→0, arousal→0.5, dominance→0.5).

Decay rate ordering (arousal fastest, valence slowest) is grounded in:
  - Fading Affect Bias (Walker & Skowronski 2009): negative affect
    fades faster than positive (51% vs 37%).
  - Gibbons et al. 2013: arousal decay is independent of valence decay.
  - NRC VAD Lexicon v2 (Mohammad 2025): near-orthogonal dimensions
    (|ρ(A,D)| ≈ 0.1) validate independent per-dimension decay.

Decay rates are configurable via Config.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("lukezoom.emotional")


# EmotionalEvent


@dataclass
class EmotionalEvent:
    """A single event that changed emotional state."""

    timestamp: str  # ISO-8601
    description: str
    valence_delta: float
    arousal_delta: float
    dominance_delta: float
    source: str
    intensity: float  # 0.0 – 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "description": self.description,
            "valence_delta": self.valence_delta,
            "arousal_delta": self.arousal_delta,
            "dominance_delta": self.dominance_delta,
            "source": self.source,
            "intensity": self.intensity,
        }


# EmotionalSystem


class EmotionalSystem:
    """
    Dimensional emotional state (Valence-Arousal-Dominance) with
    temporal decay and event history.

    Parameters
    ----------
    storage_dir : Path
        Where ``emotional_state.json`` and daily JSONL events live.
    valence_decay, arousal_decay, dominance_decay : float
        Exponential decay factors *per hour* toward neutral.
    """

    def __init__(
        self,
        storage_dir: Path,
        valence_decay: float = 0.9,
        arousal_decay: float = 0.7,
        dominance_decay: float = 0.8,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.valence: float = 0.0
        self.arousal: float = 0.5
        self.dominance: float = 0.5
        self.valence_decay = valence_decay
        self.arousal_decay = arousal_decay
        self.dominance_decay = dominance_decay
        self.last_update: datetime = datetime.now(timezone.utc)
        self.events: List[EmotionalEvent] = []
        self.state_history: List[Dict[str, Any]] = []
        self._load()

    # -- public API ---------------------------------------------------------

    def update(
        self,
        description: str,
        valence_delta: float = 0.0,
        arousal_delta: float = 0.0,
        dominance_delta: float = 0.0,
        source: str = "unknown",
        intensity: float = 0.5,
    ) -> Dict[str, Any]:
        """Apply an emotional event and return current state."""
        self._apply_decay()
        now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        evt = EmotionalEvent(
            timestamp=now_str,
            description=description,
            valence_delta=valence_delta,
            arousal_delta=arousal_delta,
            dominance_delta=dominance_delta,
            source=source,
            intensity=intensity,
        )
        self.events.append(evt)
        self.valence = max(-1.0, min(1.0, self.valence + valence_delta * intensity))
        self.arousal = max(0.0, min(1.0, self.arousal + arousal_delta * intensity))
        self.dominance = max(
            0.0, min(1.0, self.dominance + dominance_delta * intensity)
        )
        snap = {
            "timestamp": now_str,
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "trigger": description,
        }
        self.state_history.append(snap)
        self._save()
        return self.current_state()

    def current_state(self) -> Dict[str, Any]:
        """Return current dimensional state with mood label and trend."""
        self._apply_decay()
        trend = self._trend()
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "mood": self._mood_label(),
            "trend": trend,
            "recent_events": [e.description for e in self.events[-3:]],
            "stability": self._stability(),
        }

    def mood_history(self, n: int = 10) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events[-n:]]

    def grounding_text(self) -> str:
        """Short block for before-pipeline context injection."""
        state = self.current_state()
        return (
            f"Current emotional state: {state['mood']} "
            f"(valence={state['valence']:+.2f}, arousal={state['arousal']:.2f}, "
            f"dominance={state['dominance']:.2f}). Trend: {state['trend']}."
        )

    def purge_person(self, person: str) -> Dict:
        """Remove emotional events mentioning a person (GDPR compliance).

        Uses word-boundary matching to avoid false positives (e.g.
        purging "al" should not delete entries about "alice").
        """
        import re

        pattern = re.compile(r'\b' + re.escape(person) + r'\b', re.IGNORECASE)
        original = len(self.events)
        self.events = [
            e for e in self.events
            if not pattern.search(e.description)
            and not pattern.search(e.source)
        ]
        # Also scrub state_history entries triggered by this person
        self.state_history = [
            s for s in self.state_history
            if not pattern.search(s.get("trigger", ""))
        ]
        removed = original - len(self.events)
        if removed > 0:
            self._save()
        return {"removed": removed}

    # -- internal -----------------------------------------------------------

    def _apply_decay(self) -> None:
        now = datetime.now(timezone.utc)
        hours = (now - self.last_update).total_seconds() / 3600.0
        if hours > 0:
            self.valence *= self.valence_decay**hours
            self.arousal = 0.5 + (self.arousal - 0.5) * (self.arousal_decay**hours)
            self.dominance = 0.5 + (self.dominance - 0.5) * (
                self.dominance_decay**hours
            )
            # Clamp to valid ranges after decay (guard against float drift)
            self.valence = max(-1.0, min(1.0, self.valence))
            self.arousal = max(0.0, min(1.0, self.arousal))
            self.dominance = max(0.0, min(1.0, self.dominance))
        self.last_update = now

    def _mood_label(self) -> str:
        v, a, d = self.valence, self.arousal, self.dominance
        if v > 0.6 and a > 0.6:
            return "excited" if d > 0.5 else "enthusiastic"
        if v > 0.6 and a < 0.4:
            return "serene" if d > 0.5 else "content"
        if v > 0.3:
            return "positive"
        if v < -0.6 and a > 0.6:
            return "distressed" if d < 0.5 else "angry"
        if v < -0.6 and a < 0.4:
            return "depressed" if d < 0.3 else "sad"
        if v < -0.3:
            return "negative"
        if a > 0.7:
            return "activated" if d > 0.5 else "anxious"
        if a < 0.3:
            return "calm" if v > -0.2 else "withdrawn"
        return "neutral"

    def _trend(self) -> str:
        if len(self.state_history) < 2:
            return "insufficient_data"
        recent = self.state_history[-5:]
        delta = recent[-1]["valence"] - recent[0]["valence"]
        if delta > 0.1:
            return "improving"
        if delta < -0.1:
            return "declining"
        return "stable"

    def _stability(self) -> str:
        if len(self.events) < 5:
            return "insufficient_data"
        recent = self.events[-10:]
        avg = sum(abs(e.valence_delta) + abs(e.arousal_delta) for e in recent) / len(
            recent
        )
        if avg < 0.1:
            return "very_stable"
        if avg < 0.2:
            return "stable"
        if avg < 0.4:
            return "moderate"
        return "volatile"

    # -- persistence --------------------------------------------------------

    def _state_path(self) -> Path:
        return self.storage_dir / "emotional_state.json"

    def _save(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "last_update": self.last_update.isoformat().replace("+00:00", "Z"),
            "state_history": self.state_history[-100:],
            "events": [e.to_dict() for e in self.events[-100:]],
        }
        self._state_path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        p = self._state_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.valence = max(-1.0, min(1.0, float(data.get("valence", 0.0))))
            self.arousal = max(0.0, min(1.0, float(data.get("arousal", 0.5))))
            self.dominance = max(0.0, min(1.0, float(data.get("dominance", 0.5))))
            ts = data.get("last_update")
            if ts:
                self.last_update = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            self.state_history = data.get("state_history", [])
            # Restore events
            events_raw = data.get("events", [])
            self.events = [
                EmotionalEvent(**e) for e in events_raw if isinstance(e, dict)
            ]
        except Exception:
            log.warning("Could not load emotional state from %s", p, exc_info=True)
