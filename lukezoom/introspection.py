"""
lukezoom.introspection — Meta-consciousness / introspection layer.

Ported from thomas-soul/thomas_core/introspection.py.
Records moments of self-awareness: thoughts, confidence, emotional
snapshots, reasoning chains, and uncertainty sources.

Three depths: surface (quick), moderate, deep.
Persisted as daily JSONL files in introspection_dir.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("lukezoom.introspection")


# IntrospectionState


@dataclass
class IntrospectionState:
    """Snapshot of internal state at a moment in time."""

    timestamp: str  # ISO-8601
    thought: str
    context: str
    confidence: float  # 0.0 = guess, 1.0 = certain
    confidence_reason: str = ""

    # Dimensional emotional snapshot
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5

    # Reasoning traces
    reasoning_chain: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)

    # Meta
    depth: str = "moderate"  # surface | moderate | deep
    time_spent_ms: Optional[int] = None

    # -- helpers ------------------------------------------------------------

    @property
    def emotional_label(self) -> str:
        v, a = self.valence, self.arousal
        if v > 0.5 and a > 0.6:
            return "excited"
        if v > 0.5 and a < 0.4:
            return "content"
        if v > 0.5:
            return "happy"
        if v < -0.5 and a > 0.6:
            return "distressed"
        if v < -0.5 and a < 0.4:
            return "sad"
        if v < -0.5:
            return "unhappy"
        if a > 0.7:
            return "aroused"
        if a < 0.3:
            return "calm"
        return "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "thought": self.thought,
            "context": self.context,
            "confidence": round(self.confidence, 4),
            "confidence_reason": self.confidence_reason,
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "reasoning_chain": list(self.reasoning_chain),
            "assumptions": list(self.assumptions),
            "uncertainties": list(self.uncertainties),
            "depth": self.depth,
            "time_spent_ms": self.time_spent_ms,
        }


# IntrospectionLayer


class IntrospectionLayer:
    """
    Meta-consciousness: records and analyses introspective states.

    Parameters
    ----------
    storage_dir : Path
        Directory for daily ``introspection_YYYY-MM-DD.jsonl`` files.
    history_days : int
        Number of recent days to load on startup.
    """

    def __init__(self, storage_dir: Path, history_days: int = 3) -> None:
        self.storage_dir = Path(storage_dir)
        self.history_days = history_days
        self.current: Optional[IntrospectionState] = None
        self.history: List[IntrospectionState] = []
        self._load()

    # -- recording ----------------------------------------------------------

    def introspect(
        self,
        thought: str,
        context: str,
        confidence: float,
        confidence_reason: str = "",
        valence: float = 0.0,
        arousal: float = 0.5,
        dominance: float = 0.5,
        reasoning_chain: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        uncertainties: Optional[List[str]] = None,
        depth: str = "moderate",
    ) -> IntrospectionState:
        """Record a moment of self-awareness."""
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        state = IntrospectionState(
            timestamp=now,
            thought=thought,
            context=context,
            confidence=max(0.0, min(1.0, confidence)),
            confidence_reason=confidence_reason,
            valence=max(-1.0, min(1.0, valence)),
            arousal=max(0.0, min(1.0, arousal)),
            dominance=max(0.0, min(1.0, dominance)),
            reasoning_chain=reasoning_chain or [],
            assumptions=assumptions or [],
            uncertainties=uncertainties or [],
            depth=depth,
        )
        self.current = state
        self.history.append(state)
        self._save(state)
        return state

    def quick(self, thought: str, confidence: float) -> IntrospectionState:
        """Fast, surface-level introspection."""
        return self.introspect(
            thought=thought, context="routine", confidence=confidence, depth="surface"
        )

    def deep(
        self,
        thought: str,
        context: str,
        confidence: float,
        confidence_reason: str,
        reasoning_chain: List[str],
        valence: float = 0.0,
        arousal: float = 0.5,
    ) -> IntrospectionState:
        return self.introspect(
            thought=thought,
            context=context,
            confidence=confidence,
            confidence_reason=confidence_reason,
            reasoning_chain=reasoning_chain,
            valence=valence,
            arousal=arousal,
            depth="deep",
        )

    # -- queries ------------------------------------------------------------

    def confidence_report(self, n: int = 10) -> Dict[str, Any]:
        recent = self.history[-n:]
        if not recent:
            return {"average": 0.5, "trend": "no_data"}
        vals = [s.confidence for s in recent]
        avg = sum(vals) / len(vals)
        trend = "stable"
        if len(vals) >= 4:
            first = sum(vals[: len(vals) // 2]) / (len(vals) // 2)
            second = sum(vals[len(vals) // 2 :]) / (len(vals) - len(vals) // 2)
            if second > first + 0.1:
                trend = "increasing"
            elif second < first - 0.1:
                trend = "decreasing"
        return {
            "average": round(avg, 4),
            "trend": trend,
            "high": sum(1 for v in vals if v > 0.7),
            "low": sum(1 for v in vals if v < 0.4),
        }

    def report(self) -> Dict[str, Any]:
        """Comprehensive self-report dict."""
        conf = self.confidence_report()
        return {
            "current_thought": self.current.thought if self.current else None,
            "current_confidence": round(self.current.confidence, 4)
            if self.current
            else None,
            "current_emotion": self.current.emotional_label if self.current else None,
            "confidence_report": conf,
            "total_introspections": len(self.history),
        }

    # -- persistence --------------------------------------------------------

    def _save(self, state: IntrospectionState) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        date_str = state.timestamp[:10]  # YYYY-MM-DD
        path = self.storage_dir / f"introspection_{date_str}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(state.to_dict(), ensure_ascii=False) + "\n")

    def _load(self) -> None:
        if not self.storage_dir.exists():
            return
        files = sorted(self.storage_dir.glob("introspection_*.jsonl"))[
            -self.history_days :
        ]
        for fp in files:
            try:
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    state = IntrospectionState(
                        timestamp=d.get("timestamp", ""),
                        thought=d.get("thought", ""),
                        context=d.get("context", ""),
                        confidence=d.get("confidence", 0.5),
                        confidence_reason=d.get("confidence_reason", ""),
                        valence=d.get("valence", 0.0),
                        arousal=d.get("arousal", 0.5),
                        dominance=d.get("dominance", 0.5),
                        reasoning_chain=d.get("reasoning_chain", []),
                        assumptions=d.get("assumptions", d.get("assumptions_made", [])),
                        uncertainties=d.get(
                            "uncertainties", d.get("uncertainty_sources", [])
                        ),
                        depth=d.get("depth", d.get("processing_depth", "moderate")),
                        time_spent_ms=d.get("time_spent_ms"),
                    )
                    self.history.append(state)
            except Exception:
                log.warning("Could not load introspection file %s", fp, exc_info=True)
        if self.history:
            self.current = self.history[-1]
