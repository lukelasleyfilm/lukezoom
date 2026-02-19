"""
lukezoom.core.types — Data types for the LukeZOOM memory system.

Every structure is a plain dataclass: no ORM, no magic,
serialisable to dict/JSON in one call.

1.17 changes from 1.3:
  - Removed plugin-only TraceKind values (relic_assessment, personality_change,
    emotional_state) — they belong in their respective plugin modules.
  - Signal.state thresholds cleaned to match measure.py constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
import uuid

from lukezoom.core.tokens import estimate_tokens

# Type aliases
LLMFunc = Callable[[str, str], str]


# Helpers

def generate_id() -> str:
    """12-hex-char unique identifier."""
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    """Current UTC timestamp in ISO-8601 with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ── Trace kinds ─────────────────────────────────────────────────────

TRACE_KINDS = frozenset({
    # Core episodic
    "episode", "realization", "emotion", "correction",
    "relational", "mood", "factual",
    # Identity
    "identity_core", "uncertainty", "anticipation",
    "creative_journey", "reflection", "emotional_thread",
    "promise", "confidence",
    # Consolidation (MemGPT-inspired)
    "summary", "thread", "arc",
    # Cognitive integration
    "temporal", "utility", "introspection",
    "workspace_eviction", "belief_evolution", "dissociation_event",
})


# ── Trace ───────────────────────────────────────────────────────────

@dataclass
class Trace:
    """
    Atomic unit of episodic memory.

    A trace records a single moment of experience — something said,
    felt, realised, or corrected.  Traces are the rows of the episodic
    store and the primary unit of salience decay / reinforcement.
    """

    content: str
    kind: str = "episode"
    tags: List[str] = field(default_factory=list)
    salience: float = 0.5
    metadata: Dict = field(default_factory=dict)

    # auto-populated
    id: str = field(default_factory=generate_id)
    created: str = field(default_factory=now_iso)
    tokens: int = 0
    access_count: int = 0
    last_accessed: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        if self.kind not in TRACE_KINDS:
            raise ValueError(
                f"Invalid trace kind {self.kind!r}; "
                f"expected one of {sorted(TRACE_KINDS)}"
            )
        if self.tokens == 0:
            self.tokens = estimate_tokens(self.content)
        self.salience = max(0.0, min(1.0, self.salience))

    def touch(self) -> None:
        """Record an access (context-load) of this trace."""
        self.access_count += 1
        self.last_accessed = now_iso()

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "content": self.content, "created": self.created,
            "kind": self.kind, "tags": list(self.tags),
            "salience": round(self.salience, 4), "tokens": self.tokens,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Trace":
        return cls(
            id=d["id"], content=d["content"],
            created=d.get("created", now_iso()),
            kind=d.get("kind", "episode"),
            tags=d.get("tags", []),
            salience=d.get("salience", 0.5),
            tokens=d.get("tokens", 0),
            access_count=d.get("access_count", 0),
            last_accessed=d.get("last_accessed", now_iso()),
            metadata=d.get("metadata", {}),
        )


# ── Message ─────────────────────────────────────────────────────────

@dataclass
class Message:
    """One turn in a conversation."""

    person: str
    speaker: str
    content: str
    source: str = "direct"
    salience: float = 0.5
    signal: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

    id: str = field(default_factory=generate_id)
    timestamp: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        self.salience = max(0.0, min(1.0, self.salience))

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "person": self.person, "speaker": self.speaker,
            "content": self.content, "source": self.source,
            "timestamp": self.timestamp,
            "salience": round(self.salience, 4),
            "signal": self.signal, "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Message":
        return cls(
            id=d["id"], person=d["person"], speaker=d["speaker"],
            content=d["content"], source=d.get("source", "direct"),
            timestamp=d.get("timestamp", now_iso()),
            salience=d.get("salience", 0.5),
            signal=d.get("signal"), metadata=d.get("metadata", {}),
        )


# ── Signal ──────────────────────────────────────────────────────────

@dataclass
class Signal:
    """
    Four-facet identity coherence signal.

    Each facet is 0.0–1.0:
      alignment  — how true-to-identity the response is (weight 0.35)
      embodiment — first-person presence vs detached observer (0.25)
      clarity    — coherence of thought, concrete vs jargon (0.20)
      vitality   — aliveness / engagement vs flat performance (0.20)

    Weights from Ensoul identity research (Lasley, 2024).
    """

    alignment: float = 0.5
    embodiment: float = 0.5
    clarity: float = 0.5
    vitality: float = 0.5
    trace_ids: List[str] = field(default_factory=list)

    _WEIGHTS = {
        "alignment": 0.35, "embodiment": 0.25,
        "clarity": 0.20, "vitality": 0.20,
    }

    def __post_init__(self) -> None:
        for attr in ("alignment", "embodiment", "clarity", "vitality"):
            setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))

    @property
    def health(self) -> float:
        """Weighted composite score."""
        return sum(
            getattr(self, k) * w for k, w in self._WEIGHTS.items()
        )

    @property
    def state(self) -> str:
        h = self.health
        if h >= 0.75:
            return "coherent"
        if h >= 0.50:
            return "developing"
        if h >= 0.35:
            return "drifting"
        return "dissociated"

    @property
    def needs_correction(self) -> bool:
        return self.health < 0.5

    @property
    def weakest_facet(self) -> str:
        facets = {k: getattr(self, k) for k in self._WEIGHTS}
        return min(facets, key=facets.get)  # type: ignore

    @property
    def polarity_gap(self) -> float:
        vals = [getattr(self, k) for k in self._WEIGHTS]
        return max(vals) - min(vals)

    def to_dict(self) -> Dict:
        return {
            "alignment": round(self.alignment, 4),
            "embodiment": round(self.embodiment, 4),
            "clarity": round(self.clarity, 4),
            "vitality": round(self.vitality, 4),
            "health": round(self.health, 4),
            "state": self.state,
            "needs_correction": self.needs_correction,
            "weakest_facet": self.weakest_facet,
            "trace_ids": list(self.trace_ids),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Signal":
        return cls(
            alignment=d.get("alignment", 0.5),
            embodiment=d.get("embodiment", 0.5),
            clarity=d.get("clarity", 0.5),
            vitality=d.get("vitality", 0.5),
            trace_ids=d.get("trace_ids", []),
        )


# ── Context ─────────────────────────────────────────────────────────

@dataclass
class Context:
    """Assembled context from the before-pipeline, ready for LLM injection."""

    text: str
    trace_ids: List[str] = field(default_factory=list)
    person: str = ""
    tokens_used: int = 0
    token_budget: int = 6000
    memories_loaded: int = 0
    health: Optional["HealthBitmap"] = None

    def __post_init__(self) -> None:
        if self.tokens_used == 0 and self.text:
            self.tokens_used = estimate_tokens(self.text)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    def to_dict(self) -> Dict:
        result = {
            "text": self.text, "trace_ids": list(self.trace_ids),
            "person": self.person, "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "memories_loaded": self.memories_loaded,
            "budget_remaining": self.budget_remaining,
        }
        if self.health is not None:
            result["health"] = self.health.to_dict()
        return result


# ── HealthBitmap ───────────────────────────────────────────────────

@dataclass
class HealthBitmap:
    """Tracks subsystem success/failure for a single pipeline call.

    Each subsystem that runs during before() or after() records its
    outcome here, replacing the old fire-and-forget exception swallowing
    with structured, inspectable health data.
    """

    results: Dict[str, bool] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def record(self, subsystem: str, exc: Optional[Exception] = None) -> None:
        """Record a subsystem outcome. None exc = success."""
        if exc is None:
            self.results[subsystem] = True
        else:
            self.results[subsystem] = False
            self.errors[subsystem] = f"{type(exc).__name__}: {exc}"

    @property
    def all_ok(self) -> bool:
        return all(self.results.values()) if self.results else True

    @property
    def failed_subsystems(self) -> List[str]:
        return [k for k, v in self.results.items() if not v]

    def to_dict(self) -> Dict:
        return {
            "results": dict(self.results),
            "errors": dict(self.errors),
            "all_ok": self.all_ok,
            "failed": self.failed_subsystems,
        }


# ── AfterResult ─────────────────────────────────────────────────────

@dataclass
class AfterResult:
    """Output of the after-pipeline: signal + logged IDs."""

    signal: Signal
    salience: float = 0.5
    updates: List[Dict] = field(default_factory=list)
    logged_message_id: str = ""
    logged_trace_id: Optional[str] = None
    health: Optional["HealthBitmap"] = None

    def __post_init__(self) -> None:
        self.salience = max(0.0, min(1.0, self.salience))

    def to_dict(self) -> Dict:
        result = {
            "signal": self.signal.to_dict(),
            "salience": round(self.salience, 4),
            "updates": list(self.updates),
            "logged_message_id": self.logged_message_id,
            "logged_trace_id": self.logged_trace_id,
        }
        if self.health is not None:
            result["health"] = self.health.to_dict()
        return result


# ── MemoryStats ─────────────────────────────────────────────────────

@dataclass
class MemoryStats:
    """High-level system health snapshot."""

    episodic_count: int = 0
    semantic_facts: int = 0
    procedural_skills: int = 0
    total_messages: int = 0
    avg_salience: float = 0.0
    memory_pressure: float = 0.0
    status: str = ""

    def __post_init__(self) -> None:
        self.memory_pressure = max(0.0, min(1.0, self.memory_pressure))
        if not self.status:
            if self.memory_pressure > 0.9:
                self.status = "critical"
            elif self.memory_pressure > 0.7:
                self.status = "warning"
            else:
                self.status = "ok"

    @property
    def total_memories(self) -> int:
        return self.episodic_count + self.semantic_facts + self.procedural_skills

    def to_dict(self) -> Dict:
        return {
            "episodic_count": self.episodic_count,
            "semantic_facts": self.semantic_facts,
            "procedural_skills": self.procedural_skills,
            "total_memories": self.total_memories,
            "avg_salience": round(self.avg_salience, 4),
            "memory_pressure": round(self.memory_pressure, 4),
            "status": self.status,
        }
