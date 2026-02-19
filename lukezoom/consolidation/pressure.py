"""
lukezoom.consolidation.pressure — Memory pressure monitoring and decay throttling.

Inspired by MemGPT's memory pressure warnings: the system detects when
memory utilisation is high and triggers compaction/eviction. Instead of
running a full O(N) decay pass on every after() call, we throttle decay
based on pressure level and time since last run.

Pressure levels:
    NORMAL   — trace count < 60% of max_traces. Decay hourly.
    ELEVATED — 60-80%. Decay every 10 min, trigger compaction.
    CRITICAL — > 80%. Decay every call, aggressive compaction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lukezoom.core.config import Config
    from lukezoom.episodic.store import EpisodicStore

log = logging.getLogger(__name__)


class PressureLevel(Enum):
    """Memory pressure levels — higher means more aggressive maintenance."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


@dataclass
class PressureState:
    """Snapshot of memory pressure at a point in time.

    Attributes
    ----------
    level : PressureLevel
        Current pressure category.
    utilisation : float
        Fraction of max_traces currently in use (0.0–1.0).
    trace_count : int
        Current number of traces.
    max_traces : int
        Configured maximum.
    should_decay : bool
        Whether a decay pass should run this cycle.
    should_compact : bool
        Whether conversation compaction should run this cycle.
    seconds_since_decay : float
        Wall-clock seconds since the last decay pass completed.
    """

    level: PressureLevel
    utilisation: float
    trace_count: int
    max_traces: int
    should_decay: bool
    should_compact: bool
    seconds_since_decay: float


# Decay interval per pressure level (seconds)
_DECAY_INTERVALS = {
    PressureLevel.NORMAL: 3600.0,  # once per hour
    PressureLevel.ELEVATED: 600.0,  # once per 10 minutes
    PressureLevel.CRITICAL: 0.0,  # every call
}

# Utilisation thresholds
_ELEVATED_THRESHOLD = 0.60
_CRITICAL_THRESHOLD = 0.80


class MemoryPressure:
    """Track memory utilisation and decide when to run decay/compaction.

    Wire this into the after-pipeline to replace the unconditional
    ``decay_pass()`` call.

    Parameters
    ----------
    config : Config
        Lukezoom configuration (uses ``max_traces``).
    """

    def __init__(self, config: Config) -> None:
        self.max_traces = config.max_traces
        self._last_decay: Optional[datetime] = None
        self._last_compact: Optional[datetime] = None

    # Public API

    def check(self, episodic: EpisodicStore) -> PressureState:
        """Evaluate current pressure and return a recommendation.

        The caller should inspect ``state.should_decay`` and
        ``state.should_compact`` and act accordingly.
        """
        trace_count = episodic.count_traces()
        utilisation = trace_count / max(1, self.max_traces)
        level = self._classify(utilisation)

        now = datetime.now(timezone.utc)
        seconds_since = self._seconds_since_decay(now)

        should_decay = self._should_decay(level, seconds_since)
        should_compact = self._should_compact(level, now)

        state = PressureState(
            level=level,
            utilisation=utilisation,
            trace_count=trace_count,
            max_traces=self.max_traces,
            should_decay=should_decay,
            should_compact=should_compact,
            seconds_since_decay=seconds_since,
        )

        if level != PressureLevel.NORMAL:
            log.info(
                "Memory pressure: %s (%.0f%%, %d/%d traces)",
                level.value,
                utilisation * 100,
                trace_count,
                self.max_traces,
            )

        return state

    def record_decay(self) -> None:
        """Mark that a decay pass just completed."""
        self._last_decay = datetime.now(timezone.utc)

    def record_compaction(self) -> None:
        """Mark that a compaction just completed."""
        self._last_compact = datetime.now(timezone.utc)

    # Internals

    @staticmethod
    def _classify(utilisation: float) -> PressureLevel:
        if utilisation >= _CRITICAL_THRESHOLD:
            return PressureLevel.CRITICAL
        if utilisation >= _ELEVATED_THRESHOLD:
            return PressureLevel.ELEVATED
        return PressureLevel.NORMAL

    def _seconds_since_decay(self, now: datetime) -> float:
        if self._last_decay is None:
            return float("inf")
        return (now - self._last_decay).total_seconds()

    def _should_decay(self, level: PressureLevel, seconds_since: float) -> bool:
        interval = _DECAY_INTERVALS[level]
        return seconds_since >= interval

    def _should_compact(self, level: PressureLevel, now: datetime) -> bool:
        """Compact when elevated/critical and at least 5 min since last compaction."""
        if level == PressureLevel.NORMAL:
            return False
        if self._last_compact is None:
            return True
        return (now - self._last_compact).total_seconds() >= 300.0
