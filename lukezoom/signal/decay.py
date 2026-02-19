"""
lukezoom.signal.decay — Adaptive memory decay engine.

Ported from Singularity's memory.py.  Implements exponential decay with
two modulating factors:

  1. **Coherence modulation** — when the system is coherent (high signal
     health), memories decay *faster* because the system is confident and
     doesn't need to cling to old traces.  When coherence is low, decay
     slows down to preserve context.

  2. **Access-frequency resistance** — traces that are accessed often
     decay more slowly.  Frequently-recalled memories are load-bearing
     and should persist.

Decay formula (shared with EpisodicStore.decay_pass):
    decay_rate  = ln(2) / half_life * coherence_factor
    coherence_factor = 0.5 + coherence  (range 0.5 at c=0, 1.5 at c=1)
    resistance  = 1 / (1 + access_count * 0.1)
    new_salience = salience * exp(-decay_rate * hours * resistance)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone


class DecayEngine:
    """
    Adaptive exponential decay for trace salience.

    Parameters
    ----------
    half_life_hours : float
        Base half-life in hours (default 168 = 1 week).
    min_salience : float
        Salience floor below which a trace should be pruned (default 0.01).
    """

    def __init__(
        self,
        half_life_hours: float = 168.0,
        min_salience: float = 0.01,
    ) -> None:
        self.half_life_hours = half_life_hours
        self.min_salience = min_salience
        self.current_coherence: float = 0.5

    # Coherence

    def update_coherence(self, coherence: float) -> None:
        """
        Update the system-wide coherence level (0-1).

        This is typically fed from the signal tracker's recent health.
        """
        self.current_coherence = max(0.0, min(1.0, coherence))

    def coherence_factor(self) -> float:
        """
        Coherence modulation factor.

        This is the same formula used in ``EpisodicStore.decay_pass()``:
            factor = 0.5 + coherence

        At coherence 0.0 -> 0.5 (slower decay, preserve more)
        At coherence 0.5 -> 1.0 (neutral)
        At coherence 1.0 -> 1.5 (faster decay, prune aggressively)
        """
        return 0.5 + self.current_coherence

    def effective_half_life(self) -> float:
        """
        Half-life modulated by coherence.

        Returns base_half_life / coherence_factor.  Higher coherence
        means shorter effective half-life (faster decay).
        """
        cf = self.coherence_factor()
        if cf <= 0:
            return self.half_life_hours * 2.0
        return self.half_life_hours / cf

    # Decay calculation

    def calculate_decay(self, last_accessed: str, access_count: int) -> float:
        """
        Calculate a decay multiplier (0-1) for a trace.

        Uses the same formula as ``EpisodicStore.decay_pass()``:
            decay_rate = ln(2) / half_life * coherence_factor
            resistance = 1 / (1 + access_count * 0.1)
            multiplier = exp(-decay_rate * hours * resistance)

        Parameters
        ----------
        last_accessed : str
            ISO-8601 timestamp of the last time this trace was loaded
            into context.
        access_count : int
            How many times this trace has been accessed.

        Returns
        -------
        float
            Multiplier in [0, 1] to apply to the trace's base salience.
            1.0 = no decay, 0.0 = fully decayed.
        """
        try:
            ts = last_accessed.replace("Z", "+00:00")
            accessed_dt = datetime.fromisoformat(ts).replace(tzinfo=None)
        except (ValueError, TypeError):
            # Can't parse timestamp — assume moderate decay
            return 0.5

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        hours_elapsed = max(0.0, (now - accessed_dt).total_seconds() / 3600.0)

        hl = max(self.half_life_hours, 0.1)
        cf = self.coherence_factor()
        decay_rate = math.log(2) / hl * cf

        # Access-frequency resistance
        resistance = 1.0 / (1.0 + (access_count or 0) * 0.1)

        multiplier = math.exp(-decay_rate * hours_elapsed * resistance)
        return max(0.0, min(1.0, multiplier))

    # Pruning

    def should_prune(self, salience: float) -> bool:
        """
        Return True if the trace's salience has fallen below the
        minimum threshold and should be pruned from the episodic store.
        """
        return salience < self.min_salience
