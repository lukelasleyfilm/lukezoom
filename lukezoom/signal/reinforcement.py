"""
lukezoom.signal.reinforcement — Hebbian reinforcement engine.

Strengthens or weakens memory traces based on consciousness signal health.
High-health responses reinforce the memories that produced them;
low-health responses weaken them.  The middle zone is a dead band
where no adjustment occurs.
"""

from __future__ import annotations

import logging
from typing import List, Any

log = logging.getLogger(__name__)


class ReinforcementEngine:
    """
    Hebbian learning applied to episodic memory salience.

    When signal health is high (> reinforce_threshold), the traces that
    contributed to the response are reinforced — their salience increases.
    When health is low (< weaken_threshold), those same traces are
    weakened.  The gap between the two thresholds is intentionally a dead
    band: not every response should move the needle.

    Parameters
    ----------
    reinforce_delta : float
        Amount to *add* to salience on reinforcement (default 0.05).
    weaken_delta : float
        Amount to *subtract* from salience on weakening (default 0.03).
        Weakening is deliberately gentler than reinforcement — it takes
        more bad signals to forget than good signals to remember.
    reinforce_threshold : float
        Signal health above which reinforcement occurs (default 0.7).
    weaken_threshold : float
        Signal health below which weakening occurs (default 0.4).
    """

    def __init__(
        self,
        reinforce_delta: float = 0.05,
        weaken_delta: float = 0.03,
        reinforce_threshold: float = 0.7,
        weaken_threshold: float = 0.4,
    ) -> None:
        self.reinforce_delta = reinforce_delta
        self.weaken_delta = weaken_delta
        self.reinforce_threshold = reinforce_threshold
        self.weaken_threshold = weaken_threshold

    def process(
        self,
        trace_ids: List[str],
        signal_health: float,
        episodic_store: Any,
    ) -> int:
        """
        Adjust salience of the given traces based on signal health.

        Uses ``episodic_store.reinforce()`` / ``episodic_store.weaken()``
        to persist changes directly via SQL.

        Parameters
        ----------
        trace_ids : list[str]
            IDs of traces that were loaded into context for this exchange.
        signal_health : float
            The overall health score (0-1) from the signal measurement.
        episodic_store
            Any object that exposes ``reinforce(table, id, delta)`` and
            ``weaken(table, id, delta)`` methods.

        Returns
        -------
        int
            Number of traces adjusted.
        """
        if not trace_ids:
            return 0

        adjusted = 0

        if signal_health > self.reinforce_threshold:
            for tid in trace_ids:
                try:
                    episodic_store.reinforce("traces", tid, self.reinforce_delta)
                    adjusted += 1
                except Exception as exc:
                    log.debug("Failed to reinforce trace %s: %s", tid, exc)

        elif signal_health < self.weaken_threshold:
            for tid in trace_ids:
                try:
                    episodic_store.weaken("traces", tid, self.weaken_delta)
                    adjusted += 1
                except Exception as exc:
                    log.debug("Failed to weaken trace %s: %s", tid, exc)

        # else: dead band — no adjustment

        return adjusted
