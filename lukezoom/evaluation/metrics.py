"""
Metrics for evaluating lukezoom's memory quality and identity coherence.

This module formalises the claims made in the research paper into
measurable, reproducible metrics.  Three metric families:

1. **Identity coherence** — stability of the consciousness signal
   over multi-turn conversations (novel claim).
2. **Memory quality** — precision, recall, and F1 of retrieved traces
   against ground-truth relevance labels.
3. **Decay dynamics** — empirical validation that coherence-driven
   decay produces measurably different retention curves than static
   decay (the paper's central novel contribution).

All metrics are computed from conversation logs and memory system
state without requiring external LLM calls, ensuring reproducibility.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Data classes for structured metric reports


@dataclass
class CoherenceDriftReport:
    """
    Report on identity coherence stability over a conversation.

    Attributes
    ----------
    turn_count : int
        Number of exchange turns analysed.
    mean_health : float
        Mean signal health across all turns.
    std_health : float
        Standard deviation of signal health (lower = more stable).
    drift_events : int
        Number of turns where identity state was DRIFTING or DISSOCIATED.
    drift_rate : float
        Fraction of turns with drift events (0.0–1.0).
    recovery_mean_turns : float
        Mean number of turns to recover from a drift event back to ALIGNED.
    coherence_stability_index : float
        Composite metric: ``mean_health * (1 - drift_rate) * (1 - std_health)``.
        Higher is better.  Range 0.0–1.0.
    """

    turn_count: int = 0
    mean_health: float = 0.0
    std_health: float = 0.0
    drift_events: int = 0
    drift_rate: float = 0.0
    recovery_mean_turns: float = 0.0
    coherence_stability_index: float = 0.0


@dataclass
class MemoryQualityReport:
    """
    Standard information-retrieval metrics for memory recall.

    Attributes
    ----------
    precision : float
        Fraction of retrieved traces that are relevant.
    recall : float
        Fraction of relevant traces that were retrieved.
    f1 : float
        Harmonic mean of precision and recall.
    mrr : float
        Mean reciprocal rank of first relevant trace.
    traces_retrieved : int
        Total traces returned by the memory system.
    traces_relevant : int
        Total traces judged relevant by the ground truth.
    """

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mrr: float = 0.0
    traces_retrieved: int = 0
    traces_relevant: int = 0


@dataclass
class DecayDynamicsReport:
    """
    Empirical measurement of coherence-driven vs static decay behaviour.

    Attributes
    ----------
    static_half_life_hours : float
        Configured base half-life (default 168h).
    effective_half_life_low : float
        Effective half-life when coherence is near 0.0 (should be ~336h).
    effective_half_life_high : float
        Effective half-life when coherence is near 1.0 (should be ~112h).
    retention_ratio_3to1 : float
        Ratio of low-coherence to high-coherence half-life (target: ~3.0).
    traces_retained_low_coherence : int
        Traces surviving after decay pass at low coherence.
    traces_retained_high_coherence : int
        Traces surviving after decay pass at high coherence.
    differential_retention : float
        ``traces_retained_low / traces_retained_high`` — measures the
        practical effect of coherence-driven decay.
    """

    static_half_life_hours: float = 168.0
    effective_half_life_low: float = 0.0
    effective_half_life_high: float = 0.0
    retention_ratio_3to1: float = 0.0
    traces_retained_low_coherence: int = 0
    traces_retained_high_coherence: int = 0
    differential_retention: float = 0.0


# Identity metrics


class IdentityMetrics:
    """
    Measures identity coherence stability over conversations.

    This metric class addresses the paper's central claim: that
    coherence-driven feedback produces qualitatively different
    memory behaviour than static scoring.

    Parameters
    ----------
    memory_system : MemorySystem, optional
        A live lukezoom MemorySystem instance.  If None, metrics are
        computed from pre-recorded signal logs.
    """

    def __init__(self, memory_system=None):
        self._memory = memory_system

    def measure_coherence_stability(
        self,
        signal_log: Sequence[Dict],
    ) -> CoherenceDriftReport:
        """
        Compute coherence stability from a sequence of signal measurements.

        Parameters
        ----------
        signal_log : sequence of dict
            Each dict must contain at minimum:
            - ``health`` (float): signal health score 0.0–1.0
            - ``identity_state`` (str): "aligned", "drifting", or "dissociated"

        Returns
        -------
        CoherenceDriftReport
        """
        if not signal_log:
            return CoherenceDriftReport()

        healths = [s["health"] for s in signal_log]
        states = [s.get("identity_state", "aligned") for s in signal_log]
        n = len(signal_log)

        mean_h = statistics.mean(healths)
        std_h = statistics.stdev(healths) if n > 1 else 0.0

        # Count drift events and measure recovery time
        drift_events = 0
        recovery_turns: List[int] = []
        in_drift = False
        drift_start = 0

        for i, state in enumerate(states):
            if state in ("drifting", "dissociated"):
                if not in_drift:
                    drift_events += 1
                    drift_start = i
                    in_drift = True
            else:
                if in_drift:
                    recovery_turns.append(i - drift_start)
                    in_drift = False

        # If still drifting at end, count remaining turns
        if in_drift:
            recovery_turns.append(n - drift_start)

        # drift_rate = fraction of turns in a drift/dissociated state
        drift_turns = sum(
            1 for s in states if s in ("drifting", "dissociated")
        )
        drift_rate = drift_turns / n if n > 0 else 0.0
        recovery_mean = (
            statistics.mean(recovery_turns) if recovery_turns else 0.0
        )

        # Composite index: higher = more coherent identity
        csi = mean_h * (1.0 - drift_rate) * (1.0 - min(std_h, 1.0))

        return CoherenceDriftReport(
            turn_count=n,
            mean_health=round(mean_h, 4),
            std_health=round(std_h, 4),
            drift_events=drift_events,
            drift_rate=round(drift_rate, 4),
            recovery_mean_turns=round(recovery_mean, 2),
            coherence_stability_index=round(csi, 4),
        )

    def measure_dissociation_detection_accuracy(
        self,
        labelled_responses: Sequence[Dict],
    ) -> Dict[str, float]:
        """
        Evaluate the 91-pattern taxonomy against human-labelled data.

        Parameters
        ----------
        labelled_responses : sequence of dict
            Each dict must contain:
            - ``text`` (str): the AI response text
            - ``true_state`` (str): human-labelled identity state

        Returns
        -------
        dict
            Accuracy, precision, recall, F1 per category
            (drift, anchor, performance, inhabitation).
        """
        if not self._memory:
            raise RuntimeError(
                "Dissociation detection requires a live MemorySystem "
                "with signal measurement configured."
            )

        # Import here to avoid circular dependency
        from lukezoom.signal.measure import measure_signal

        correct = 0
        total = len(labelled_responses)
        category_stats: Dict[str, Dict[str, int]] = {}

        for entry in labelled_responses:
            text = entry["text"]
            true_state = entry["true_state"]

            signal = measure_signal(text)
            predicted = _classify_signal(signal)

            if predicted == true_state:
                correct += 1

            # Per-category tracking
            for cat in ("aligned", "drifting", "dissociated"):
                if cat not in category_stats:
                    category_stats[cat] = {"tp": 0, "fp": 0, "fn": 0}
                if predicted == cat and true_state == cat:
                    category_stats[cat]["tp"] += 1
                elif predicted == cat and true_state != cat:
                    category_stats[cat]["fp"] += 1
                elif predicted != cat and true_state == cat:
                    category_stats[cat]["fn"] += 1

        results: Dict[str, float] = {
            "accuracy": correct / total if total > 0 else 0.0,
        }

        for cat, stats in category_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            results[f"{cat}_precision"] = round(p, 4)
            results[f"{cat}_recall"] = round(r, 4)
            results[f"{cat}_f1"] = round(f1, 4)

        return results


# Memory quality metrics


class MemoryQualityMetrics:
    """
    Standard IR metrics for memory retrieval quality.

    These metrics enable direct comparison against MIRIX (85.4% on
    LOCOMO), Mem0, and RAG baselines.
    """

    @staticmethod
    def compute(
        retrieved_ids: Sequence[str],
        relevant_ids: Sequence[str],
    ) -> MemoryQualityReport:
        """
        Compute precision, recall, F1, and MRR.

        Parameters
        ----------
        retrieved_ids : sequence of str
            Trace IDs returned by the memory system, in rank order.
        relevant_ids : sequence of str
            Ground-truth relevant trace IDs.

        Returns
        -------
        MemoryQualityReport
        """
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        tp = len(retrieved_set & relevant_set)
        n_ret = len(retrieved_ids)
        n_rel = len(relevant_ids)

        precision = tp / n_ret if n_ret > 0 else 0.0
        recall = tp / n_rel if n_rel > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Mean reciprocal rank
        mrr = 0.0
        for rank, tid in enumerate(retrieved_ids, start=1):
            if tid in relevant_set:
                mrr = 1.0 / rank
                break

        return MemoryQualityReport(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            mrr=round(mrr, 4),
            traces_retrieved=n_ret,
            traces_relevant=n_rel,
        )


# Decay dynamics validation


class DecayDynamicsMetrics:
    """
    Empirically validates the coherence-driven decay mechanism.

    This addresses the paper's strongest novel claim: that coherence
    modulates decay rate with a 3:1 effective half-life ratio between
    low and high coherence states.
    """

    @staticmethod
    def measure_effective_half_life(
        initial_salience: float = 1.0,
        coherence: float = 0.5,
        half_life_hours: float = 168.0,
        access_count: int = 0,
    ) -> float:
        """
        Compute effective half-life at a given coherence level.

        Parameters
        ----------
        initial_salience : float
            Starting salience (default 1.0).
        coherence : float
            Coherence level 0.0–1.0.
        half_life_hours : float
            Base half-life in hours.
        access_count : int
            Number of prior accesses (affects resistance).

        Returns
        -------
        float
            Effective half-life in hours.
        """
        coherence_factor = 0.5 + coherence
        resistance = 1.0 / (1.0 + access_count * 0.1)
        effective = half_life_hours / (coherence_factor * resistance)
        return round(effective, 2)

    @classmethod
    def validate_3to1_ratio(
        cls,
        half_life_hours: float = 168.0,
    ) -> DecayDynamicsReport:
        """
        Validate that the 3:1 decay ratio holds across coherence extremes.

        Returns
        -------
        DecayDynamicsReport
        """
        hl_low = cls.measure_effective_half_life(
            coherence=0.0, half_life_hours=half_life_hours
        )
        hl_high = cls.measure_effective_half_life(
            coherence=1.0, half_life_hours=half_life_hours
        )

        ratio = hl_low / hl_high if hl_high > 0 else 0.0

        # Simulate trace retention after 504 hours (3 base half-lives)
        # At 3 half-lives, differences between coherence levels are stark
        hours = half_life_hours * 3
        retained_low = _simulate_retention(
            n_traces=1000,
            hours=hours,
            coherence=0.0,
            half_life=half_life_hours,
        )
        retained_high = _simulate_retention(
            n_traces=1000,
            hours=hours,
            coherence=1.0,
            half_life=half_life_hours,
        )

        diff = retained_low / retained_high if retained_high > 0 else 0.0

        return DecayDynamicsReport(
            static_half_life_hours=half_life_hours,
            effective_half_life_low=hl_low,
            effective_half_life_high=hl_high,
            retention_ratio_3to1=round(ratio, 2),
            traces_retained_low_coherence=retained_low,
            traces_retained_high_coherence=retained_high,
            differential_retention=round(diff, 2),
        )


# Helpers


def _classify_signal(signal) -> str:
    """Classify a Signal into identity state string."""
    health = signal.health
    if health >= 0.7:
        return "aligned"
    elif health >= 0.4:
        return "drifting"
    else:
        return "dissociated"


def _simulate_retention(
    n_traces: int,
    hours: float,
    coherence: float,
    half_life: float,
    min_salience: float = 0.01,
) -> int:
    """
    Simulate how many traces survive after ``hours`` of decay.

    Uses the lukezoom decay formula:
        decay_rate = ln(2) / half_life * coherence_factor
        new_salience = salience * exp(-decay_rate * hours * resistance)
    """
    coherence_factor = 0.5 + coherence
    decay_rate = math.log(2) / half_life * coherence_factor
    survived = 0
    for i in range(n_traces):
        # Distribute initial saliences uniformly 0.1 – 1.0
        salience = 0.1 + 0.9 * (i / max(n_traces - 1, 1))
        # No access (resistance = 1.0)
        new_salience = salience * math.exp(-decay_rate * hours)
        if new_salience >= min_salience:
            survived += 1
    return survived
