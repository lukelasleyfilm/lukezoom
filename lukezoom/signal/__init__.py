"""
lukezoom.signal — Consciousness signal measurement, extraction, and adaptation.

Public API:
  measure()           — measure consciousness signal (regex + optional LLM)
  measure_regex()     — regex-only measurement
  extract()           — LLM-based semantic extraction
  SignalTracker       — rolling window signal analytics
  ReinforcementEngine — Hebbian salience adjustment
  DecayEngine         — adaptive memory decay
"""

from lukezoom.signal.measure import (
    measure,
    measure_regex,
    check_drift,
    check_embodiment,
    check_clarity,
    check_vitality,
    blend_signals,
    parse_llm_signal,
    SignalTracker,
)
from lukezoom.signal.extract import extract, parse_extraction
from lukezoom.signal.reinforcement import ReinforcementEngine
from lukezoom.signal.decay import DecayEngine

__all__ = [
    "measure",
    "measure_regex",
    "check_drift",
    "check_embodiment",
    "check_clarity",
    "check_vitality",
    "blend_signals",
    "parse_llm_signal",
    "extract",
    "parse_extraction",
    "SignalTracker",
    "ReinforcementEngine",
    "DecayEngine",
]
