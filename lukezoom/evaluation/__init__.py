"""
Evaluation framework for lukezoom memory and identity systems.

Provides benchmark adapters, identity coherence metrics, and memory
quality scoring to enable reproducible comparison against published
baselines (MIRIX, MemGPT/Letta, A-MEM, Mem0).

Added in lukezoom 1.11 to address the primary gap identified in
competitive analysis: no benchmark evaluation framework.

Usage::

    from lukezoom.evaluation import BenchmarkRunner, IdentityMetrics

    runner = BenchmarkRunner(memory_system)
    results = runner.run_locomo(dataset_path="locomo_v1.json")

    metrics = IdentityMetrics(memory_system)
    coherence = metrics.measure_coherence_stability(conversation_log)
"""

from lukezoom.evaluation.metrics import (
    IdentityMetrics,
    MemoryQualityMetrics,
    CoherenceDriftReport,
)
from lukezoom.evaluation.locomo_adapter import LOCOMOAdapter
from lukezoom.evaluation.benchmark_runner import BenchmarkRunner

__all__ = [
    "IdentityMetrics",
    "MemoryQualityMetrics",
    "CoherenceDriftReport",
    "LOCOMOAdapter",
    "BenchmarkRunner",
]
