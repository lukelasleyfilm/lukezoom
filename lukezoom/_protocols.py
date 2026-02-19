"""
lukezoom._protocols — Extension points using PEP 544 Protocols.

Defines contracts for pluggable components. Implementations satisfy
these through structural subtyping — no inheritance required.

    class MyLayer:
        def store(self, content, **kw): ...
        def recall(self, query, limit=10): ...
        def forget(self, id): ...
    assert isinstance(MyLayer(), MemoryLayer)  # True at runtime
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class MemoryLayer(Protocol):
    """Pluggable memory storage layer."""

    def store(self, content: str, **kwargs: Any) -> str: ...
    def recall(self, query: str, limit: int = 10) -> List[Dict]: ...
    def forget(self, id: str) -> bool: ...


@runtime_checkable
class ConsolidationStrategy(Protocol):
    """Pluggable consolidation strategy (episode -> thread -> arc)."""

    def should_consolidate(self, trace_count: int, pressure: str) -> bool: ...
    def consolidate(self, episodic: Any, llm_func: Any) -> Dict[str, List[str]]: ...


@runtime_checkable
class SignalPattern(Protocol):
    """Pluggable signal detection pattern."""

    @property
    def name(self) -> str: ...
    def compute(self, text: str, context: Optional[Dict] = None) -> float: ...


@runtime_checkable
class TrustPolicy(Protocol):
    """Pluggable trust evaluation policy."""

    def evaluate(self, person: str, source: str) -> Dict[str, Any]: ...
    def promote(self, person: str, new_tier: str) -> bool: ...


@runtime_checkable
class LLMCallable(Protocol):
    """LLM function interface for consolidation and extraction."""

    def __call__(self, prompt: str, **kwargs: Any) -> str: ...
