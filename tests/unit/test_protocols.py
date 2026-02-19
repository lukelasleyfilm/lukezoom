"""Tests for lukezoom._protocols — extension point interface compliance."""

import pytest
from lukezoom._protocols import (
    MemoryLayer, ConsolidationStrategy, SignalPattern, TrustPolicy, LLMCallable,
)


class TestProtocolStructuralTyping:
    """Protocols should allow structural subtyping (duck typing)."""

    def test_memory_layer_compliance(self):
        """A class with store/recall/forget satisfies MemoryLayer."""
        class CustomLayer:
            def store(self, content, **kwargs):
                return "id123"
            def recall(self, query, limit=10):
                return []
            def forget(self, id):
                return True

        assert isinstance(CustomLayer(), MemoryLayer)

    def test_incomplete_memory_layer_rejected(self):
        """A class missing methods does NOT satisfy MemoryLayer."""
        class Incomplete:
            def store(self, content, **kwargs):
                return "id123"
            # missing recall and forget

        assert not isinstance(Incomplete(), MemoryLayer)

    def test_signal_pattern_compliance(self):
        class CustomPattern:
            @property
            def name(self):
                return "test_pattern"
            def compute(self, text, context=None):
                return 0.5

        assert isinstance(CustomPattern(), SignalPattern)

    def test_llm_callable_compliance(self):
        """Any callable with (prompt, **kwargs) -> str satisfies LLMCallable."""
        class CustomLLM:
            def __call__(self, prompt, **kwargs):
                return "response"

        assert isinstance(CustomLLM(), LLMCallable)

    def test_consolidation_strategy_compliance(self):
        class CustomStrategy:
            def should_consolidate(self, trace_count, pressure):
                return trace_count > 100
            def consolidate(self, episodic, llm_func):
                return {}

        assert isinstance(CustomStrategy(), ConsolidationStrategy)

    def test_trust_policy_compliance(self):
        class CustomPolicy:
            def evaluate(self, person, source):
                return {"tier": "friend", "memory_persistent": True,
                        "allowed_tools": [], "blocked_tools": []}
            def promote(self, person, new_tier):
                return True

        assert isinstance(CustomPolicy(), TrustPolicy)
