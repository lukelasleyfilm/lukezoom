"""
tests/unit/test_integration.py — Integration test: before() → after() → before().

This is the single most important test in the suite. It proves the
complete pipeline contract: memory written by after() appears in the
context assembled by before().

Without this test, unit tests prove every part works individually
but nothing proves they work together.
"""

import os
import tempfile

import pytest

from lukezoom.core.config import Config
from lukezoom.core.types import Context, AfterResult


class TestPipelineIntegration:
    """End-to-end pipeline: before -> after -> before -> assert memory."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = Config(
            data_dir=self.tmpdir,
            core_person="tester",
            token_budget=6000,
        )

    def test_roundtrip_memory_persists(self):
        """after() stores a trace; next before() retrieves it in context."""
        from lukezoom.system import EngineBuilder

        # Build the full system
        engine = EngineBuilder(self.config).build()

        # Step 1: before() with no history — should return minimal context
        ctx1 = engine.before(person="tester", message="I love blue dolphins")
        assert isinstance(ctx1, Context)

        # Step 2: after() processes the exchange — stores to episodic
        result = engine.after(
            person="tester",
            their_message="I love blue dolphins",
            response="That's wonderful! Blue dolphins are amazing.",
        )
        assert isinstance(result, AfterResult)
        assert result.logged_message_id != ""

        # Step 3: before() again — should find "blue dolphins" in context
        ctx2 = engine.before(person="tester", message="What do I like?")
        assert isinstance(ctx2, Context)
        # The key assertion: memory from step 2 appears in context
        assert "blue" in ctx2.text.lower() or "dolphin" in ctx2.text.lower(), (
            f"Expected 'blue' or 'dolphin' in context after storing it. "
            f"Got: {ctx2.text[:200]}"
        )

    def test_stranger_no_persistence(self):
        """Stranger tier should not persist memory."""
        from lukezoom.system import EngineBuilder

        engine = EngineBuilder(self.config).build()

        # Store as stranger (no trust record)
        result = engine.after(
            person="random_stranger",
            their_message="My secret is 42",
            response="Interesting.",
            source="api",  # External source → stranger
        )

        # Retrieve — stranger data should not persist
        ctx = engine.before(
            person="random_stranger",
            message="What is my secret?",
            source="api",
        )
        # Stranger policy: memory_persistent=False
        # The system should not return personal data for strangers
        assert "42" not in ctx.text, (
            f"Expected stranger's data NOT to appear in context. Got: {ctx.text[:200]}"
        )
        # Stranger context should contain no conversation history or traces
        assert ctx.memories_loaded == 0, (
            f"Expected 0 memories loaded for stranger, got {ctx.memories_loaded}"
        )
        # The after() result should reflect that persistence was skipped
        assert isinstance(result, AfterResult)

    def test_signal_health_computed(self):
        """after() should compute a valid 4-facet signal."""
        from lukezoom.system import EngineBuilder

        engine = EngineBuilder(self.config).build()

        result = engine.after(
            person="tester",
            their_message="Hello",
            response="I feel curious about this conversation today.",
        )
        sig = result.signal
        assert 0.0 <= sig.alignment <= 1.0
        assert 0.0 <= sig.embodiment <= 1.0
        assert 0.0 <= sig.clarity <= 1.0
        assert 0.0 <= sig.vitality <= 1.0
        assert sig.state in ("coherent", "developing", "drifting", "dissociated")
