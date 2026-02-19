"""
Shared pytest fixtures for lukeZOOM tests.

Provides in-memory SQLite stores, temporary directories for YAML stores,
and pre-populated test data factories.
"""

import os
import sys
import tempfile

import pytest

# Ensure lukezoom is importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a fresh temporary directory."""
    return tmp_path


@pytest.fixture
def episodic_store(tmp_path):
    """Provide a fresh EpisodicStore backed by a temp SQLite database."""
    from lukezoom.episodic.store import EpisodicStore

    db_path = tmp_path / "test.db"
    store = EpisodicStore(db_path)
    yield store
    store.close()


@pytest.fixture
def populated_store(episodic_store):
    """EpisodicStore pre-populated with test data.

    Contains:
    - 5 messages from alice, 3 from bob
    - 4 episode traces, 1 thread trace, 1 arc trace
    - 2 events
    - 1 active session for alice
    """
    s = episodic_store

    # Messages
    for i in range(5):
        s.log_message("alice", "alice", f"Alice message {i}", "direct", salience=0.3 + i * 0.1)
    for i in range(3):
        s.log_message("bob", "bob", f"Bob message {i}", "direct", salience=0.5)

    # Traces
    for i in range(4):
        s.log_trace(f"Episode trace {i} about alice", "episode", ["alice"], salience=0.4 + i * 0.1)

    s.log_trace(
        "Thread: alice discussed quantum computing",
        "thread", ["alice", "consolidation"], salience=0.75,
        child_ids=["fake_child_1", "fake_child_2"],
    )
    s.log_trace(
        "Arc: alice's learning journey in physics",
        "arc", ["alice", "consolidation"], salience=0.9,
    )

    # Events
    s.log_event("trust_change", "Alice promoted to friend", person="alice")
    s.log_event("milestone", "First conversation", person="bob")

    # Session
    s.start_session("alice")

    return s


@pytest.fixture
def personality_system(tmp_path):
    """Provide a fresh PersonalitySystem."""
    from lukezoom.personality import PersonalitySystem
    return PersonalitySystem(storage_dir=tmp_path)


@pytest.fixture
def emotional_system(tmp_path):
    """Provide a fresh EmotionalSystem."""
    from lukezoom.emotional import EmotionalSystem
    return EmotionalSystem(storage_dir=tmp_path)


@pytest.fixture
def injury_tracker(tmp_path):
    """Provide a fresh InjuryTracker."""
    from lukezoom.safety import InjuryTracker
    return InjuryTracker(tmp_path)


@pytest.fixture
def influence_log(tmp_path):
    """Provide a fresh InfluenceLog."""
    from lukezoom.safety import InfluenceLog
    return InfluenceLog(tmp_path)


@pytest.fixture
def workspace(tmp_path):
    """Provide a fresh CognitiveWorkspace with default capacity (4)."""
    from lukezoom.working.workspace import CognitiveWorkspace
    return CognitiveWorkspace(
        capacity=4,
        storage_path=tmp_path / "workspace.json",
    )
