"""Tests for lukezoom.episodic.integrity — health diagnostics and repair."""

import json
import pytest


class TestMemoryHealth:
    def test_empty_database_healthy(self, episodic_store):
        health = episodic_store.memory_health()
        assert health["status"] == "healthy"
        assert health["fragmentation"] == 0.0
        assert health["total_traces"] == 0

    def test_fragmentation_scoring(self, episodic_store):
        # Add traces with low salience
        for i in range(10):
            episodic_store.log_trace(f"Trace {i}", "episode", ["test"],
                                     salience=0.05 if i < 7 else 0.8)
        health = episodic_store.memory_health()
        assert health["total_traces"] == 10
        assert health["low_salience_traces"] == 7
        assert health["fragmentation"] > 0.3  # Should trigger warning

    def test_status_levels(self, episodic_store):
        health = episodic_store.memory_health()
        assert health["status"] in ("healthy", "warning", "critical")


class TestRebuildFTS:
    def test_rebuild_returns_counts(self, episodic_store):
        episodic_store.log_message("alice", "alice", "msg1", "direct")
        episodic_store.log_message("alice", "alice", "msg2", "direct")
        episodic_store.log_trace("trace1", "episode", [])
        fts = episodic_store.rebuild_fts()
        assert fts["messages"] == 2
        assert fts["traces"] == 1

    def test_search_works_after_rebuild(self, episodic_store):
        episodic_store.log_message("alice", "alice", "searchable content", "direct")
        episodic_store.rebuild_fts()
        results = episodic_store.search_messages("searchable")
        assert len(results) == 1


class TestCleanupOrphans:
    def test_repairs_orphaned_marks(self, episodic_store):
        tid = episodic_store.log_trace("orphaned", "episode", ["test"])
        episodic_store.update_trace_metadata(tid, "consolidated_into", "nonexistent_id")
        counts = episodic_store.cleanup_orphaned_consolidations()
        assert counts["orphaned_marks"] == 1
        # Verify mark was removed
        trace = episodic_store.get_trace(tid)
        meta = trace.get("metadata") or {}
        assert "consolidated_into" not in meta or meta["consolidated_into"] == ""

    def test_repairs_dead_children(self, episodic_store):
        # Create a thread with child_ids pointing to non-existent traces
        tid = episodic_store.log_trace(
            "thread with dead children", "thread", ["test"],
            child_ids=["dead_1", "dead_2"],
        )
        counts = episodic_store.cleanup_orphaned_consolidations()
        assert counts["dead_children"] == 2

    def test_clean_database_no_repairs(self, episodic_store):
        episodic_store.log_trace("clean trace", "episode", ["test"])
        counts = episodic_store.cleanup_orphaned_consolidations()
        assert all(v == 0 for v in counts.values())


class TestSchema:
    def test_schema_version_set(self, episodic_store):
        from lukezoom.episodic.schema import get_schema_version, SCHEMA_VERSION
        version = get_schema_version(episodic_store.conn)
        assert version == SCHEMA_VERSION

    def test_migration_idempotent(self, episodic_store):
        from lukezoom.episodic.schema import initialize, get_schema_version, SCHEMA_VERSION
        # Running initialize again should be safe
        initialize(episodic_store.conn)
        assert get_schema_version(episodic_store.conn) == SCHEMA_VERSION
