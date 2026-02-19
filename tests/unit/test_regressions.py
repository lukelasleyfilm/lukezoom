"""Regression tests for bugs fixed in 1.3."""

import os
import pytest
import tempfile


class TestSalienceClamping:
    """Bugs #2/#3: salience was stored unclamped in episodic store."""

    def test_negative_salience_clamped_messages(self, episodic_store):
        mid = episodic_store.log_message("a", "a", "hi", "d", salience=-5.0)
        row = episodic_store.conn.execute(
            "SELECT salience FROM messages WHERE id=?", (mid,)
        ).fetchone()
        assert row[0] == 0.0

    def test_over_one_salience_clamped_messages(self, episodic_store):
        mid = episodic_store.log_message("a", "a", "hi", "d", salience=999.0)
        row = episodic_store.conn.execute(
            "SELECT salience FROM messages WHERE id=?", (mid,)
        ).fetchone()
        assert row[0] == 1.0

    def test_negative_salience_clamped_traces(self, episodic_store):
        tid = episodic_store.log_trace("x", "episode", ["a"], salience=-2.0)
        row = episodic_store.conn.execute(
            "SELECT salience FROM traces WHERE id=?", (tid,)
        ).fetchone()
        assert row[0] == 0.0

    def test_over_one_salience_clamped_traces(self, episodic_store):
        tid = episodic_store.log_trace("x", "episode", ["a"], salience=50.0)
        row = episodic_store.conn.execute(
            "SELECT salience FROM traces WHERE id=?", (tid,)
        ).fetchone()
        assert row[0] == 1.0

    def test_negative_salience_clamped_events(self, episodic_store):
        eid = episodic_store.log_event("t", "d", salience=-1.0)
        row = episodic_store.conn.execute(
            "SELECT salience FROM events WHERE id=?", (eid,)
        ).fetchone()
        assert row[0] == 0.0

    def test_normal_salience_unchanged(self, episodic_store):
        mid = episodic_store.log_message("a", "a", "hi", "d", salience=0.75)
        row = episodic_store.conn.execute(
            "SELECT salience FROM messages WHERE id=?", (mid,)
        ).fetchone()
        assert row[0] == 0.75


class TestFullMaintenance:
    """Bug #1: full_maintenance() called non-existent .level() method."""

    def test_full_maintenance_completes(self, tmp_path):
        from lukezoom import MemorySystem

        ms = MemorySystem(data_dir=tmp_path)
        report = ms.full_maintenance()
        assert "decay" in report
        assert "health" in report
        assert "orphan_cleanup" in report
        ms.close()

    def test_full_maintenance_with_data(self, tmp_path):
        from lukezoom import MemorySystem

        ms = MemorySystem(data_dir=tmp_path)
        ms.episodic.log_message("alice", "alice", "hello", "direct")
        ms.episodic.log_trace("test trace", "episode", ["alice"], salience=0.5)
        report = ms.full_maintenance()
        assert report["health"]["total_traces"] >= 1
        ms.close()
