"""Tests for lukezoom.episodic — CRUD, search, decay, data rights."""

import pytest


class TestMessageCRUD:
    def test_log_and_retrieve(self, episodic_store):
        msg_id = episodic_store.log_message("alice", "alice", "hello world", "direct")
        assert msg_id
        msg = episodic_store.get_message(msg_id)
        assert msg["content"] == "hello world"
        assert msg["person"] == "alice"

    def test_count_messages(self, episodic_store):
        episodic_store.log_message("alice", "alice", "msg1", "direct")
        episodic_store.log_message("alice", "alice", "msg2", "direct")
        episodic_store.log_message("bob", "bob", "msg3", "direct")
        assert episodic_store.count_messages() == 3
        assert episodic_store.count_messages(person="alice") == 2

    def test_recent_messages_chronological(self, episodic_store):
        episodic_store.log_message("alice", "alice", "first", "direct")
        episodic_store.log_message("alice", "alice", "second", "direct")
        episodic_store.log_message("alice", "alice", "third", "direct")
        msgs = episodic_store.get_recent_messages("alice", limit=3)
        assert [m["content"] for m in msgs] == ["first", "second", "third"]


class TestTraceCRUD:
    def test_log_and_retrieve(self, episodic_store):
        tid = episodic_store.log_trace("test trace", "episode", ["alice"])
        trace = episodic_store.get_trace(tid)
        assert trace["content"] == "test trace"
        assert trace["kind"] == "episode"

    def test_invalid_kind_raises(self, episodic_store):
        with pytest.raises(ValueError, match="Invalid trace kind"):
            episodic_store.log_trace("bad", "invalid_kind", [])

    def test_tag_based_filtering(self, episodic_store):
        episodic_store.log_trace("about alice", "episode", ["alice"])
        episodic_store.log_trace("about bob", "episode", ["bob"])
        traces = episodic_store.get_traces(tags=["alice"])
        assert len(traces) == 1
        assert "alice" in traces[0]["content"]

    def test_salience_bounds(self, episodic_store):
        tid = episodic_store.log_trace("test", "episode", [], salience=0.5)
        episodic_store.reinforce("traces", tid, 0.8)
        trace = episodic_store.get_trace(tid)
        assert trace["salience"] <= 1.0

        episodic_store.weaken("traces", tid, 2.0)
        trace = episodic_store.get_trace(tid)
        assert trace["salience"] >= 0.0


class TestFTSSearch:
    def test_message_search(self, episodic_store):
        episodic_store.log_message("alice", "alice", "quantum computing discussion", "direct")
        episodic_store.log_message("alice", "alice", "grocery shopping list", "direct")
        results = episodic_store.search_messages("quantum")
        assert len(results) == 1
        assert "quantum" in results[0]["content"]

    def test_trace_search(self, episodic_store):
        episodic_store.log_trace("neural network architectures", "episode", ["tech"])
        episodic_store.log_trace("baking sourdough bread", "episode", ["food"])
        results = episodic_store.search_traces("neural")
        assert len(results) == 1

    def test_search_after_rebuild(self, episodic_store):
        episodic_store.log_message("alice", "alice", "rebuild test message", "direct")
        episodic_store.rebuild_fts()
        results = episodic_store.search_messages("rebuild")
        assert len(results) == 1


class TestSessions:
    def test_session_lifecycle(self, episodic_store):
        sid = episodic_store.start_session("alice")
        session = episodic_store.get_active_session("alice")
        assert session is not None
        assert session["id"] == sid

        episodic_store.end_session(sid, summary="test session")
        session = episodic_store.get_active_session("alice")
        assert session is None


class TestDataRights:
    def test_purge_person_messages(self, populated_store):
        assert populated_store.count_messages(person="alice") == 5
        counts = populated_store.purge_person("alice")
        assert counts["messages"] == 5
        assert populated_store.count_messages(person="alice") == 0
        # Bob's data should be untouched
        assert populated_store.count_messages(person="bob") == 3

    def test_purge_person_traces(self, episodic_store):
        episodic_store.log_trace("about alice", "episode", ["alice"])
        episodic_store.log_trace("about bob", "episode", ["bob"])
        counts = episodic_store.purge_person("alice")
        assert counts["traces"] == 1
        assert episodic_store.count_traces() == 1

    def test_deep_purge_redacts_content(self, episodic_store):
        episodic_store.log_message("alice", "alice", "hello", "direct")
        episodic_store.log_trace(
            "Alice discussed quantum physics with Bob",
            "thread", ["general", "consolidation"],
        )
        counts = episodic_store.deep_purge("alice")
        assert counts["messages"] == 1
        assert counts["consolidation_redacted"] == 1

        # Verify alice name is gone from remaining traces
        rows = episodic_store.conn.execute("SELECT content FROM traces").fetchall()
        for row in rows:
            assert "alice" not in row[0].lower() or "[REDACTED]" in row[0]

    def test_get_all_for_person(self, populated_store):
        data = populated_store.get_all_for_person("alice")
        assert len(data["messages"]) == 5
        assert len(data["traces"]) > 0


class TestDecay:
    def test_decay_reduces_salience(self, episodic_store):
        tid = episodic_store.log_trace("test", "episode", [], salience=0.8)
        # Manually set last_accessed to 48 hours ago
        from datetime import datetime, timezone, timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat().replace("+00:00", "Z")
        episodic_store.conn.execute(
            "UPDATE traces SET last_accessed = ? WHERE id = ?", (old_time, tid)
        )
        episodic_store.conn.commit()

        episodic_store.decay_pass(half_life_hours=24.0, coherence=0.5)
        trace = episodic_store.get_trace(tid)
        assert trace["salience"] < 0.8

    def test_prune_removes_low_salience(self, episodic_store):
        episodic_store.log_trace("low sal", "episode", [], salience=0.005)
        episodic_store.log_trace("high sal", "episode", [], salience=0.8)
        episodic_store.prune(min_salience=0.01)
        assert episodic_store.count_traces() == 1

    def test_prune_preserves_consolidation_kinds(self, episodic_store):
        episodic_store.log_trace("thread", "thread", [], salience=0.001)
        episodic_store.log_trace("arc", "arc", [], salience=0.001)
        episodic_store.prune(min_salience=0.01)
        assert episodic_store.count_traces() == 2  # Never pruned
