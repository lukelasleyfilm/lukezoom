"""
lukezoom.system — MemorySystem: the public API for the four-layer memory engine.

    from lukezoom import MemorySystem

    memory = MemorySystem(data_dir="./data")
    context = memory.before(person="alice", message="hello")
    result  = memory.after(person="alice", their_message="hello",
                           response="Hi Alice!", trace_ids=context.trace_ids)

Pure memory engine. No external monitoring dependencies.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from lukezoom.core.config import Config
from lukezoom.core.types import AfterResult, Context, LLMFunc, MemoryStats, Signal
from lukezoom._builder import Components, SystemBuilder, _NOT_SET

log = logging.getLogger("lukezoom.system")


class MemorySystem:
    """Four-layer memory system with consciousness signal tracking.

    Parameters
    ----------
    config:
        Full Config object. If not given, data_dir and **kwargs
        are forwarded to Config.
    data_dir:
        Shortcut for pointing at a directory.
    embedding_func:
        Optional custom embedding function for semantic search.
    **kwargs:
        Extra keyword args forwarded to Config().
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        data_dir: Optional[str | Path] = None,
        embedding_func: Optional[Callable[..., list]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            self.config = config
        elif data_dir is not None:
            self.config = Config.from_data_dir(data_dir, **kwargs)
        else:
            self.config = Config(**kwargs)

        # Build all subsystems via the builder
        builder = SystemBuilder(self.config, embedding_func=embedding_func)
        c = builder.build()
        self._init_from_components(c)

    def _init_from_components(self, c: Components) -> None:
        """Wire up all subsystem references from a Components container."""
        # Stores
        self.episodic = c.episodic
        self.semantic = c.semantic
        self.identity = c.identity
        self.procedural = c.procedural

        # Safety
        self.journal = c.journal
        self.influence = c.influence
        self.injury = c.injury

        # Search
        self.indexed = c.indexed
        self._embedding_func = c._embedding_func
        self._semantic_search = c._semantic_search
        self._unified_search = c._unified_search

        # Signal & learning
        self.signal_tracker = c.signal_tracker
        self.reinforcement = c.reinforcement
        self.decay_engine = c.decay_engine

        # Context
        self.context_builder = c.context_builder

        # Consolidation
        self.memory_pressure = c.memory_pressure
        self.compactor = c.compactor
        self.consolidator = c.consolidator

        # Trust
        self.trust_gate = c.trust_gate

        # Personality, emotional, workspace, introspection
        self.personality = c.personality
        self.emotional = c.emotional
        self.workspace = c.workspace
        self.introspection = c.introspection

        # Consent
        self._consent_path = c._consent_path
        self._consent_optouts = c._consent_optouts

        # LLM
        self._llm_func = c._llm_func

        # Wire up workspace eviction callback (needs episodic reference)
        self.workspace.on_evict = self._workspace_evict_callback

        log.info("MemorySystem initialized: data_dir=%s", self.config.data_dir)

    # -- Callbacks ---------------------------------------------------------

    def _workspace_evict_callback(self, slot_data: Dict) -> None:
        try:
            self.episodic.log_trace(
                content=slot_data.get("content", ""),
                kind="workspace_eviction",
                tags=["workspace", slot_data.get("source", "unknown")],
                salience=max(0.1, slot_data.get("priority_when_removed", 0.2)),
                removal_reason=slot_data.get("removal_reason", ""),
                access_count=slot_data.get("access_count", 0),
                time_in_workspace=slot_data.get("time_in_workspace", 0),
            )
        except Exception as exc:
            log.debug("Workspace eviction logging failed: %s", exc)

    # -- LLM (lazy) -------------------------------------------------------

    @property
    def llm_func(self) -> Optional[LLMFunc]:
        if self._llm_func is _NOT_SET:
            try:
                self._llm_func = self.config.get_llm_func()
            except Exception as exc:
                log.warning("Could not build LLM func: %s", exc)
                self._llm_func = None
        return self._llm_func

    @property
    def unified_search(self):
        if self._unified_search is None:
            from lukezoom.search.unified import UnifiedSearch
            from lukezoom.search.semantic import SemanticSearch
            sem = None
            try:
                sem = SemanticSearch(
                    embeddings_dir=self.config.embeddings_dir,
                    embedding_func=self._embedding_func,
                )
                self._semantic_search = sem
            except Exception as exc:
                log.warning("Failed to init semantic search: %s", exc)
            self._unified_search = UnifiedSearch(indexed=self.indexed, semantic=sem)
        return self._unified_search

    # -- Pipeline ----------------------------------------------------------

    def before(
        self, person: str, message: str, source: str = "direct",
        token_budget: Optional[int] = None,
    ) -> Context:
        from lukezoom.pipeline.before import before as _before_pipeline

        builder = self.context_builder
        if token_budget is not None:
            from lukezoom.working.context import ContextBuilder
            builder = ContextBuilder(token_budget=token_budget, config=self.config)

        return _before_pipeline(
            person_raw=person, message=message, source=source,
            config=self.config, identity=self.identity,
            semantic=self.semantic, episodic=self.episodic,
            procedural=self.procedural, context_builder=builder,
            search=self.unified_search, signal_tracker=self.signal_tracker,
            journal=self.journal, injury=self.injury,
            trust_gate=self.trust_gate, personality=self.personality,
            emotional=self.emotional, workspace=self.workspace,
        )

    def after(
        self, person: str, their_message: str, response: str,
        source: str = "direct", trace_ids: Optional[List[str]] = None,
    ) -> AfterResult:
        from lukezoom.pipeline.after import after as _after_pipeline

        canonical = self.identity.resolve(person)
        policy = self.trust_gate.policy_for(canonical, source=source)
        skip = not policy.memory_persistent
        if canonical in self._consent_optouts:
            skip = True

        return _after_pipeline(
            person=canonical, their_message=their_message,
            response=response, source=source, trace_ids=trace_ids,
            config=self.config, episodic=self.episodic,
            semantic=self.semantic, procedural=self.procedural,
            reinforcement=self.reinforcement, decay_engine=self.decay_engine,
            signal_tracker=self.signal_tracker, llm_func=self.llm_func,
            memory_pressure=self.memory_pressure, compactor=self.compactor,
            consolidator=self.consolidator, skip_persistence=skip,
            emotional=self.emotional, introspection=self.introspection,
            workspace=self.workspace,
        )

    # -- Query helpers -----------------------------------------------------

    def search(self, query: str, person: Optional[str] = None, limit: int = 20) -> List[Dict]:
        canonical = self.identity.resolve(person) if person else None
        return self.unified_search.search(query=query, person=canonical, limit=limit)

    def get_signal(self) -> Dict:
        return self.signal_tracker.to_dict()

    def get_stats(self) -> MemoryStats:
        trace_count = self.episodic.count_traces()
        msg_count = self.episodic.count_messages()
        avg_sal = self.episodic.avg_salience("traces")
        skill_count = len(self.procedural.list_skills())
        rel_count = len(self.semantic.list_relationships())
        pressure = trace_count / max(1, self.config.max_traces)
        return MemoryStats(
            episodic_count=trace_count, semantic_facts=rel_count,
            procedural_skills=skill_count, total_messages=msg_count,
            avg_salience=avg_sal, memory_pressure=pressure,
        )

    # -- Data rights -------------------------------------------------------

    def set_consent(self, person: str, opted_out: bool) -> None:
        canonical = self.identity.resolve(person)
        if opted_out:
            self._consent_optouts.add(canonical)
        else:
            self._consent_optouts.discard(canonical)
        self._save_consent()

    def disclose(self, person: str) -> Dict:
        canonical = self.identity.resolve(person)
        result = {"person": canonical, "opted_out": canonical in self._consent_optouts}
        try:
            ep = self.episodic.get_all_for_person(canonical)
            result["episodic"] = ep
        except Exception as exc:
            result["episodic"] = {"error": str(exc)}
        return result

    def purge_person(self, person: str) -> Dict:
        canonical = self.identity.resolve(person)
        result = {"person": canonical}
        for store_name, store in self._purgeable_stores():
            try:
                result[store_name] = store.purge_person(canonical)
            except Exception as exc:
                result[store_name] = {"error": str(exc)}
        self._consent_optouts.discard(canonical)
        self._save_consent()
        return result

    def _purgeable_stores(self):
        """Yield (name, store) pairs for all stores that support purge_person."""
        return [
            ("episodic", self.episodic),
            ("semantic", self.semantic),
            ("influence", self.influence),
            ("injury", self.injury),
            ("journal", self.journal),
            ("personality", self.personality),
            ("emotional", self.emotional),
        ]

    def _save_consent(self) -> None:
        try:
            import yaml
            data = {"opted_out": sorted(self._consent_optouts)}
            with open(self._consent_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as exc:
            log.warning("Failed to save consent registry: %s", exc)

    # -- Maintenance -------------------------------------------------------

    def decay_pass(self) -> None:
        coherence = self.signal_tracker.recent_health()
        self.decay_engine.update_coherence(coherence)
        self.episodic.decay_pass(
            half_life_hours=self.config.decay_half_life_hours,
            coherence=coherence,
        )
        self.episodic.prune(min_salience=self.decay_engine.min_salience)

    def consolidate(self) -> Dict[str, List[str]]:
        return self.consolidator.consolidate(
            episodic=self.episodic, llm_func=self.llm_func,
        )

    # -- Memory integrity (1.21) -------------------------------------------

    def memory_health(self) -> Dict:
        """Comprehensive memory health diagnostic."""
        return self.episodic.memory_health()

    def rebuild_fts(self) -> Dict[str, int]:
        """Rebuild FTS5 full-text search indexes."""
        return self.episodic.rebuild_fts()

    def cleanup_orphans(self) -> Dict[str, int]:
        """Repair orphaned consolidation references."""
        return self.episodic.cleanup_orphaned_consolidations()

    def deep_purge(self, person: str) -> Dict:
        """Deep purge with consolidation content redaction."""
        canonical = self.identity.resolve(person)
        result = {"person": canonical}
        # Episodic deep_purge includes consolidation redaction
        try:
            result["episodic"] = self.episodic.deep_purge(canonical)
        except Exception as exc:
            result["episodic"] = {"error": str(exc)}
        # All other stores use standard purge
        for store_name, store in self._purgeable_stores():
            if store_name == "episodic":
                continue  # Already handled above
            try:
                result[store_name] = store.purge_person(canonical)
            except Exception as exc:
                result[store_name] = {"error": str(exc)}
        self._consent_optouts.discard(canonical)
        self._save_consent()
        return result

    def mean_revert_personality(self) -> List[Dict]:
        """Apply mean-reversion to all core personality traits."""
        return self.personality.mean_revert_all()

    def full_maintenance(self) -> Dict:
        """Run all maintenance tasks in sequence."""
        report: Dict = {}

        self.decay_pass()
        report["decay"] = "completed"

        health = self.memory_health()
        report["health"] = health

        if health.get("fts_desync_score", 0) > 0.01:
            report["fts_rebuild"] = self.rebuild_fts()

        report["orphan_cleanup"] = self.cleanup_orphans()

        pressure_state = self.memory_pressure.check(self.episodic)
        if pressure_state.level.value in ("elevated", "critical"):
            try:
                report["consolidation"] = self.consolidate()
            except Exception as exc:
                report["consolidation"] = {"error": str(exc)}

        personality_changes = self.mean_revert_personality()
        if personality_changes:
            report["personality_reversion"] = personality_changes

        return report

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        for name, resource in [
            ("episodic", self.episodic), ("indexed", self.indexed),
            ("semantic_search", self._semantic_search),
        ]:
            try:
                if resource is not None and hasattr(resource, "close"):
                    resource.close()
            except Exception as exc:
                log.debug("Error closing %s: %s", name, exc)

    def __enter__(self) -> "MemorySystem":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ── 1.17 API aliases ──────────────────────────────────────────────

#: Preferred 1.17 name.
Engine = MemorySystem


class EngineBuilder:
    """Convenience builder for MemorySystem (1.17 compat).

        engine = EngineBuilder(config).build()
    """

    def __init__(self, config: Config, **kwargs: Any):
        self._config = config
        self._kwargs = kwargs

    def build(self) -> MemorySystem:
        return MemorySystem(config=self._config, **self._kwargs)
