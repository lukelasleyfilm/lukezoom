"""
lukezoom._builder — SystemBuilder: construction logic for MemorySystem.

Extracts the ~120-line initialization from MemorySystem.__init__ into a
dedicated builder, keeping the facade class thin and focused on public API.

Usage:
    builder = SystemBuilder(config)
    components = builder.build()
    system = MemorySystem._from_components(components)

Or just use MemorySystem(data_dir=...) which calls this internally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from lukezoom.core.config import Config

log = logging.getLogger(__name__)

_NOT_SET = object()


@dataclass
class Components:
    """Container for all subsystem instances built by SystemBuilder."""

    config: Config

    # Stores
    episodic: Any = None
    semantic: Any = None
    identity: Any = None
    procedural: Any = None

    # Safety
    journal: Any = None
    influence: Any = None
    injury: Any = None

    # Search
    indexed: Any = None
    _embedding_func: Any = None
    _semantic_search: Any = None
    _unified_search: Any = None

    # Signal & learning
    signal_tracker: Any = None
    reinforcement: Any = None
    decay_engine: Any = None

    # Context
    context_builder: Any = None

    # Consolidation
    memory_pressure: Any = None
    compactor: Any = None
    consolidator: Any = None

    # Trust
    trust_gate: Any = None

    # Personality, emotional, workspace, introspection
    personality: Any = None
    emotional: Any = None
    workspace: Any = None
    introspection: Any = None

    # Consent
    _consent_path: Any = None
    _consent_optouts: set = field(default_factory=set)

    # LLM
    _llm_func: Any = _NOT_SET


class SystemBuilder:
    """Builds all MemorySystem subsystems from a Config object.

    Separates construction complexity from the public API surface.
    """

    def __init__(
        self,
        config: Config,
        embedding_func: Optional[Callable[..., list]] = None,
    ):
        self.config = config
        self.embedding_func = embedding_func

    def build(self) -> Components:
        """Construct all subsystems and return a Components container."""
        cfg = self.config
        cfg.ensure_directories()

        c = Components(config=cfg)

        # -- Stores --------------------------------------------------------
        from lukezoom.episodic.store import EpisodicStore
        from lukezoom.semantic.store import SemanticStore
        from lukezoom.semantic.identity import IdentityResolver
        from lukezoom.procedural.store import ProceduralStore

        c.episodic = EpisodicStore(cfg.db_path)
        c.semantic = SemanticStore(
            semantic_dir=cfg.semantic_dir, soul_dir=cfg.soul_dir,
        )
        c.identity = IdentityResolver(cfg.identities_path)
        c.procedural = ProceduralStore(cfg.procedural_dir)

        # -- Journal, safety -----------------------------------------------
        from lukezoom.journal import JournalStore
        from lukezoom.safety import InfluenceLog, InjuryTracker

        safety_dir = cfg.data_dir / "safety"
        c.journal = JournalStore(cfg.soul_dir / "journal")
        c.influence = InfluenceLog(safety_dir)
        c.injury = InjuryTracker(safety_dir)

        # -- Search --------------------------------------------------------
        from lukezoom.search.indexed import IndexedSearch

        c.indexed = IndexedSearch(cfg.db_path)
        c._embedding_func = self.embedding_func

        # -- Signal & learning ---------------------------------------------
        from lukezoom.signal.measure import SignalTracker
        from lukezoom.signal.reinforcement import ReinforcementEngine
        from lukezoom.signal.decay import DecayEngine

        c.signal_tracker = SignalTracker(window_size=50)
        c.reinforcement = ReinforcementEngine(
            reinforce_delta=cfg.reinforce_delta,
            weaken_delta=cfg.weaken_delta,
            reinforce_threshold=cfg.reinforce_threshold,
            weaken_threshold=cfg.weaken_threshold,
        )
        c.decay_engine = DecayEngine(half_life_hours=cfg.decay_half_life_hours)

        # -- Context builder -----------------------------------------------
        from lukezoom.working.context import ContextBuilder

        c.context_builder = ContextBuilder(
            token_budget=cfg.token_budget, config=cfg,
        )

        # -- Consolidation -------------------------------------------------
        from lukezoom.consolidation.pressure import MemoryPressure
        from lukezoom.consolidation.compactor import ConversationCompactor
        from lukezoom.consolidation.consolidator import MemoryConsolidator

        c.memory_pressure = MemoryPressure(cfg)
        c.compactor = ConversationCompactor(
            keep_recent=cfg.compaction_keep_recent,
            segment_size=cfg.compaction_segment_size,
            min_messages_to_compact=cfg.compaction_min_messages,
        )
        c.consolidator = MemoryConsolidator(
            min_episodes_per_thread=cfg.consolidation_min_episodes,
            thread_time_window_hours=cfg.consolidation_time_window_hours,
            min_threads_per_arc=cfg.consolidation_min_threads,
            max_episodes_per_run=cfg.consolidation_max_episodes_per_run,
        )

        # -- Trust gate ----------------------------------------------------
        from lukezoom.trust import TrustGate

        c.trust_gate = TrustGate(
            semantic=c.semantic, core_person=cfg.core_person,
        )
        c.trust_gate.ensure_core_person()

        # -- Personality (Big Five) ----------------------------------------
        from lukezoom.personality import PersonalitySystem

        c.personality = PersonalitySystem(storage_dir=cfg.personality_dir)

        # -- Emotional continuity (VAD) ------------------------------------
        from lukezoom.emotional import EmotionalSystem

        c.emotional = EmotionalSystem(
            storage_dir=cfg.emotional_dir,
            valence_decay=cfg.emotional_valence_decay,
            arousal_decay=cfg.emotional_arousal_decay,
            dominance_decay=cfg.emotional_dominance_decay,
        )

        # -- Cognitive workspace (Cowan's 4±1) -----------------------------
        from lukezoom.working.workspace import CognitiveWorkspace

        c.workspace = CognitiveWorkspace(
            capacity=cfg.workspace_capacity,
            decay_rate=cfg.workspace_decay_rate,
            rehearsal_boost=cfg.workspace_rehearsal_boost,
            expiry_threshold=cfg.workspace_expiry_threshold,
            storage_path=cfg.workspace_path,
            # on_evict callback is set by MemorySystem after construction
        )

        # -- Introspection -------------------------------------------------
        from lukezoom.introspection import IntrospectionLayer

        c.introspection = IntrospectionLayer(
            storage_dir=cfg.introspection_dir,
            history_days=cfg.introspection_history_days,
        )

        # -- Consent registry ----------------------------------------------
        c._consent_path = cfg.data_dir / "consent.yaml"
        c._consent_optouts = set()
        try:
            if c._consent_path.exists():
                import yaml
                with open(c._consent_path) as f:
                    data = yaml.safe_load(f) or {}
                c._consent_optouts = set(data.get("opted_out", []))
        except Exception as exc:
            log.warning("Failed to load consent registry: %s", exc)

        log.info("SystemBuilder: all components constructed (data_dir=%s)", cfg.data_dir)
        return c
