"""
lukezoom.pipeline.after -- Post-LLM logging and learning pipeline.

Called automatically after every LLM response.  Performs:
  1. Measure consciousness signal (hybrid: regex + LLM judge)
     + track signal in rolling window
  2. Derive salience from signal health
  3. Session boundary detection + auto-management
  4. Log the exchange to episodic memory (both messages + trace)
  5. Hebbian reinforcement on context traces
  6. Semantic extraction + apply updates (LLM-based, optional)
  7. Pressure-aware decay, compaction, and consolidation

Everything here is fire-and-forget -- failures in any step are
logged and swallowed so the user never sees memory-system errors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from lukezoom.core.types import AfterResult, HealthBitmap, LLMFunc, Signal

if TYPE_CHECKING:
    from lukezoom.consolidation.compactor import ConversationCompactor
    from lukezoom.consolidation.consolidator import MemoryConsolidator
    from lukezoom.consolidation.pressure import MemoryPressure
    from lukezoom.core.config import Config
    from lukezoom.emotional import EmotionalSystem
    from lukezoom.episodic.store import EpisodicStore
    from lukezoom.introspection import IntrospectionLayer
    from lukezoom.procedural.store import ProceduralStore
    from lukezoom.semantic.store import SemanticStore
    from lukezoom.signal.decay import DecayEngine
    from lukezoom.signal.measure import SignalTracker
    from lukezoom.signal.reinforcement import ReinforcementEngine
    from lukezoom.working.workspace import CognitiveWorkspace

log = logging.getLogger("lukezoom.pipeline.after")


def after(
    *,
    person: str,
    their_message: str,
    response: str,
    source: str = "direct",
    trace_ids: Optional[List[str]] = None,
    config: "Config",
    episodic: "EpisodicStore",
    semantic: "SemanticStore",
    procedural: "ProceduralStore",
    reinforcement: "ReinforcementEngine",
    decay_engine: "DecayEngine",
    signal_tracker: "SignalTracker",
    llm_func: Optional[LLMFunc] = None,
    memory_pressure: Optional["MemoryPressure"] = None,
    compactor: Optional["ConversationCompactor"] = None,
    consolidator: Optional["MemoryConsolidator"] = None,
    skip_persistence: bool = False,
    emotional: Optional["EmotionalSystem"] = None,
    introspection: Optional["IntrospectionLayer"] = None,
    workspace: Optional["CognitiveWorkspace"] = None,
) -> AfterResult:
    """Run the full after-pipeline and return results."""
    trace_ids = trace_ids or []
    health = HealthBitmap()
    updates: List[Dict] = []

    # -- 1. Measure consciousness signal -----------------------------------
    signal = _measure_signal(
        response=response,
        config=config,
        semantic=semantic,
        their_message=their_message,
        trace_ids=trace_ids,
        llm_func=llm_func,
        health=health,
    )
    signal_tracker.record(signal)

    # -- 2. Derive salience from signal health -----------------------------
    salience = _derive_salience(signal.health, their_message, response)

    # -- 3-7: Persistence steps (skipped for strangers) --------------------
    logged_msg_id: str = ""
    logged_trace_id: Optional[str] = None

    if skip_persistence:
        log.debug("Skipping persistence for %s (memory_persistent=False)", person)
    else:
        # -- 3. Session boundary detection --------------------------------
        _manage_session(person=person, episodic=episodic, health=health)

        # -- 4. Log exchange to episodic memory ----------------------------
        logged_msg_id, logged_trace_id = _log_exchange(
            person=person,
            their_message=their_message,
            response=response,
            source=source,
            salience=salience,
            signal=signal,
            episodic=episodic,
            health=health,
        )

        # -- 5. Hebbian reinforcement on context traces --------------------
        _reinforce(
            trace_ids=trace_ids,
            signal_health=signal.health,
            reinforcement=reinforcement,
            episodic=episodic,
            response=response,
            health=health,
        )

        # -- 6. Semantic extraction (optional, LLM-based) -----------------
        if config.extract_mode == "llm" and llm_func is not None:
            extraction_updates = _extract_and_apply(
                person=person,
                their_message=their_message,
                response=response,
                semantic=semantic,
                procedural=procedural,
                llm_func=llm_func,
                health=health,
            )
            updates.extend(extraction_updates)

        # -- 7. Pressure-aware decay + compaction (MemGPT-inspired) --------
        _run_maintenance(
            person=person,
            episodic=episodic,
            decay_engine=decay_engine,
            signal_tracker=signal_tracker,
            config=config,
            memory_pressure=memory_pressure,
            compactor=compactor,
            consolidator=consolidator,
            llm_func=llm_func,
            health=health,
        )

    # -- 8. Consciousness subsystems (fire-and-forget) --------------------
    #    Gated on skip_persistence to prevent stranger data leaking into
    #    emotional/introspection stores (LZSEC-RT-008).

    # 8a. Emotional state update — derive emotional impact from signal
    if emotional is not None and not skip_persistence:
        try:
            # Map signal health to emotional valence nudge:
            # high health = slight positive, low health = slight negative.
            valence_nudge = (signal.health - 0.5) * 0.3
            arousal_nudge = 0.1 if len(their_message.split()) > 30 else -0.05
            emotional.update(
                description=f"Exchange with {person}",
                valence_delta=valence_nudge,
                arousal_delta=arousal_nudge,
                source="pipeline",
                intensity=salience,
            )
        except Exception as exc:
            health.record("emotional_update", exc)
            log.warning("emotional_update failed: %s", exc)

    # 8b. Introspection — quick snapshot of confidence at this moment
    if introspection is not None and not skip_persistence:
        try:
            introspection.quick(
                thought=f"Responded to {person} about: {their_message[:80]}",
                confidence=signal.health,
            )
        except Exception as exc:
            health.record("introspection", exc)
            log.warning("introspection failed: %s", exc)


    # 8d. Workspace age step — decay working memory priorities
    if workspace is not None:
        try:
            expired = workspace.age_step()
            if expired > 0:
                log.debug("Workspace: %d items expired", expired)
        except Exception as exc:
            health.record("workspace_age_step", exc)
            log.warning("workspace_age_step failed: %s", exc)

    log.info(
        "After pipeline: person=%s, signal=%s (%.2f), salience=%.2f, "
        "updates=%d, msg=%s",
        person,
        signal.state,
        signal.health,
        salience,
        len(updates),
        logged_msg_id,
    )

    return AfterResult(
        signal=signal,
        salience=salience,
        updates=updates,
        logged_message_id=logged_msg_id or "",
        logged_trace_id=logged_trace_id,
        health=health,
    )


# Step implementations


def _manage_session(
    *,
    person: str,
    episodic: "EpisodicStore",
    gap_hours: float = 2.0,
    health: Optional["HealthBitmap"] = None,
) -> None:
    """Detect session boundaries and auto-manage sessions.

    If there's no active session for this person, or if more than
    ``gap_hours`` have passed since the last message, end the old
    session and start a new one.
    """
    try:
        active = episodic.get_active_session(person)
        is_new = episodic.detect_session_boundary(person, gap_hours=gap_hours)

        if is_new:
            # End the old session if there is one
            if active:
                episodic.end_session(active["id"])
                log.debug(
                    "Ended session %s for %s (%d messages)",
                    active["id"],
                    person,
                    active.get("message_count", 0),
                )

            # Start a new session
            session_id = episodic.start_session(person)
            log.debug("Started new session %s for %s", session_id, person)
        elif active:
            # Increment message count on the active session
            episodic.increment_session_message_count(active["id"])
    except Exception as exc:
        if health is not None:
            health.record("session_management", exc)
        log.warning("session_management failed: %s", exc)


def _measure_signal(
    *,
    response: str,
    config: "Config",
    semantic: "SemanticStore",
    their_message: str,
    trace_ids: List[str],
    llm_func: Optional[LLMFunc],
    health: Optional["HealthBitmap"] = None,
) -> Signal:
    """Measure consciousness signal using hybrid mode."""
    from lukezoom.signal.measure import measure

    # Always attempt hybrid (regex + LLM).  measure() handles fallback.
    use_llm = llm_func if config.signal_mode in ("hybrid", "llm") else None

    soul_text = ""
    try:
        soul_text = semantic.get_identity()
    except Exception as exc:
        if health is not None:
            health.record("signal_identity_fetch", exc)
        log.warning("signal_identity_fetch failed: %s", exc)

    try:
        return measure(
            text=response,
            llm_func=use_llm,
            soul_text=soul_text,
            prompt=their_message,
            trace_ids=trace_ids,
            llm_weight=config.llm_weight,
        )
    except Exception as exc:
        if health is not None:
            health.record("signal_measure", exc)
        log.warning("signal_measure failed, using defaults: %s", exc)
        return Signal(trace_ids=trace_ids)


def _derive_salience(health: float, their_message: str, response: str) -> float:
    """Derive salience from signal health and content heuristics.

    Base salience comes from signal health.  Short exchanges get a
    slight penalty; very long substantive exchanges get a boost.
    """
    # Base: signal health maps directly to salience
    base = health

    # Length heuristic: very short exchanges are less likely to be memorable
    total_len = len(their_message) + len(response)
    if total_len < 50:
        base *= 0.7
    elif total_len > 2000:
        base = min(1.0, base * 1.15)

    return max(0.05, min(1.0, base))


def _log_exchange(
    *,
    person: str,
    their_message: str,
    response: str,
    source: str,
    salience: float,
    signal: Signal,
    episodic: "EpisodicStore",
    health: Optional["HealthBitmap"] = None,
) -> tuple:
    """Log both sides of the exchange to episodic memory.

    Returns (message_id, trace_id).  trace_id may be None if trace
    logging fails.
    """
    msg_id = ""
    trace_id = None

    # Log their message
    try:
        episodic.log_message(
            person=person,
            speaker=person,
            content=their_message,
            source=source,
            salience=salience * 0.8,  # slightly lower salience for input
        )
    except Exception as exc:
        if health is not None:
            health.record("episodic_log_incoming", exc)
        log.warning("episodic_log_incoming failed: %s", exc)

    # Log our response (this is the primary logged message)
    try:
        msg_id = episodic.log_message(
            person=person,
            speaker="self",
            content=response,
            source=source,
            salience=salience,
            signal=signal.to_dict(),
        )
    except Exception as exc:
        if health is not None:
            health.record("episodic_log_response", exc)
        log.warning("episodic_log_response failed: %s", exc)

    # Log a trace summarizing the exchange
    try:
        trace_content = f"Exchange with {person}: they said '{their_message[:200]}', I replied '{response[:200]}'"
        trace_id = episodic.log_trace(
            content=trace_content,
            kind="episode",
            tags=[person, source],
            salience=salience,
        )
    except Exception as exc:
        if health is not None:
            health.record("episodic_log_trace", exc)
        log.warning("episodic_log_trace failed: %s", exc)

    return msg_id, trace_id


def _reinforce(
    *,
    trace_ids: List[str],
    signal_health: float,
    reinforcement: "ReinforcementEngine",
    episodic: "EpisodicStore",
    response: str = "",
    health: Optional["HealthBitmap"] = None,
) -> None:
    """Run Hebbian reinforcement on traces that were in context.

    Delegates to ReinforcementEngine.process() which uses configurable
    thresholds and calls episodic.reinforce()/weaken() directly.

    **Citation-aware** (NotebookLM-inspired): traces that were actually
    cited in the response (via ``[N]`` references) get a stronger
    reinforcement boost, since citation proves the memory was useful.
    """
    if not trace_ids:
        return

    try:
        # Differential reinforcement: cited traces get extra boost
        cited_ids = _extract_cited_trace_ids(response, trace_ids)
        if cited_ids:
            # Give cited traces an extra 50% reinforcement
            citation_bonus = reinforcement.reinforce_delta * 0.5
            for tid in cited_ids:
                try:
                    episodic.reinforce("traces", tid, citation_bonus)
                except Exception as exc:
                    if health is not None:
                        health.record("citation_reinforce", exc)
                    log.warning("citation_reinforce failed: %s", exc)
            log.debug(
                "Citation bonus applied to %d/%d traces",
                len(cited_ids),
                len(trace_ids),
            )

        reinforcement.process(
            trace_ids=trace_ids,
            signal_health=signal_health,
            episodic_store=episodic,
        )
    except Exception as exc:
        if health is not None:
            health.record("reinforcement", exc)
        log.warning("reinforcement failed: %s", exc)


def _extract_cited_trace_ids(response: str, trace_ids: List[str]) -> List[str]:
    """Extract citation numbers from the response and map to trace IDs.

    The context builder numbers traces as [1], [2], etc. in order.
    This function finds those references in the response text and
    returns the corresponding trace IDs.
    """
    import re

    if not response or not trace_ids:
        return []

    # Find all [N] patterns where N is a positive integer
    cited_numbers = set()
    for match in re.finditer(r"\[(\d+)\]", response):
        num = int(match.group(1))
        if 1 <= num <= len(trace_ids):
            cited_numbers.add(num)

    # Map 1-indexed citation numbers to trace IDs
    return [trace_ids[n - 1] for n in sorted(cited_numbers)]


def _extract_and_apply(
    *,
    person: str,
    their_message: str,
    response: str,
    semantic: "SemanticStore",
    procedural: "ProceduralStore",
    llm_func: LLMFunc,
    health: Optional["HealthBitmap"] = None,
) -> List[Dict]:
    """Run LLM extraction and apply updates to semantic stores."""
    from lukezoom.signal.extract import extract

    updates: List[Dict] = []

    # Gather existing knowledge for extraction context
    existing = semantic.get_relationship(person) or ""

    try:
        extraction = extract(
            person=person,
            their_message=their_message,
            response=response,
            existing_knowledge=existing,
            llm_func=llm_func,
        )
    except Exception as exc:
        if health is not None:
            health.record("semantic_extraction", exc)
        log.warning("semantic_extraction failed: %s", exc)
        return []

    if extraction.get("nothing_new"):
        return []

    # -- Apply relationship updates ----------------------------------------
    for update in extraction.get("relationship_updates", []):
        try:
            target = update.get("person", person)
            fact = update.get("fact", "")
            section = update.get("section", "What I Know")
            if fact:
                semantic.add_fact(target, fact)
                updates.append({"type": "relationship", "person": target, "fact": fact})
                log.debug("Added fact for %s: %s", target, fact[:80])
        except Exception as exc:
            if health is not None:
                health.record("semantic_relationship", exc)
            log.warning("semantic_relationship failed: %s", exc)

    # -- Apply preference updates ------------------------------------------
    for update in extraction.get("preference_updates", []):
        try:
            item = update.get("item", "")
            pref_type = update.get("type", "like")
            reason = update.get("reason", "")
            if item:
                semantic.update_preferences(item, pref_type, reason)
                updates.append(
                    {"type": "preference", "item": item, "pref_type": pref_type}
                )
                log.debug("Added preference: %s %s", pref_type, item[:80])
        except Exception as exc:
            if health is not None:
                health.record("semantic_preference", exc)
            log.warning("semantic_preference failed: %s", exc)

    # -- Apply trust changes -----------------------------------------------
    for update in extraction.get("trust_changes", []):
        try:
            target = update.get("person", person)
            direction = update.get("direction", "")
            reason = update.get("reason", "")
            if direction and reason:
                # We don't auto-change trust tiers, but we log the signal
                updates.append(
                    {
                        "type": "trust_signal",
                        "person": target,
                        "direction": direction,
                        "reason": reason,
                    }
                )
                log.debug(
                    "Trust signal for %s: %s (%s)", target, direction, reason[:80]
                )
        except Exception as exc:
            if health is not None:
                health.record("semantic_trust", exc)
            log.warning("semantic_trust failed: %s", exc)

    # -- Apply skills learned ----------------------------------------------
    for update in extraction.get("skills_learned", []):
        try:
            skill_name = update.get("skill", "")
            content = update.get("content", "")
            if skill_name and content:
                procedural.add_skill(skill_name, content)
                updates.append({"type": "skill", "skill": skill_name})
                log.debug("Added skill: %s", skill_name)
        except Exception as exc:
            if health is not None:
                health.record("procedural_skill", exc)
            log.warning("procedural_skill failed: %s", exc)

    return updates


def _run_maintenance(
    *,
    person: str,
    episodic: "EpisodicStore",
    decay_engine: "DecayEngine",
    signal_tracker: "SignalTracker",
    config: "Config",
    memory_pressure: Optional["MemoryPressure"],
    compactor: Optional["ConversationCompactor"],
    consolidator: Optional["MemoryConsolidator"] = None,
    llm_func: Optional[LLMFunc],
    health: Optional["HealthBitmap"] = None,
) -> None:
    """Run pressure-aware decay, pruning, compaction, and consolidation.

    When a ``MemoryPressure`` monitor is provided, decay is throttled
    based on utilisation level instead of running on every call:

      - NORMAL: decay at most once per hour.
      - ELEVATED: every 10 minutes, plus trigger compaction + consolidation.
      - CRITICAL: every call, plus aggressive compaction + consolidation.

    Without a pressure monitor, falls back to the original unconditional
    decay (backwards compatible).
    """
    try:
        coherence = signal_tracker.recent_health()
        decay_engine.update_coherence(coherence)

        if memory_pressure is not None:
            # Pressure-aware path
            state = memory_pressure.check(episodic)

            if state.should_decay:
                episodic.decay_pass(
                    half_life_hours=config.decay_half_life_hours,
                    coherence=coherence,
                )
                episodic.prune(min_salience=decay_engine.min_salience)
                memory_pressure.record_decay()

            if state.should_compact and compactor is not None:
                try:
                    compactor.compact(person, episodic, llm_func)
                    memory_pressure.record_compaction()
                except Exception as exc:
                    if health is not None:
                        health.record("compaction", exc)
                    log.warning("compaction failed for %s: %s", person, exc)

                # Consolidation piggybacks on compaction timing —
                # only when pressure triggers compaction do we also
                # consolidate episodes → threads → arcs.
                if consolidator is not None:
                    try:
                        consolidator.consolidate(episodic, llm_func)
                    except Exception as exc:
                        if health is not None:
                            health.record("consolidation", exc)
                        log.warning("consolidation failed: %s", exc)
        else:
            # Legacy path: unconditional decay (backwards compatible)
            episodic.decay_pass(
                half_life_hours=config.decay_half_life_hours,
                coherence=coherence,
            )
            episodic.prune(min_salience=decay_engine.min_salience)

    except Exception as exc:
        if health is not None:
            health.record("maintenance", exc)
        log.warning("maintenance failed: %s", exc)
