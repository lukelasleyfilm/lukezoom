"""
lukezoom.consolidation.compactor — Conversation compaction engine.

Inspired by MemGPT's conversation summarisation: when message history
for a person grows beyond a threshold, older messages are summarised
into thread-level traces and the originals are marked as archived.

Two modes:
  - **LLM-based** (preferred): summarises via the configured LLM func.
  - **Extractive fallback**: when no LLM is available, picks the
    highest-salience messages and concatenates them into a digest.

Original messages are NEVER deleted — they get an ``archived=1`` flag
so they're excluded from ``get_recent_messages()`` but remain
searchable via FTS.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from lukezoom.core.types import LLMFunc
    from lukezoom.episodic.store import EpisodicStore

log = logging.getLogger(__name__)

# System prompt for the summariser LLM
_SUMMARISER_SYSTEM = (
    "You are a memory-consolidation system. Your job is to summarise "
    "a segment of conversation history from the perspective of the AI "
    "(first person). Preserve: key facts learned, emotional moments, "
    "promises made, decisions reached, and any changes in the "
    "relationship. Be concise but complete. Keep under 200 words."
)


@dataclass
class CompactionResult:
    """Result of a compaction run.

    Attributes
    ----------
    person : str
        Who was compacted.
    messages_archived : int
        How many messages were archived.
    summaries_created : int
        How many summary traces were created.
    segments_processed : int
        How many conversation segments were processed.
    """

    person: str
    messages_archived: int
    summaries_created: int
    segments_processed: int


class ConversationCompactor:
    """Compact old conversations into summary traces.

    Parameters
    ----------
    keep_recent : int
        Number of most-recent messages per person to leave untouched
        (default 20).
    segment_size : int
        Maximum messages per summary segment (default 30).
        Larger segments produce denser summaries but require more
        LLM context.
    min_messages_to_compact : int
        Don't bother compacting unless the person has at least this
        many unarchived messages (default 40).
    """

    def __init__(
        self,
        keep_recent: int = 20,
        segment_size: int = 30,
        min_messages_to_compact: int = 40,
    ) -> None:
        self.keep_recent = keep_recent
        self.segment_size = segment_size
        self.min_messages_to_compact = min_messages_to_compact

    def compact(
        self,
        person: str,
        episodic: EpisodicStore,
        llm_func: Optional[LLMFunc] = None,
    ) -> CompactionResult:
        """Run compaction for a single person.

        Steps:
          1. Fetch all unarchived messages for ``person``.
          2. If count <= ``min_messages_to_compact``, skip.
          3. Keep the ``keep_recent`` most-recent messages untouched.
          4. Segment the older messages into chunks of ``segment_size``.
          5. Summarise each segment (LLM or extractive fallback).
          6. Store each summary as a trace (kind="summary").
          7. Mark original messages as archived.

        Returns
        -------
        CompactionResult
        """
        # Step 1: fetch all messages (unarchived only)
        all_messages = _get_unarchived_messages(episodic, person)

        # Step 2: threshold check
        if len(all_messages) < self.min_messages_to_compact:
            return CompactionResult(
                person=person,
                messages_archived=0,
                summaries_created=0,
                segments_processed=0,
            )

        # Step 3: split into keep vs. compact
        # Messages are in chronological order (oldest first)
        compact_messages = all_messages[: -self.keep_recent]
        if not compact_messages:
            return CompactionResult(
                person=person,
                messages_archived=0,
                summaries_created=0,
                segments_processed=0,
            )

        # Step 4: segment
        segments = _segment_messages(compact_messages, self.segment_size)

        # Step 5 + 6 + 7: summarise, store, archive
        summaries_created = 0
        messages_archived = 0

        for segment in segments:
            # Build conversation text for summarisation
            conv_text = _format_segment(segment)

            # Summarise
            if llm_func is not None:
                summary = _llm_summarise(conv_text, llm_func)
            else:
                summary = _extractive_summarise(segment)

            if not summary:
                log.warning(
                    "Empty summary for %s segment (%d msgs), skipping",
                    person,
                    len(segment),
                )
                continue

            # Derive time range for metadata
            first_ts = segment[0].get("timestamp", "")
            last_ts = segment[-1].get("timestamp", "")

            # Store as a high-salience summary trace
            try:
                episodic.log_trace(
                    content=summary,
                    kind="summary",
                    tags=[person, "compaction"],
                    salience=0.7,  # summaries start with high salience
                    time_range=f"{first_ts} to {last_ts}",
                    message_count=len(segment),
                )
                summaries_created += 1
            except Exception as exc:
                log.warning("Failed to store summary trace: %s", exc)
                continue

            # Archive the original messages
            msg_ids = [m["id"] for m in segment if m.get("id")]
            archived = _archive_messages(episodic, msg_ids)
            messages_archived += archived

        result = CompactionResult(
            person=person,
            messages_archived=messages_archived,
            summaries_created=summaries_created,
            segments_processed=len(segments),
        )

        log.info(
            "Compaction complete for %s: %d messages archived, "
            "%d summaries created from %d segments",
            person,
            result.messages_archived,
            result.summaries_created,
            result.segments_processed,
        )
        return result

    def compact_all(
        self,
        episodic: EpisodicStore,
        llm_func: Optional[LLMFunc] = None,
    ) -> List[CompactionResult]:
        """Run compaction for all people who have enough messages.

        Returns a list of CompactionResult for each person that was
        actually compacted (skips people below threshold).
        """
        people = _get_all_people(episodic)
        results = []
        for person in people:
            result = self.compact(person, episodic, llm_func)
            if result.messages_archived > 0:
                results.append(result)
        return results


# Private helpers


def _get_unarchived_messages(episodic: EpisodicStore, person: str) -> List[Dict]:
    """Get all unarchived messages for a person in chronological order."""
    rows = episodic.conn.execute(
        """
        SELECT * FROM messages
        WHERE person = ? AND COALESCE(json_extract(metadata, '$.archived'), 0) = 0
        ORDER BY timestamp ASC
        """,
        (person,),
    ).fetchall()
    return [episodic._row_to_dict(r) for r in rows]


def _get_all_people(episodic: EpisodicStore) -> List[str]:
    """Get a list of all distinct person names in messages."""
    rows = episodic.conn.execute(
        "SELECT DISTINCT person FROM messages WHERE person IS NOT NULL"
    ).fetchall()
    return [r[0] for r in rows]


def _segment_messages(messages: List[Dict], segment_size: int) -> List[List[Dict]]:
    """Split messages into segments of at most ``segment_size``.

    Also breaks on large time gaps (> 2 hours) to keep sessions
    together even if they're smaller than segment_size.
    """
    if not messages:
        return []

    segments: List[List[Dict]] = []
    current: List[Dict] = []

    for msg in messages:
        # Check for time gap
        if current and _time_gap_hours(current[-1], msg) > 2.0:
            if current:
                segments.append(current)
            current = [msg]
            continue

        current.append(msg)

        if len(current) >= segment_size:
            segments.append(current)
            current = []

    if current:
        segments.append(current)

    return segments


def _time_gap_hours(msg_a: Dict, msg_b: Dict) -> float:
    """Compute the time gap in hours between two messages."""
    try:
        ts_a = msg_a.get("timestamp", "")
        ts_b = msg_b.get("timestamp", "")
        dt_a = datetime.fromisoformat(ts_a.replace("Z", "+00:00"))
        dt_b = datetime.fromisoformat(ts_b.replace("Z", "+00:00"))
        return abs((dt_b - dt_a).total_seconds()) / 3600.0
    except (ValueError, TypeError, AttributeError):
        return 0.0


def _format_segment(segment: List[Dict]) -> str:
    """Format a message segment into readable conversation text for the LLM."""
    lines = []
    for msg in segment:
        speaker = msg.get("speaker", "?")
        content = msg.get("content", "")
        lines.append(f"[{speaker}]: {content}")
    return "\n".join(lines)


def _llm_summarise(conv_text: str, llm_func: "LLMFunc") -> str:
    """Summarise a conversation segment using the LLM."""
    # Truncate if absurdly long (protect the LLM's context)
    if len(conv_text) > 12_000:
        conv_text = conv_text[:12_000] + "\n\n[...truncated]"

    prompt = (
        "Summarise the following conversation segment:\n\n"
        f"{conv_text}\n\n"
        "Write a concise first-person summary (under 200 words) "
        "capturing key facts, emotions, and decisions."
    )

    try:
        return llm_func(prompt, _SUMMARISER_SYSTEM).strip()
    except Exception as exc:
        log.warning("LLM summarisation failed, using extractive fallback: %s", exc)
        return ""


def _extractive_summarise(segment: List[Dict]) -> str:
    """Fallback: pick the highest-salience messages and concatenate."""
    # Sort by salience descending, take top 5
    ranked = sorted(segment, key=lambda m: m.get("salience", 0.0), reverse=True)
    top = ranked[: min(5, len(ranked))]
    # Re-sort chronologically
    top.sort(key=lambda m: m.get("timestamp", ""))

    lines = []
    for msg in top:
        speaker = msg.get("speaker", "?")
        content = msg.get("content", "")
        # Truncate individual messages
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"[{speaker}]: {content}")

    if not lines:
        return ""

    person = segment[0].get("person", "someone") if segment else "someone"
    first_ts = segment[0].get("timestamp", "")
    last_ts = segment[-1].get("timestamp", "")

    header = f"Summary of conversation with {person} ({first_ts} to {last_ts}):"
    return header + "\n" + "\n".join(lines)


def _archive_messages(episodic: EpisodicStore, msg_ids: List[str]) -> int:
    """Mark messages as archived by setting metadata.archived = 1.

    Returns the number of messages successfully archived.
    """
    if not msg_ids:
        return 0

    archived = 0
    for msg_id in msg_ids:
        try:
            # Read existing metadata
            row = episodic.conn.execute(
                "SELECT metadata FROM messages WHERE id = ?", (msg_id,)
            ).fetchone()
            if row is None:
                continue

            metadata = {}
            if row[0]:
                try:
                    metadata = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            metadata["archived"] = 1
            metadata["archived_at"] = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

            episodic.conn.execute(
                "UPDATE messages SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), msg_id),
            )
            archived += 1
        except Exception as exc:
            log.debug("Failed to archive message %s: %s", msg_id, exc)

    if archived:
        episodic.conn.commit()
    return archived
