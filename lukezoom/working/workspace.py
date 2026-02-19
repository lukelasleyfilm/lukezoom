"""
lukezoom.workspace — Cognitive workspace (Cowan's 4±1).

Updated from Miller's 7±2 based on Cowan (2001, Behavioral and Brain
Sciences) which showed true working memory capacity is ~4 chunks when
rehearsal and chunking are controlled. Morra et al. (2024) confirm 4
remains the dominant estimate for core central capacity.

Tracks what's "active in mind" with limited capacity, priority
decay, rehearsal boost, and eviction to episodic store.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("lukezoom.workspace")


# WorkspaceSlot


@dataclass
class WorkspaceSlot:
    """Single item in working memory."""

    item: str
    priority: float = 0.5
    source: str = "unknown"
    age: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def age_step(self, decay_rate: float = 0.95) -> None:
        self.age += 1
        self.priority *= decay_rate

    def rehearse(self, boost: float = 0.15) -> None:
        self.priority = min(1.0, self.priority + boost)
        self.age = 0
        self.last_accessed = time.time()
        self.access_count += 1

    def is_expired(self, threshold: float = 0.1) -> bool:
        return self.priority < threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item,
            "priority": round(self.priority, 4),
            "source": self.source,
            "age": self.age,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkspaceSlot":
        slot = cls(
            item=d["item"],
            priority=d.get("priority", 0.5),
            source=d.get("source", "unknown"),
        )
        slot.age = d.get("age", 0)
        slot.created_at = d.get("created_at", time.time())
        slot.last_accessed = d.get("last_accessed", time.time())
        slot.access_count = d.get("access_count", 0)
        return slot


# CognitiveWorkspace


class CognitiveWorkspace:
    """
    Human-like working memory with limited capacity.

    Default capacity is 4 items, based on Cowan's (2001) revised estimate
    of the focus of attention capacity (~4 chunks when rehearsal and
    chunking are controlled). This replaces Miller's classic 7±2 which
    conflated working memory with long-term memory support.

    Parameters
    ----------
    capacity : int
        Max items (default 4, per Cowan 2001).
    decay_rate : float
        Priority multiplier per ``age_step()`` call.
    rehearsal_boost : float
        Priority increase when an item is rehearsed.
    expiry_threshold : float
        Priority below which items are expired.
    storage_path : Path | None
        File for JSON persistence (None = in-memory only).
    on_evict : callable | None
        ``(slot_dict) -> None`` — callback when a slot is evicted.
    """

    def __init__(
        self,
        capacity: int = 4,
        decay_rate: float = 0.95,
        rehearsal_boost: float = 0.15,
        expiry_threshold: float = 0.1,
        storage_path: Optional[Path] = None,
        on_evict: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.rehearsal_boost = rehearsal_boost
        self.expiry_threshold = expiry_threshold
        self.storage_path = Path(storage_path) if storage_path else None
        self.on_evict = on_evict
        self.slots: List[WorkspaceSlot] = []
        self.focus_index: Optional[int] = None
        if self.storage_path:
            self._load()

    # -- public API ---------------------------------------------------------

    def add(
        self, item: str, priority: float = 0.5, source: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Add *item* to workspace.  If already present, rehearse instead.
        If full, evict the lowest-priority slot.
        """
        result: Dict[str, Any] = {"action": "added", "item": item[:80], "evicted": None}

        # De-dup: rehearse if already present
        for i, slot in enumerate(self.slots):
            if slot.item == item:
                slot.rehearse(self.rehearsal_boost)
                self.focus_index = i
                result["action"] = "rehearsed"
                self._save()
                return result

        new_slot = WorkspaceSlot(
            item=item, priority=max(0.0, min(1.0, priority)), source=source
        )

        if len(self.slots) < self.capacity:
            self.slots.append(new_slot)
            self.focus_index = len(self.slots) - 1
        else:
            lowest = min(range(len(self.slots)), key=lambda i: self.slots[i].priority)
            # Only evict if new item has higher priority than the weakest slot
            if new_slot.priority <= self.slots[lowest].priority:
                result["action"] = "rejected"
                self._save()
                return result
            evicted = self.slots[lowest]
            self._offload(evicted, "evicted")
            result["evicted"] = {
                "item": evicted.item[:80],
                "priority": round(evicted.priority, 4),
                "age": evicted.age,
            }
            self.slots[lowest] = new_slot
            self.focus_index = lowest

        self._save()
        return result

    def access(self, index: int) -> Optional[str]:
        """Access (rehearse) the slot at *index*."""
        if 0 <= index < len(self.slots):
            self.slots[index].rehearse(self.rehearsal_boost)
            self.focus_index = index
            self._save()
            return self.slots[index].item
        return None

    def find(self, substring: str) -> Optional[int]:
        """Find a slot containing *substring* (case-insensitive)."""
        sub = substring.lower()
        for i, slot in enumerate(self.slots):
            if sub in slot.item.lower():
                return i
        return None

    def age_step(self) -> int:
        """Decay all items and expire those below threshold.  Returns expired count."""
        for slot in self.slots:
            slot.age_step(self.decay_rate)
        # Single-pass partition to avoid TOCTOU between two list comprehensions
        surviving: List[WorkspaceSlot] = []
        expired: List[WorkspaceSlot] = []
        for s in self.slots:
            (expired if s.is_expired(self.expiry_threshold) else surviving).append(s)
        self.slots = surviving
        for s in expired:
            self._offload(s, "expired")
        if self.focus_index is not None and self.focus_index >= len(self.slots):
            self.focus_index = None
        self._save()
        return len(expired)

    def items(self, n: Optional[int] = None) -> List[str]:
        """Current contents sorted by priority (highest first)."""
        ordered = sorted(self.slots, key=lambda s: s.priority, reverse=True)
        if n is not None:
            ordered = ordered[:n]
        return [s.item for s in ordered]

    def detailed(self) -> List[Dict[str, Any]]:
        """All slots as dicts."""
        return [s.to_dict() for s in self.slots]

    def status(self) -> str:
        """Human-readable status."""
        n = len(self.slots)
        avg_p = sum(s.priority for s in self.slots) / n if n else 0
        lines = [
            f"Working memory: {n}/{self.capacity} slots (avg priority {avg_p:.2f})"
        ]
        for i, s in enumerate(self.slots):
            focus = " [FOCUS]" if i == self.focus_index else ""
            lines.append(
                f"  [{i}] {s.item[:50]}{'...' if len(s.item) > 50 else ''} "
                f"(p={s.priority:.2f}, age={s.age}){focus}"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        self.slots.clear()
        self.focus_index = None
        self._save()

    # -- internal -----------------------------------------------------------

    def _offload(self, slot: WorkspaceSlot, reason: str) -> None:
        if self.on_evict:
            try:
                self.on_evict(
                    {
                        "content": slot.item,
                        "priority_when_removed": slot.priority,
                        "age_when_removed": slot.age,
                        "time_in_workspace": time.time() - slot.created_at,
                        "access_count": slot.access_count,
                        "source": slot.source,
                        "removal_reason": reason,
                    }
                )
            except Exception:
                log.debug("Workspace eviction callback failed", exc_info=True)

    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "capacity": self.capacity,
            "slots": [s.to_dict() for s in self.slots],
            "focus_index": self.focus_index,
            "saved_at": time.time(),
        }
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            # Don't restore capacity from stale file — constructor/config wins
            # self.capacity = data.get("capacity", self.capacity)
            self.slots = [WorkspaceSlot.from_dict(s) for s in data.get("slots", [])]
            self.focus_index = data.get("focus_index")
        except Exception:
            log.warning(
                "Could not load workspace from %s", self.storage_path, exc_info=True
            )
