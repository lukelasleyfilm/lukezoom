"""
lukezoom.consolidation — Memory pressure management, compaction, and
hierarchical consolidation.

Inspired by:
  - MemGPT's virtual-context paging and memory pressure warnings
  - NotebookLM's auto-generated Source Guides
  - Neuroscience: hippocampal replay during sleep
"""

from lukezoom.consolidation.pressure import MemoryPressure, PressureState
from lukezoom.consolidation.compactor import ConversationCompactor, CompactionResult
from lukezoom.consolidation.consolidator import MemoryConsolidator

__all__ = [
    "MemoryPressure",
    "PressureState",
    "ConversationCompactor",
    "CompactionResult",
    "MemoryConsolidator",
]
