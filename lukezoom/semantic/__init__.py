"""lukezoom.semantic — File-backed semantic memory (YAML + Markdown)."""

from lukezoom.semantic.store import SemanticStore
from lukezoom.semantic.identity import IdentityResolver

__all__ = ["SemanticStore", "IdentityResolver"]
