"""lukezoom.search — FTS5 indexed search, optional ChromaDB semantic search."""

from lukezoom.search.indexed import IndexedSearch

# UnifiedSearch and SemanticSearch require chromadb (optional)
try:
    from lukezoom.search.unified import UnifiedSearch
    __all__ = ["IndexedSearch", "UnifiedSearch"]
except ImportError:
    __all__ = ["IndexedSearch"]
