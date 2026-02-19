"""
lukezoom.search.unified — Unified search combining FTS5 + semantic.

Merges keyword (SQLite FTS5) and vector (ChromaDB) search results,
deduplicates, and returns a combined ranking.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from lukezoom.search.indexed import IndexedSearch

try:
    from lukezoom.search.semantic import SemanticSearch
except ImportError:
    SemanticSearch = None  # type: ignore


class UnifiedSearch:
    """Combined keyword + semantic search across all memory types.

    If only ``indexed`` is provided, semantic search is skipped.
    This lets the system work without embeddings while still
    benefiting from them when available.
    """

    def __init__(
        self,
        indexed: IndexedSearch,
        semantic: Optional[SemanticSearch] = None,
    ) -> None:
        self.indexed = indexed
        self.semantic = semantic

    def search(
        self,
        query: str,
        person: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Run FTS + semantic search, merge, deduplicate, and rank.

        Parameters
        ----------
        query:
            Search query (natural language or keywords).
        person:
            Filter to this person (where applicable).
        memory_type:
            ``"messages"``, ``"traces"``, or ``None`` for both.
        limit:
            Maximum total results to return.

        Returns
        -------
        list[dict]
            Merged results, each with a ``"combined_score"`` key.
            Lower score = better match.
        """
        if not query or not query.strip():
            return []

        # -- FTS5 search ---------------------------------------------------
        fts_results = self.indexed.search(
            query=query,
            memory_type=memory_type,
            person=person,
            limit=limit,
        )

        # -- Semantic search -----------------------------------------------
        sem_results: List[Dict] = []
        if self.semantic is not None:
            # Map memory_type to ChromaDB collections
            collections = self._map_collections(memory_type)
            where = None
            if person:
                where = {"person": person}

            try:
                sem_results = self.semantic.search(
                    query=query,
                    collections=collections,
                    n_results=limit,
                    where=where,
                )
            except Exception:
                # Semantic search is best-effort; don't fail the whole query
                sem_results = []

        # -- Merge and deduplicate -----------------------------------------
        merged = self._merge(fts_results, sem_results)

        # -- Sort by combined score ----------------------------------------
        merged.sort(key=lambda r: r.get("combined_score", 999.0))

        return merged[:limit]

    # Internal

    @staticmethod
    def _map_collections(memory_type: Optional[str]) -> Optional[List[str]]:
        """Map a ``memory_type`` filter to ChromaDB collection names."""
        if memory_type == "messages":
            # Messages aren't in ChromaDB by default; search episodic
            # which contains traces from conversations
            return ["episodic"]
        if memory_type == "traces":
            return ["episodic"]
        # None → search all
        return None

    @staticmethod
    def _merge(
        fts_results: List[Dict],
        sem_results: List[Dict],
    ) -> List[Dict]:
        """Merge FTS and semantic results, deduplicate, compute combined score.

        Deduplication key priority:
        1. ``id`` field (messages/traces have a 12-char hex ID)
        2. ``doc_id`` field (semantic results)
        3. Content hash as fallback

        Scoring:
        - FTS rank is negative (more negative = better).  We normalise
          to 0..1 range within the result set.
        - Semantic distance is 0..2 (cosine).  We normalise to 0..1.
        - Combined score = weighted average: 0.5 * fts_norm + 0.5 * sem_norm.
        - Items appearing in only one source get 0.5 for the missing score.
        """
        seen: Dict[str, Dict] = {}

        # -- Process FTS results -------------------------------------------
        # Normalise FTS ranks to 0..1 (lower = better)
        fts_ranks = [abs(r.get("rank", 0)) for r in fts_results]
        fts_max = max(fts_ranks) if fts_ranks else 1.0
        fts_max = fts_max if fts_max > 0 else 1.0

        for i, result in enumerate(fts_results):
            key = _dedup_key(result)
            normalised_rank = abs(result.get("rank", 0)) / fts_max
            # Invert so that best match → lowest score
            fts_score = 1.0 - normalised_rank if fts_max > 0 else 0.5

            if key in seen:
                seen[key]["fts_score"] = fts_score
            else:
                entry = dict(result)
                entry["fts_score"] = fts_score
                entry["sem_score"] = 0.5  # default if not in semantic
                seen[key] = entry

        # -- Process semantic results --------------------------------------
        sem_distances = [r.get("distance", 1.0) for r in sem_results]
        sem_max = max(sem_distances) if sem_distances else 1.0
        sem_max = sem_max if sem_max > 0 else 1.0

        for result in sem_results:
            key = _dedup_key(result)
            sem_score = result.get("distance", 1.0) / sem_max

            if key in seen:
                seen[key]["sem_score"] = sem_score
                # Enrich with semantic metadata if missing
                if "collection" not in seen[key]:
                    seen[key]["collection"] = result.get("collection", "")
            else:
                entry = dict(result)
                entry["sem_score"] = sem_score
                entry["fts_score"] = 0.5  # default if not in FTS
                # Normalise field names to match FTS format
                if "content" not in entry and "doc_id" in entry:
                    entry["content"] = result.get("content", "")
                seen[key] = entry

        # -- Compute combined scores ---------------------------------------
        for entry in seen.values():
            fts_s = entry.get("fts_score", 0.5)
            sem_s = entry.get("sem_score", 0.5)
            entry["combined_score"] = 0.5 * fts_s + 0.5 * sem_s

        return list(seen.values())


def _dedup_key(result: Dict) -> str:
    """Compute a deduplication key for a search result."""
    # Prefer explicit ID fields
    if result.get("id"):
        return f"id:{result['id']}"
    if result.get("doc_id"):
        return f"doc:{result['doc_id']}"
    if result.get("trace_id"):
        return f"trace:{result['trace_id']}"
    # Fall back to content hash
    content = result.get("content", "")
    return f"hash:{hash(content)}"
