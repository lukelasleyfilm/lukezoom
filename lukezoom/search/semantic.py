"""
lukezoom.search.semantic — Semantic (vector) search using ChromaDB.

Provides RAG-style retrieval across all four memory layers:
soul (identity), episodic (traces), semantic (relationships/facts),
and procedural (skills).

If an ``embedding_func`` is provided (e.g. an Ollama embeddings
wrapper), it is used for all collections.  Otherwise ChromaDB's
built-in default embedding function handles it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    HAS_CHROMADB = False


# The four collection names, one per memory layer.
COLLECTIONS = ("soul", "episodic", "semantic", "procedural")


class SemanticSearch:
    """ChromaDB-backed vector search across all memory layers."""

    def __init__(
        self,
        embeddings_dir: Path,
        embedding_func: Optional[Callable] = None,
    ) -> None:
        if not HAS_CHROMADB:
            raise ImportError(
                "chromadb is required for SemanticSearch. "
                "Install with: pip install lukezoom[vector]"
            )
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.embeddings_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Wrap a user-supplied embedding callable into the ChromaDB
        # EmbeddingFunction protocol if provided.
        self._ef: Optional[chromadb.EmbeddingFunction] = None
        if embedding_func is not None:
            self._ef = _WrappedEmbeddingFunction(embedding_func)

        # Pre-create / get all collections
        self._collections: Dict[str, chromadb.Collection] = {}
        for name in COLLECTIONS:
            kwargs: Dict = {"name": name}
            if self._ef is not None:
                kwargs["embedding_function"] = self._ef
            self._collections[name] = self._client.get_or_create_collection(**kwargs)

    # Indexing

    def index_text(
        self,
        text: str,
        metadata: Dict,
        collection: str,
        doc_id: str,
    ) -> None:
        """Add or update a single document in a collection."""
        col = self._get_collection(collection)
        col.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[self._sanitise_metadata(metadata)],
        )

    def index_soul(self, soul_text: str) -> None:
        """Chunk SOUL.md into paragraphs and index each."""
        paragraphs = self._split_paragraphs(soul_text)
        col = self._get_collection("soul")

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for i, para in enumerate(paragraphs):
            doc_id = f"soul_p{i}_{self._hash(para)}"
            ids.append(doc_id)
            documents.append(para)
            metadatas.append({"type": "soul", "paragraph": i})

        if ids:
            col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def index_relationship(self, person: str, content: str) -> None:
        """Index a relationship file."""
        doc_id = f"rel_{self._hash(person)}"
        self.index_text(
            text=content,
            metadata={"type": "relationship", "person": person},
            collection="semantic",
            doc_id=doc_id,
        )

    def index_trace(self, trace_id: str, content: str, metadata: Dict) -> None:
        """Index an episodic trace."""
        safe_meta = self._sanitise_metadata(metadata)
        safe_meta["type"] = "trace"
        safe_meta["trace_id"] = trace_id
        self.index_text(
            text=content,
            metadata=safe_meta,
            collection="episodic",
            doc_id=f"trace_{trace_id}",
        )

    def index_skill(self, skill_name: str, content: str) -> None:
        """Index a procedural skill."""
        doc_id = f"skill_{self._hash(skill_name)}"
        self.index_text(
            text=content,
            metadata={"type": "skill", "skill_name": skill_name},
            collection="procedural",
            doc_id=doc_id,
        )

    # Search

    def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search across one or more collections.

        Parameters
        ----------
        query:
            Natural language search query.
        collections:
            Which collections to search (default: all).
        n_results:
            Max results per collection.
        where:
            ChromaDB metadata filter dict.

        Returns
        -------
        list[dict]
            Each result has ``collection``, ``doc_id``, ``content``,
            ``metadata``, and ``distance``.  Sorted by distance
            (ascending — lower is closer).
        """
        target_collections = collections or list(COLLECTIONS)
        all_results: List[Dict] = []

        for col_name in target_collections:
            if col_name not in self._collections:
                continue

            col = self._collections[col_name]

            # ChromaDB requires at least 1 document in the collection
            if col.count() == 0:
                continue

            query_kwargs: Dict = {
                "query_texts": [query],
                "n_results": min(n_results, col.count()),
            }
            if where:
                query_kwargs["where"] = where

            try:
                results = col.query(**query_kwargs)
            except Exception:
                # Collection might be empty or query malformed
                continue

            # Unpack ChromaDB's nested list format
            if not results or not results.get("ids"):
                continue

            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, doc_id in enumerate(ids):
                all_results.append(
                    {
                        "collection": col_name,
                        "doc_id": doc_id,
                        "content": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "distance": distances[i] if i < len(distances) else 1.0,
                    }
                )

        # Sort by distance (lower = more similar)
        all_results.sort(key=lambda r: r["distance"])
        return all_results

    # Lifecycle

    def close(self) -> None:
        """Release the ChromaDB client resources."""
        # PersistentClient doesn't have an explicit close, but we
        # clear references so the GC can collect promptly.
        self._collections.clear()
        self._client = None  # type: ignore[assignment]

    # Removal

    def remove(self, doc_id: str, collection: str) -> None:
        """Remove a document from a collection."""
        col = self._get_collection(collection)
        try:
            col.delete(ids=[doc_id])
        except Exception:
            pass  # Silently ignore if doc doesn't exist

    # Full reindex

    def reindex_all(
        self,
        episodic_store: object,
        semantic_store: object,
        procedural_store: object,
    ) -> Dict[str, int]:
        """Full reindex of all memory layers.

        Parameters
        ----------
        episodic_store:
            ``EpisodicStore`` instance — uses ``get_traces()``
        semantic_store:
            Object with ``soul_path`` and ``identities_path`` attrs,
            or a dict with ``{"soul_text": str, "relationships": dict}``
        procedural_store:
            ``ProceduralStore`` instance — uses ``list_skills()``
            and ``get_skill()``

        Returns
        -------
        dict
            Counts of indexed documents per collection.
        """
        counts: Dict[str, int] = {c: 0 for c in COLLECTIONS}

        # -- Soul / identity -----------------------------------------------
        soul_text = ""
        if hasattr(semantic_store, "soul_dir"):
            soul_path = Path(getattr(semantic_store, "soul_dir")) / "SOUL.md"
            if soul_path.exists():
                soul_text = soul_path.read_text(encoding="utf-8")
        elif hasattr(semantic_store, "get_identity"):
            soul_text = semantic_store.get_identity()
        elif isinstance(semantic_store, dict):
            soul_text = semantic_store.get("soul_text", "")

        if soul_text:
            self.index_soul(soul_text)
            counts["soul"] = len(self._split_paragraphs(soul_text))

        # -- Relationships -------------------------------------------------
        relationships: Dict[str, str] = {}
        if isinstance(semantic_store, dict):
            relationships = semantic_store.get("relationships", {})
        elif hasattr(semantic_store, "list_relationships") and hasattr(
            semantic_store, "get_relationship"
        ):
            # Use SemanticStore API directly
            for rel in semantic_store.list_relationships():
                person = rel.get("name", "")
                if person:
                    content = semantic_store.get_relationship(person)
                    if content:
                        relationships[person] = content

        for person, content in relationships.items():
            self.index_relationship(person, content)
            counts["semantic"] += 1

        # -- Episodic traces -----------------------------------------------
        if hasattr(episodic_store, "get_traces"):
            traces = episodic_store.get_traces(limit=10000)
            for trace in traces:
                trace_id = trace.get("id", "")
                content = trace.get("content", "")
                if trace_id and content:
                    meta = {
                        k: v
                        for k, v in trace.items()
                        if k not in ("content",)
                        and isinstance(v, (str, int, float, bool))
                    }
                    self.index_trace(trace_id, content, meta)
                    counts["episodic"] += 1

        # -- Procedural skills ---------------------------------------------
        if hasattr(procedural_store, "list_skills") and hasattr(
            procedural_store, "get_skill"
        ):
            for skill_info in procedural_store.list_skills():
                name = skill_info.get("name", "")
                content = procedural_store.get_skill(name)
                if name and content:
                    self.index_skill(name, content)
                    counts["procedural"] += 1

        return counts

    # Helpers

    def _get_collection(self, name: str) -> chromadb.Collection:
        if name not in self._collections:
            raise ValueError(
                f"Unknown collection {name!r}; expected one of {list(COLLECTIONS)}"
            )
        return self._collections[name]

    @staticmethod
    def _hash(text: str) -> str:
        """Short hash for stable IDs."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split text into non-empty paragraphs."""
        raw = re.split(r"\n\s*\n", text)
        return [p.strip() for p in raw if p.strip()]

    @staticmethod
    def _sanitise_metadata(metadata: Dict) -> Dict:
        """Ensure all metadata values are ChromaDB-compatible scalars."""
        clean: Dict = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                # Convert lists to comma-separated strings
                clean[k] = ", ".join(str(item) for item in v)
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = str(v)
        return clean


# Embedding function adapter


if HAS_CHROMADB:
    class _WrappedEmbeddingFunction(chromadb.EmbeddingFunction):
        """Adapts a plain ``Callable[[str], list[float]]`` (e.g. Ollama
        embeddings) into ChromaDB's ``EmbeddingFunction`` protocol."""

        def __init__(self, func: Callable) -> None:
            self._func = func

        def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
            return [self._func(text) for text in input]
