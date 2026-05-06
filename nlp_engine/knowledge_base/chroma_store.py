"""
ChromaDB client wrapper.

Manages a single persistent ChromaDB collection called "ppu_knowledge".
All CRUD operations (add, query, delete, list) go through this module.

Usage
-----
    from nlp_engine.knowledge_base.chroma_store import get_store

    store = get_store()
    store.add(ids=["doc1_chunk0"], embeddings=[[...]], documents=["text..."], metadatas=[{...}])
    results = store.query(query_embedding=[...], n_results=5)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from shared.config.settings import settings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "ppu_knowledge"


class ChromaStore:
    """Wrapper around a ChromaDB persistent collection."""

    def __init__(self, persist_dir: str | None = None):
        _dir = persist_dir or settings.CHROMA_PERSIST_DIR
        logger.info("Initialising ChromaDB at: %s", _dir)
        self._client = chromadb.PersistentClient(
            path=_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            "ChromaDB collection '%s' ready — %d documents indexed.",
            _COLLECTION_NAME,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Upsert chunks into the collection.

        Parameters
        ----------
        ids         : unique string ID per chunk
        embeddings  : pre-computed embedding vectors
        documents   : raw text of each chunk
        metadatas   : dict of metadata per chunk (doc_name, page, chunk_index, …)
        """
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("Upserted %d chunks into ChromaDB.", len(ids))

    def delete_by_doc(self, doc_name: str) -> None:
        """Remove all chunks belonging to a specific document."""
        self._collection.delete(where={"doc_name": doc_name})
        logger.info("Deleted all chunks for document: %s", doc_name)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-n most similar chunks.

        Returns
        -------
        List[dict] — each dict has: id, document, metadata, distance
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results = []
        for i, doc_id in enumerate(raw["ids"][0]):
            results.append(
                {
                    "id":       doc_id,
                    "document": raw["documents"][0][i],
                    "metadata": raw["metadatas"][0][i],
                    "distance": raw["distances"][0][i],
                }
            )
        return results

    def list_documents(self) -> List[str]:
        """Return a deduplicated list of indexed document names."""
        if self._collection.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("doc_name", "unknown") for m in all_meta})

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Return all chunks in the collection.

        Each item contains: id, document, metadata.
        """
        if self._collection.count() == 0:
            return []
        raw = self._collection.get(include=["documents", "metadatas"])
        ids = raw.get("ids", [])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])

        results: List[Dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            results.append(
                {
                    "id": doc_id,
                    "document": documents[i],
                    "metadata": metadatas[i],
                }
            )
        return results

    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: ChromaStore | None = None


def get_store(persist_dir: str | None = None) -> ChromaStore:
    """Return (or create) the process-level ChromaStore singleton."""
    global _store
    if _store is None:
        _store = ChromaStore(persist_dir)
    return _store
