"""
RAG Retriever — semantic search against ChromaDB.

Usage
-----
    from nlp_engine.rag.retriever import retrieve

    chunks = retrieve("What are the CS tuition fees?", n_results=5)
    # [{"id": ..., "document": ..., "metadata": {...}, "distance": 0.12}, ...]
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional

from nlp_engine.knowledge_base.embed import get_embedder
from nlp_engine.knowledge_base.chroma_store import get_store

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    n_results: int = 5,
    doc_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Embed *query* and return the top-n most similar chunks from ChromaDB.

    Parameters
    ----------
    query      : user question (pre-normalized text)
    n_results  : number of chunks to return
    doc_filter : if set, restrict search to a specific document name

    Returns
    -------
    List[dict] — each item: {id, document, metadata, distance}
                 Sorted by ascending distance (most similar first).
    """
    if not query or not query.strip():
        return []

    embedder = get_embedder()
    store    = get_store()

    if store.count() == 0:
        logger.warning("ChromaDB is empty — no documents have been ingested yet.")
        return []

    query_vector = embedder.embed_one(query)

    where = {"doc_name": doc_filter} if doc_filter else None

    results = store.query(
        query_embedding=query_vector,
        n_results=n_results,
        where=where,
    )

    logger.debug("Retrieved %d chunks for query: '%s'", len(results), query[:60])
    return results
