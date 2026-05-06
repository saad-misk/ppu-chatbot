"""
RAG Reranker — re-ranks retrieved chunks by cosine similarity between the
query embedding and each chunk embedding, then applies a relevance threshold.

ChromaDB already returns results sorted by distance, but reranking allows us to:
  • Apply a hard cut-off (drop chunks with score < threshold)
  • Optionally incorporate future cross-encoder signals without changing the
    retriever interface.

Usage
-----
    from nlp_engine.rag.retriever import retrieve
    from nlp_engine.rag.reranker import rerank

    chunks  = retrieve(query, n_results=10)
    top3    = rerank(query, chunks, top_k=3)
"""
from __future__ import annotations

import logging
from typing import List, Dict

import numpy as np

from nlp_engine.knowledge_base.embed import get_embedder

logger = logging.getLogger(__name__)

# Chunks with cosine similarity below this are dropped entirely
_DEFAULT_THRESHOLD = 0.25


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def rerank(
    query: str,
    chunks: List[Dict],
    top_k: int = 3,
    threshold: float = _DEFAULT_THRESHOLD,
) -> List[Dict]:
    """
    Re-rank *chunks* by cosine similarity to *query* and return the top-k.

    Each chunk in the output gets an additional ``score`` field (0–1).
    Chunks below *threshold* are discarded.

    Parameters
    ----------
    query     : original user question
    chunks    : raw results from retrieve() — must have a "document" key
    top_k     : maximum number of chunks to return
    threshold : minimum cosine similarity score to keep a chunk

    Returns
    -------
    List[dict] — re-ranked chunks with ``score`` field, best first.
    """
    if not chunks:
        return []

    embedder = get_embedder()
    query_vec = embedder.embed_one(query)

    # Embed all chunk texts in one batch
    texts = [c["document"] for c in chunks]
    chunk_vecs = embedder.embed(texts)

    scored = []
    for chunk, vec in zip(chunks, chunk_vecs):
        score = _cosine_similarity(query_vec, vec)
        if score >= threshold:
            scored.append({**chunk, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    result = scored[:top_k]

    logger.debug(
        "Reranker: %d → %d chunks (threshold=%.2f, top_k=%d)",
        len(chunks), len(result), threshold, top_k,
    )
    return result
