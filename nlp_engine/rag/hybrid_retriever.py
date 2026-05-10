"""
Hybrid retriever using:

1. Elasticsearch BM25 retrieval
2. ChromaDB embedding retrieval
3. Hybrid score fusion

This replaces the old in-memory BM25 implementation completely.

Architecture
------------
Query
 ├── Elasticsearch (BM25 lexical retrieval)
 ├── ChromaDB (embedding similarity)
 └── Hybrid fusion

Benefits
--------
- No RAM-heavy BM25 index
- No _BM25_MAX_DOCS cap
- Better Arabic retrieval
- Scales to large corpora
- Faster lexical retrieval
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any

from nlp_engine.knowledge_base.es_store import bm25_search
from nlp_engine.rag.retriever import retrieve as embed_retrieve
from shared.config.settings import settings

logger = logging.getLogger(__name__)

_DEFAULT_BM25_K  = 10
_DEFAULT_EMBED_K = 10
_DEFAULT_ALPHA   = 0.6


# ---------------------------------------------------------------------------
# BM25 Retrieval (Elasticsearch)
# ---------------------------------------------------------------------------

def bm25_retrieve(
    query: str,
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents using Elasticsearch BM25.
    """
    if not query or not query.strip():
        return []

    k = top_k or _DEFAULT_BM25_K

    try:
        return bm25_search(query=query, top_k=k)

    except Exception as e:
        logger.exception("Elasticsearch BM25 retrieval failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Score Utilities
# ---------------------------------------------------------------------------

def _distance_to_similarity(distance: float | None) -> float:
    """
    Convert Chroma cosine distance into similarity score.

    Chroma returns:
        smaller distance = better

    We convert to:
        larger similarity = better
    """
    if distance is None:
        return 0.0

    sim = 1.0 - float(distance)

    return max(0.0, min(sim, 1.0))


def _normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize scores into [0, 1].
    """
    if not score_map:
        return {}

    max_score = max(score_map.values())

    if max_score <= 0:
        return {k: 0.0 for k in score_map}

    return {
        k: v / max_score
        for k, v in score_map.items()
    }


# ---------------------------------------------------------------------------
# Hybrid Retrieval
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query: str,
    n_results: int = 8,
    bm25_k: int | None = None,
    embed_k: int | None = None,
    alpha: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval using:
        hybrid_score = alpha * bm25 + (1 - alpha) * embedding

    Parameters
    ----------
    query:
        User query

    n_results:
        Final number of results returned

    bm25_k:
        Number of Elasticsearch BM25 candidates

    embed_k:
        Number of embedding candidates

    alpha:
        BM25 weight in hybrid fusion
        0.0 = embeddings only
        1.0 = BM25 only
    """

    if not query or not query.strip() or n_results <= 0:
        return []

    bm25_k  = bm25_k  or settings.BM25_TOP_K  or _DEFAULT_BM25_K
    embed_k = embed_k or settings.EMBED_TOP_K or _DEFAULT_EMBED_K
    alpha   = settings.BM25_WEIGHT if alpha is None else alpha

    # ------------------------------------------------------------------
    # Retrieve from Elasticsearch
    # ------------------------------------------------------------------

    bm25_results = bm25_retrieve(
        query=query,
        top_k=bm25_k,
    )

    # ------------------------------------------------------------------
    # Retrieve from Chroma embeddings
    # ------------------------------------------------------------------

    embed_results = embed_retrieve(
        query=query,
        n_results=embed_k,
    )

    # ------------------------------------------------------------------
    # Merge results
    # ------------------------------------------------------------------

    combined: Dict[str, Dict[str, Any]] = {}

    # Add BM25 results first
    for r in bm25_results:

        combined[r["id"]] = {
            "id":           r["id"],
            "document":     r["document"],
            "metadata":     r["metadata"],
            "bm25_score":   r.get("bm25_score", 0.0),
            "embed_score":  0.0,
            "distance":     None,
        }

    # Merge embedding results
    for r in embed_results:

        doc_id = r["id"]

        embed_score = _distance_to_similarity(
            r.get("distance")
        )

        if doc_id in combined:

            combined[doc_id].update({
                "document":    r.get(
                    "document",
                    combined[doc_id]["document"]
                ),
                "metadata":    r.get(
                    "metadata",
                    combined[doc_id]["metadata"]
                ),
                "distance":    r.get("distance"),
                "embed_score": embed_score,
            })

        else:

            combined[doc_id] = {
                "id":           doc_id,
                "document":     r["document"],
                "metadata":     r["metadata"],
                "distance":     r.get("distance"),
                "bm25_score":   0.0,
                "embed_score":  embed_score,
            }

    # ------------------------------------------------------------------
    # Normalize scores
    # ------------------------------------------------------------------

    bm25_scores = {
        doc_id: item.get("bm25_score", 0.0)
        for doc_id, item in combined.items()
    }

    embed_scores = {
        doc_id: item.get("embed_score", 0.0)
        for doc_id, item in combined.items()
    }

    bm25_norm  = _normalize_scores(bm25_scores)
    embed_norm = _normalize_scores(embed_scores)

    # ------------------------------------------------------------------
    # Hybrid score fusion
    # ------------------------------------------------------------------

    for doc_id, item in combined.items():

        item["hybrid_score"] = (
            alpha * bm25_norm.get(doc_id, 0.0)
            + (1 - alpha) * embed_norm.get(doc_id, 0.0)
        )

    # ------------------------------------------------------------------
    # Rank final results
    # ------------------------------------------------------------------

    ranked = sorted(
        combined.values(),
        key=lambda x: x.get("hybrid_score", 0.0),
        reverse=True,
    )

    return ranked[:n_results]