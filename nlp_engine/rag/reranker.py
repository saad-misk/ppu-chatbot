"""
RAG Reranker — re-ranks retrieved chunks using existing similarity scores.

ChromaDB already computed cosine distance during retrieval. This module
re-ranks using those existing scores — NO redundant embedding needed.

Usage
-----
    from nlp_engine.rag.reranker import rerank
    
    chunks = [{"id": ..., "document": ..., "distance": 0.12}, ...]
    top3 = rerank(chunks, top_k=3, threshold=0.25)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.25


def rerank(
    chunks: List[Dict],
    top_k: int = 3,
    threshold: float = _DEFAULT_THRESHOLD,
) -> List[Dict]:
    """
    Re-rank chunks using existing distance/similarity scores.
    
    No redundant embedding — uses scores already computed during retrieval.
    
    Args:
        chunks: Retrieved chunks with 'distance' or 'score'
        top_k: Maximum chunks to return
        threshold: Minimum similarity score (0-1)
        
    Returns:
        Reranked chunks with 'score' field added
    """
    if not chunks:
        return []
    
    scored = []
    
    for chunk in chunks:
        # Get best available score
        score = _extract_score(chunk)
        
        if score >= threshold:
            # Add/update score field for downstream use
            scored.append({**chunk, "score": score})
    
    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    result = scored[:top_k]
    
    logger.debug(
        "Reranker: %d → %d chunks (threshold=%.2f, top_k=%d)",
        len(chunks), len(result), threshold, top_k,
    )
    
    return result


def _extract_score(chunk: Dict) -> float:
    """
    Extract similarity score from chunk using available fields.
    
    Priority:
    1. hybrid_score (from hybrid retriever)
    2. score (pre-computed)
    3. bm25_score (needs normalization - treated as 0.5)
    4. distance (convert to similarity: 1 - distance)
    """
    # Hybrid score already combines both signals
    if "hybrid_score" in chunk and chunk["hybrid_score"] is not None:
        return float(chunk["hybrid_score"])
    
    # Pre-computed similarity score
    if "score" in chunk and chunk["score"] is not None:
        return float(chunk["score"])
    
    # Cosine distance from ChromaDB
    if "distance" in chunk and chunk["distance"] is not None:
        return 1.0 - float(chunk["distance"])
    
    # BM25 score only (not normalized to 0-1)
    # Assign moderate score — these are keyword matches
    if "bm25_score" in chunk and chunk["bm25_score"] > 0:
        return 0.5  # Conservative estimate
    
    # No score available
    return 0.0