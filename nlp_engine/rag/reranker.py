"""
RAG Reranker — re-ranks hybrid-retrieved chunks.

Scoring pipeline (in order):
  1. Base score       — hybrid_score from retriever (BM25 + embedding fusion)
  2. Priority boost   — high/normal/low metadata multiplier
  3. Freshness penalty— old news penalized for current-state queries
  4. Threshold filter — chunks below minimum similarity dropped
  5. top_k cutoff

Note: priority boost and freshness penalty are already applied by
hybrid_retriever before chunks reach here. The reranker re-applies them
as a safety net in case chunks arrive from a different retrieval path
(e.g. embed-only fallback). Double-application is harmless because
multiplying by 1.0 (normal priority, not stale) is a no-op.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_THRESHOLD   = 0.25
_DEFAULT_TOP_K       = 3

# ── Priority multipliers (mirror hybrid_retriever constants) ──────────────────
_PRIORITY_BOOST: Dict[str, float] = {
    "high":   1.30,
    "normal": 1.00,
    "low":    0.60,
}

# ── Freshness (mirror hybrid_retriever constants) ─────────────────────────────
_FRESHNESS_CUTOFF_DAYS = 365
_OLD_NEWS_PENALTY      = 0.25

_CURRENT_STATE_RE = re.compile(
    r"\bwho is\b|\bwho are\b|\bcurrent\b|\bpresident\b|\bdean\b"
    r"|\bhead of\b|\bdirector\b|\bregistrar\b|\bnow\b|\bcurrently\b"
    r"|من هو|من هي|الحالي|الحالية|رئيس|عميد|مدير",
    re.IGNORECASE | re.UNICODE,
)


def _is_current_state_query(query: str) -> bool:
    return bool(query and _CURRENT_STATE_RE.search(query))


# ══════════════════════════════════════════════════════════════════════════════
# Score extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_base_score(chunk: Dict) -> float:
    """
    Extract the best available base score from a chunk dict.
    Priority: hybrid_score > score > distance-derived > bm25 fallback.
    """
    if "hybrid_score" in chunk and chunk["hybrid_score"] is not None:
        return float(chunk["hybrid_score"])
    if "score" in chunk and chunk["score"] is not None:
        return float(chunk["score"])
    if "distance" in chunk and chunk["distance"] is not None:
        return max(0.0, 1.0 - float(chunk["distance"]))
    if chunk.get("bm25_score", 0) > 0:
        return 0.5   # keyword-only match, no embedding signal
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Adjustments
# ══════════════════════════════════════════════════════════════════════════════

def _boost_by_priority(score: float, chunk: Dict) -> float:
    priority = chunk.get("metadata", {}).get("priority", "normal")
    return score * _PRIORITY_BOOST.get(priority, 1.0)


def _penalize_stale_news(score: float, chunk: Dict, is_current: bool) -> float:
    """
    Penalize old news chunks when the query asks about current state.
    This is the fix for: "who is the president?" returning 2017 news
    instead of the about page with the current president.
    """
    if not is_current:
        return score
    meta = chunk.get("metadata", {})
    if meta.get("category") != "news_events":
        return score

    scraped_at = meta.get("scraped_at", "")
    if not scraped_at:
        return score * _OLD_NEWS_PENALTY   # no date = assume old

    try:
        cutoff = datetime.now() - timedelta(days=_FRESHNESS_CUTOFF_DAYS)
        if datetime.fromisoformat(scraped_at) < cutoff:
            logger.debug(
                "Reranker freshness penalty: chunk %s (scraped %s)",
                chunk.get("id", "?"), scraped_at[:10],
            )
            return score * _OLD_NEWS_PENALTY
    except (ValueError, TypeError):
        pass

    return score


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def rerank(
    chunks:    List[Dict],
    top_k:     int   = _DEFAULT_TOP_K,
    threshold: float = _DEFAULT_THRESHOLD,
    query:     str   = "",
) -> List[Dict]:
    """
    Re-rank retrieved chunks with priority + freshness adjustments.

    Parameters
    ----------
    chunks    : Chunks from hybrid_retrieve (or any retriever)
    top_k     : Maximum chunks to return
    threshold : Minimum final score — chunks below this are dropped
    query     : Original user query — used for current-state detection.
                Pass it for best results; safe to omit (disables freshness).

    Returns
    -------
    List of chunk dicts with "score" field set, sorted best-first.
    """
    if not chunks:
        return []

    is_current = _is_current_state_query(query)
    scored     = []

    for chunk in chunks:
        base  = _extract_base_score(chunk)
        score = _boost_by_priority(base, chunk)
        score = _penalize_stale_news(score, chunk, is_current)

        if score < threshold:
            continue

        scored.append({**chunk, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    result = scored[:top_k]

    logger.debug(
        "rerank: %d → %d chunks (threshold=%.2f, top_k=%d, current_state=%s)",
        len(chunks), len(result), threshold, top_k, is_current,
    )

    return result