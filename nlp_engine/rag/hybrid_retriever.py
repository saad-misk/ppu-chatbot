"""
Hybrid retriever — BM25 (Elasticsearch) + embedding (ChromaDB) fusion.

Enhancements over v1
--------------------
1. Category exclusion   — news_events / jobs / community excluded by default
2. Priority boosting    — high/normal/low metadata used to adjust scores
3. Freshness penalty    — old news chunks penalized for current-state queries
4. Current-state detection — "who is X", "president", "dean" etc. triggers
                              stricter news filtering
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from nlp_engine.knowledge_base.es_store import bm25_search
from nlp_engine.rag.retriever import retrieve as embed_retrieve
from shared.config.settings import settings

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_BM25_K  = 10
_DEFAULT_EMBED_K = 10
_DEFAULT_ALPHA   = 0.6

# ── Categories excluded from retrieval by default ─────────────────────────────
# These are indexed but suppressed unless the caller explicitly asks for them.
# Rationale: news/jobs have many chunks that dominate similarity scores but
# return stale or irrelevant information for most chatbot queries.
_DEFAULT_EXCLUDE_CATEGORIES = {"news_events", "jobs", "community"}

# ── Priority score multipliers ────────────────────────────────────────────────
# Applied AFTER hybrid fusion, BEFORE final sort.
_PRIORITY_BOOST: Dict[str, float] = {
    "high":   1.30,   # academics, admissions, about, administration, contact
    "normal": 1.00,   # research, library, general
    "low":    0.60,   # news_events, jobs, community (if not excluded)
}

# ── Freshness penalty for current-state queries ───────────────────────────────
# When the query asks about "who is X" or "current president/dean",
# news chunks older than this threshold are heavily penalized.
_FRESHNESS_CUTOFF_DAYS  = 365      # 1 year
_OLD_NEWS_PENALTY       = 0.25     # multiply score by this for stale news

# Keywords that signal the user wants CURRENT information
_CURRENT_STATE_EN = [
    r"\bwho is\b", r"\bwho are\b", r"\bcurrent\b", r"\bpresent\b",
    r"\bpresident\b", r"\bdean\b", r"\bhead of\b", r"\bdirector\b",
    r"\bchair\b", r"\bchancellor\b", r"\bprovost\b", r"\bregistrar\b",
    r"\bnow\b", r"\btoday\b", r"\bcurrently\b",
]
_CURRENT_STATE_AR = [
    r"من هو", r"من هي", r"الحالي", r"الحالية", r"رئيس", r"عميد",
    r"مدير", r"رئيسة", r"الآن", r"حالياً",
]
_CURRENT_STATE_RE = re.compile(
    "|".join(_CURRENT_STATE_EN + _CURRENT_STATE_AR),
    re.IGNORECASE | re.UNICODE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Query analysis
# ══════════════════════════════════════════════════════════════════════════════

def _is_current_state_query(query: str) -> bool:
    """Return True if query is asking about the present state of something."""
    return bool(_CURRENT_STATE_RE.search(query))


# ══════════════════════════════════════════════════════════════════════════════
# Score utilities
# ══════════════════════════════════════════════════════════════════════════════

def _distance_to_similarity(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return max(0.0, min(1.0 - float(distance), 1.0))


def _normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    max_score = max(score_map.values())
    if max_score <= 0:
        return {k: 0.0 for k in score_map}
    return {k: v / max_score for k, v in score_map.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Post-fusion adjustments
# ══════════════════════════════════════════════════════════════════════════════

def _apply_priority_boost(results: List[Dict]) -> List[Dict]:
    """
    Multiply hybrid_score by the priority multiplier stored in metadata.
    This makes high-value categories (academics, admissions, about) float
    above low-value categories (news, jobs) even when similarity is close.
    """
    for r in results:
        priority = r.get("metadata", {}).get("priority", "normal")
        boost    = _PRIORITY_BOOST.get(priority, 1.0)
        r["hybrid_score"] = r.get("hybrid_score", 0.0) * boost
    return results


def _apply_freshness_penalty(
    results: List[Dict],
    is_current_query: bool,
) -> List[Dict]:
    """
    For current-state queries (who is X, current president, etc.),
    penalize news_events chunks that are older than _FRESHNESS_CUTOFF_DAYS.

    This prevents a 2017 news article mentioning an old president from
    outranking the about page that mentions the current one.
    """
    if not is_current_query:
        return results

    cutoff = datetime.now() - timedelta(days=_FRESHNESS_CUTOFF_DAYS)

    for r in results:
        meta = r.get("metadata", {})
        if meta.get("category") != "news_events":
            continue
        scraped_at = meta.get("scraped_at", "")
        if not scraped_at:
            # No date — assume old, penalize
            r["hybrid_score"] = r.get("hybrid_score", 0.0) * _OLD_NEWS_PENALTY
            continue
        try:
            scraped_dt = datetime.fromisoformat(scraped_at)
            if scraped_dt < cutoff:
                r["hybrid_score"] = r.get("hybrid_score", 0.0) * _OLD_NEWS_PENALTY
                logger.debug(
                    "Freshness penalty applied: %s (scraped %s)",
                    r.get("id", "?"), scraped_at[:10],
                )
        except (ValueError, TypeError):
            pass

    return results


def _apply_category_filter(
    results: List[Dict],
    exclude_categories: set[str],
) -> List[Dict]:
    """Remove results whose category is in the exclusion set."""
    if not exclude_categories:
        return results
    return [
        r for r in results
        if r.get("metadata", {}).get("category", "") not in exclude_categories
    ]


# ══════════════════════════════════════════════════════════════════════════════
# BM25 retrieval
# ══════════════════════════════════════════════════════════════════════════════

def bm25_retrieve(
    query: str,
    top_k: int | None = None,
    exclude_categories: set[str] | None = None,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []
    k = top_k or _DEFAULT_BM25_K
    try:
        results = bm25_search(query=query, top_k=k)
        # Filter excluded categories at retrieval time for ES
        if exclude_categories:
            results = [
                r for r in results
                if r.get("metadata", {}).get("category", "")
                not in exclude_categories
            ]
        return results
    except Exception as e:
        logger.exception("Elasticsearch BM25 retrieval failed: %s", e)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Main hybrid retriever
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_retrieve(
    query:              str,
    n_results:          int            = 8,
    bm25_k:             int | None     = None,
    embed_k:            int | None     = None,
    alpha:              float | None   = None,
    exclude_categories: set[str] | None = None,
    include_news:       bool           = False,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: BM25 + embedding fusion with category control.

    Parameters
    ----------
    query               : User query string
    n_results           : Final number of results to return
    bm25_k              : BM25 candidate pool size
    embed_k             : Embedding candidate pool size
    alpha               : BM25 weight (0=embed only, 1=BM25 only)
    exclude_categories  : Override the default exclusion set.
                          Pass set() to include everything.
    include_news        : Convenience flag — if True, news_events is NOT
                          excluded even if in the default exclusion list.
                          Use when query is clearly about news/events.
    """
    if not query or not query.strip() or n_results <= 0:
        return []

    bm25_k  = bm25_k  or getattr(settings, "BM25_TOP_K",  _DEFAULT_BM25_K)
    embed_k = embed_k or getattr(settings, "EMBED_TOP_K", _DEFAULT_EMBED_K)
    alpha   = getattr(settings, "BM25_WEIGHT", _DEFAULT_ALPHA) \
              if alpha is None else alpha

    # ── Build exclusion set ───────────────────────────────────────────────
    if exclude_categories is None:
        excluded = set(_DEFAULT_EXCLUDE_CATEGORIES)
    else:
        excluded = set(exclude_categories)

    if include_news:
        excluded.discard("news_events")

    # ── Detect current-state query ────────────────────────────────────────
    is_current = _is_current_state_query(query)
    if is_current:
        logger.debug("Current-state query detected: '%s'", query[:60])
        # For current-state queries: always exclude news (override include_news)
        excluded.add("news_events")

    # ── Retrieve from both sources ────────────────────────────────────────
    bm25_results = bm25_retrieve(
        query=query,
        top_k=bm25_k,
        exclude_categories=excluded,
    )
    embed_results = embed_retrieve(
        query=query,
        n_results=embed_k,
    )

    # ── Merge ─────────────────────────────────────────────────────────────
    combined: Dict[str, Dict[str, Any]] = {}

    for r in bm25_results:
        combined[r["id"]] = {
            "id":          r["id"],
            "document":    r["document"],
            "metadata":    r["metadata"],
            "bm25_score":  r.get("bm25_score", 0.0),
            "embed_score": 0.0,
            "distance":    None,
        }

    for r in embed_results:
        doc_id      = r["id"]
        embed_score = _distance_to_similarity(r.get("distance"))
        if doc_id in combined:
            combined[doc_id].update({
                "document":    r.get("document", combined[doc_id]["document"]),
                "metadata":    r.get("metadata", combined[doc_id]["metadata"]),
                "distance":    r.get("distance"),
                "embed_score": embed_score,
            })
        else:
            combined[doc_id] = {
                "id":          doc_id,
                "document":    r["document"],
                "metadata":    r["metadata"],
                "distance":    r.get("distance"),
                "bm25_score":  0.0,
                "embed_score": embed_score,
            }

    # ── Normalize ─────────────────────────────────────────────────────────
    bm25_norm  = _normalize_scores({k: v["bm25_score"]  for k, v in combined.items()})
    embed_norm = _normalize_scores({k: v["embed_score"] for k, v in combined.items()})

    # ── Hybrid score fusion ───────────────────────────────────────────────
    for doc_id, item in combined.items():
        item["hybrid_score"] = (
            alpha       * bm25_norm.get(doc_id, 0.0)
            + (1-alpha) * embed_norm.get(doc_id, 0.0)
        )

    results = list(combined.values())

    # ── Post-fusion: category filter (catches embed results not in BM25) ──
    results = _apply_category_filter(results, excluded)

    # ── Post-fusion: priority boost ───────────────────────────────────────
    results = _apply_priority_boost(results)

    # ── Post-fusion: freshness penalty for current-state queries ──────────
    results = _apply_freshness_penalty(results, is_current)

    # ── Final sort ────────────────────────────────────────────────────────
    results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

    logger.debug(
        "hybrid_retrieve: query='%s' | excluded=%s | current_state=%s "
        "| candidates=%d | returning=%d",
        query[:50], excluded, is_current, len(combined), min(n_results, len(results)),
    )

    return results[:n_results]


# ── BM25 cache invalidation (called by ingest pipeline) ───────────────────────
def invalidate_bm25_cache():
    """No-op — Elasticsearch has no in-process cache to invalidate."""
    logger.debug("invalidate_bm25_cache: ES-backed retriever, nothing to invalidate.")