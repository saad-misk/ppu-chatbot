"""
Hybrid retriever - combines BM25 lexical scoring with embedding similarity.

This improves Arabic PDF recall by mixing exact-term matches (BM25) with
semantic similarity (embeddings). The BM25 index is cached in memory and
invalidated after ingestion.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

from nlp_engine.preprocessing.normalizer import normalize_for_classification
from nlp_engine.preprocessing.tokenizer import tokenize_no_stopwords
from nlp_engine.knowledge_base.chroma_store import get_store
from nlp_engine.rag.retriever import retrieve as embed_retrieve
from shared.config.settings import settings

logger = logging.getLogger(__name__)

_DEFAULT_BM25_K = 25
_DEFAULT_EMBED_K = 25
_DEFAULT_ALPHA = 0.6
_BM25_K1 = 1.5
_BM25_B = 0.75


@dataclass
class _BM25Index:
    docs: List[Dict[str, Any]]
    term_freqs: List[Counter]
    doc_lens: List[int]
    avgdl: float
    idf: Dict[str, float]

    def score(self, query_tokens: List[str]) -> List[float]:
        if not self.docs or not query_tokens:
            return [0.0] * len(self.docs)

        scores = [0.0] * len(self.docs)
        q_terms = Counter(query_tokens)
        for term in q_terms:
            if term not in self.idf:
                continue
            idf_t = self.idf[term]
            for i, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue
                norm = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * (self.doc_lens[i] / self.avgdl))
                scores[i] += idf_t * (tf * (_BM25_K1 + 1)) / norm
        return scores


_BM25_CACHE: Dict[str, Any] = {
    "index": None,
    "docs": None,
}


def invalidate_bm25_cache() -> None:
    _BM25_CACHE["index"] = None
    _BM25_CACHE["docs"] = None


def _tokenize_for_bm25(text: str) -> List[str]:
    if not text:
        return []
    clean = normalize_for_classification(text)
    tokens = tokenize_no_stopwords(clean)
    return [t for t in tokens if t]


def _build_bm25_index(docs: List[Dict[str, Any]]) -> _BM25Index:
    term_freqs: List[Counter] = []
    doc_freq: Dict[str, int] = defaultdict(int)
    doc_lens: List[int] = []

    for doc in docs:
        tokens = _tokenize_for_bm25(doc.get("document", ""))
        tf = Counter(tokens)
        term_freqs.append(tf)
        doc_len = len(tokens) or 1
        doc_lens.append(doc_len)
        for term in tf.keys():
            doc_freq[term] += 1

    num_docs = len(docs) or 1
    avgdl = (sum(doc_lens) / num_docs) if num_docs else 1.0

    idf = {
        term: math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
        for term, df in doc_freq.items()
    }

    return _BM25Index(
        docs=docs,
        term_freqs=term_freqs,
        doc_lens=doc_lens,
        avgdl=avgdl,
        idf=idf,
    )


def _get_bm25_index() -> Tuple[_BM25Index | None, List[Dict[str, Any]]]:
    if _BM25_CACHE["index"] is not None and _BM25_CACHE["docs"] is not None:
        return _BM25_CACHE["index"], _BM25_CACHE["docs"]

    store = get_store()
    docs = store.get_all()
    if not docs:
        return None, []

    index = _build_bm25_index(docs)
    _BM25_CACHE["index"] = index
    _BM25_CACHE["docs"] = docs
    logger.info("BM25 index built with %d documents.", len(docs))
    return index, docs


def bm25_retrieve(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    index, docs = _get_bm25_index()
    if index is None:
        return []

    tokens = _tokenize_for_bm25(query)
    if not tokens:
        return []

    scores = index.score(tokens)
    k = top_k or _DEFAULT_BM25_K
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for i in top_indices:
        score = scores[i]
        if score <= 0:
            continue
        doc = docs[i]
        results.append({
            "id": doc["id"],
            "document": doc["document"],
            "metadata": doc["metadata"],
            "bm25_score": score,
        })
    return results


def _distance_to_similarity(distance: float | None) -> float:
    if distance is None:
        return 0.0
    sim = 1.0 - float(distance)
    return max(0.0, min(sim, 1.0))


def _normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    max_score = max(score_map.values())
    if max_score <= 0:
        return {k: 0.0 for k in score_map}
    return {k: v / max_score for k, v in score_map.items()}


def hybrid_retrieve(
    query: str,
    n_results: int = 5,
    bm25_k: int | None = None,
    embed_k: int | None = None,
    alpha: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve chunks using BM25 + embeddings and fuse scores.

    alpha weights BM25 vs embeddings: hybrid = alpha * bm25 + (1 - alpha) * embed.
    """
    if not query or not query.strip() or n_results <= 0:
        return []

    bm25_k = bm25_k or settings.BM25_TOP_K or _DEFAULT_BM25_K
    embed_k = embed_k or settings.EMBED_TOP_K or _DEFAULT_EMBED_K
    alpha = settings.BM25_WEIGHT if alpha is None else alpha

    bm25_results = bm25_retrieve(query, top_k=bm25_k)
    embed_results = embed_retrieve(query, n_results=embed_k)

    combined: Dict[str, Dict[str, Any]] = {}

    for r in bm25_results:
        combined[r["id"]] = {
            **r,
            "distance": None,
            "embed_score": 0.0,
        }

    for r in embed_results:
        doc_id = r["id"]
        embed_score = _distance_to_similarity(r.get("distance"))
        if doc_id in combined:
            combined[doc_id].update({
                "document": r.get("document", combined[doc_id]["document"]),
                "metadata": r.get("metadata", combined[doc_id]["metadata"]),
                "distance": r.get("distance"),
                "embed_score": embed_score,
            })
        else:
            combined[doc_id] = {
                **r,
                "bm25_score": 0.0,
                "embed_score": embed_score,
            }

    bm25_scores = {doc_id: v.get("bm25_score", 0.0) for doc_id, v in combined.items()}
    embed_scores = {doc_id: v.get("embed_score", 0.0) for doc_id, v in combined.items()}

    bm25_norm = _normalize_scores(bm25_scores)
    embed_norm = _normalize_scores(embed_scores)

    for doc_id, item in combined.items():
        item["hybrid_score"] = (
            alpha * bm25_norm.get(doc_id, 0.0)
            + (1 - alpha) * embed_norm.get(doc_id, 0.0)
        )

    ranked = sorted(combined.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return ranked[:n_results]
