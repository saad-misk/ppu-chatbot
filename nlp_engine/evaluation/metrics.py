"""
Evaluation metrics for the PPU NLP Engine.

Implements:
  • Intent accuracy        — % of queries where top-1 intent is correct
  • Per-class intent F1    — precision / recall / F1 per intent label
  • NER token-level F1     — precision / recall / F1 over entity spans
  • Retrieval Precision@k  — fraction of retrieved chunks that are relevant
  • Mean Reciprocal Rank   — MRR over all queries
  • Recall@k               — fraction of relevant chunks retrieved in top-k
  • NDCG@k                 — normalised discounted cumulative gain

All functions accept plain Python lists so they can be used independently
of any ML framework.
"""
from __future__ import annotations

import math
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Intent accuracy
# ---------------------------------------------------------------------------

def intent_accuracy(
    predictions: List[str],
    ground_truth: List[str],
) -> float:
    """
    Compute top-1 intent accuracy.

    Parameters
    ----------
    predictions  : list of predicted intent labels
    ground_truth : list of true intent labels (same length)

    Returns
    -------
    float — accuracy in [0, 1]
    """
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(ground_truth)


# ---------------------------------------------------------------------------
# Per-class intent metrics
# ---------------------------------------------------------------------------

def per_class_intent_f1(
    predictions: List[str],
    ground_truth: List[str],
    labels: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, F1, and support for intent classification.

    Parameters
    ----------
    predictions  : list of predicted intent labels
    ground_truth : list of true intent labels (same length)
    labels       : optional list of label names to include; defaults to all
                   labels observed in ground_truth

    Returns
    -------
    dict — { label: {"precision": float, "recall": float, "f1": float, "support": int} }
    """
    if not predictions or len(predictions) != len(ground_truth):
        return {}

    all_labels = labels if labels is not None else sorted(set(ground_truth))
    results: Dict[str, Dict[str, float]] = {}

    for label in all_labels:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != label and g == label)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        results[label] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   support,
        }

    return results


def macro_f1(per_class: Dict[str, Dict[str, float]]) -> float:
    """Compute macro-averaged F1 from the per_class_intent_f1 output."""
    if not per_class:
        return 0.0
    return round(sum(v["f1"] for v in per_class.values()) / len(per_class), 4)


# ---------------------------------------------------------------------------
# NER metrics (token / span level)
# ---------------------------------------------------------------------------

def _spans_to_set(entities: List[Dict]) -> set:
    """Convert entity list to a set of (type, value) tuples for comparison."""
    return {(e["type"], e["value"].lower().strip()) for e in entities}


def ner_precision_recall_f1(
    predicted_entities: List[List[Dict]],
    true_entities: List[List[Dict]],
) -> Dict[str, float]:
    """
    Compute micro-averaged precision, recall, and F1 for NER.

    Parameters
    ----------
    predicted_entities : list of entity lists, one per example
    true_entities      : list of ground-truth entity lists (same length)

    Returns
    -------
    dict with keys: precision, recall, f1
    """
    tp = fp = fn = 0
    for pred, true in zip(predicted_entities, true_entities):
        pred_set = _spans_to_set(pred)
        true_set = _spans_to_set(true)
        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Retrieval Precision@k
# ---------------------------------------------------------------------------

def precision_at_k(
    retrieved_ids: List[List[str]],
    relevant_ids:  List[List[str]],
    k: int = 3,
) -> float:
    """
    Compute mean Precision@k over all queries.

    Parameters
    ----------
    retrieved_ids : list of retrieved chunk ID lists, one per query
    relevant_ids  : list of ground-truth relevant chunk ID lists
    k             : cutoff rank

    Returns
    -------
    float — mean Precision@k in [0, 1]
    """
    if not retrieved_ids:
        return 0.0
    scores = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        top_k   = retrieved[:k]
        rel_set = set(relevant)
        hits    = sum(1 for r in top_k if r in rel_set)
        scores.append(hits / k if k > 0 else 0.0)
    return round(sum(scores) / len(scores), 4)


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: List[List[str]],
    relevant_ids:  List[List[str]],
    k: int = 3,
) -> float:
    """
    Compute mean Recall@k over all queries.

    Parameters
    ----------
    retrieved_ids : list of retrieved chunk ID lists, one per query
    relevant_ids  : list of ground-truth relevant chunk ID lists
    k             : cutoff rank

    Returns
    -------
    float — mean Recall@k in [0, 1]
    """
    if not retrieved_ids:
        return 0.0
    scores = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        if not relevant:
            scores.append(0.0)
            continue
        top_k   = set(retrieved[:k])
        rel_set = set(relevant)
        hits    = len(top_k & rel_set)
        scores.append(hits / len(rel_set))
    return round(sum(scores) / len(scores), 4)


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank (MRR)
# ---------------------------------------------------------------------------

def mean_reciprocal_rank(
    retrieved_ids: List[List[str]],
    relevant_ids:  List[List[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) over all queries.

    Returns
    -------
    float — MRR in [0, 1]
    """
    if not retrieved_ids:
        return 0.0
    rr_scores = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        rel_set = set(relevant)
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in rel_set:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    return round(sum(rr_scores) / len(rr_scores), 4)


# ---------------------------------------------------------------------------
# NDCG@k
# ---------------------------------------------------------------------------

def ndcg_at_k(
    retrieved_ids: List[List[str]],
    relevant_ids:  List[List[str]],
    k: int = 3,
) -> float:
    """
    Compute mean Normalised Discounted Cumulative Gain (NDCG@k).

    Uses binary relevance: 1 if the chunk is relevant, 0 otherwise.

    Returns
    -------
    float — mean NDCG@k in [0, 1]
    """
    if not retrieved_ids:
        return 0.0

    def _dcg(ranked: List[str], rel_set: set, k_: int) -> float:
        return sum(
            (1.0 / math.log2(i + 2))  # log2(rank+1), rank is 0-indexed
            for i, doc_id in enumerate(ranked[:k_])
            if doc_id in rel_set
        )

    scores = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        rel_set = set(relevant)
        dcg  = _dcg(retrieved, rel_set, k)
        # Ideal DCG: all relevant docs at the top
        ideal_ranked = list(rel_set)[:k]
        idcg = _dcg(ideal_ranked, rel_set, k)
        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return round(sum(scores) / len(scores), 4)
