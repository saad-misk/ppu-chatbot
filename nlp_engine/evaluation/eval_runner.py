"""
Evaluation runner — loads test fixtures and computes all NLP metrics.

Fixture format (tests/fixtures/sample_queries.json):
[
  {
    "text":     "What are the Computer Science tuition fees?",
    "intent":   "faq_fees",
    "entities": [{"type": "DEPARTMENT", "value": "Computer Science"}],
    "relevant_chunk_ids": ["abc123", "def456"]   # optional
  },
  ...
]

Usage
-----
    from nlp_engine.evaluation.eval_runner import run_evaluation
    metrics = run_evaluation()
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from nlp_engine.intent.classifier import get_classifier
from nlp_engine.ner.extractor import extract_entities
from nlp_engine.rag.hybrid_retriever import hybrid_retrieve
from nlp_engine.preprocessing.normalizer import normalize_for_classification
from nlp_engine.evaluation.metrics import (
    intent_accuracy,
    per_class_intent_f1,
    macro_f1,
    ner_precision_recall_f1,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)

logger = logging.getLogger(__name__)

_FIXTURES_PATH = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "sample_queries.json"
_RETRIEVAL_K   = 3


def run_evaluation(fixtures_path: str | Path | None = None) -> Dict:
    """
    Run the full evaluation suite against fixture data.

    Returns
    -------
    dict with keys:
        intent_accuracy          : float
        intent_macro_f1          : float
        intent_per_class         : dict  — {label: {precision, recall, f1, support}}
        ner_precision            : float
        ner_recall               : float
        ner_f1                   : float
        retrieval_precision_at_k : float
        retrieval_recall_at_k    : float
        retrieval_mrr            : float
        retrieval_ndcg_at_k      : float
        num_examples             : int
    """
    path = Path(fixtures_path) if fixtures_path else _FIXTURES_PATH

    if not path.exists():
        logger.warning("Fixtures file not found at %s — returning zero metrics.", path)
        return {
            "intent_accuracy":          0.0,
            "intent_macro_f1":          0.0,
            "intent_per_class":         {},
            "ner_precision":            0.0,
            "ner_recall":               0.0,
            "ner_f1":                   0.0,
            "retrieval_precision_at_k": 0.0,
            "retrieval_recall_at_k":    0.0,
            "retrieval_mrr":            0.0,
            "retrieval_ndcg_at_k":      0.0,
            "num_examples":             0,
        }

    with open(path, "r", encoding="utf-8") as f:
        fixtures: List[Dict] = json.load(f)

    classifier = get_classifier()

    pred_intents:      List[str]       = []
    true_intents:      List[str]       = []
    pred_entities_all: List[List[Dict]] = []
    true_entities_all: List[List[Dict]] = []
    retrieved_ids_all: List[List[str]]  = []
    relevant_ids_all:  List[List[str]]  = []

    for example in fixtures:
        text         = example.get("text", "")
        true_intent  = example.get("intent", "unknown")
        true_ents    = example.get("entities", [])
        relevant_ids = example.get("relevant_chunk_ids", [])

        # ── Intent ──────────────────────────────────────────────────────────
        clean  = normalize_for_classification(text)
        result = classifier.predict(clean)
        pred_intents.append(result["intent"])
        true_intents.append(true_intent)

        # ── NER ─────────────────────────────────────────────────────────────
        pred_ents = extract_entities(text)
        pred_entities_all.append(pred_ents)
        true_entities_all.append(true_ents)

        # ── Retrieval (only when fixture has ground-truth chunk IDs) ────────
        if relevant_ids:
            chunks = hybrid_retrieve(clean, n_results=_RETRIEVAL_K)
            retrieved_ids_all.append([c["id"] for c in chunks])
            relevant_ids_all.append(relevant_ids)

    # ── Compute all metrics ──────────────────────────────────────────────────
    intent_acc   = intent_accuracy(pred_intents, true_intents)
    per_class    = per_class_intent_f1(pred_intents, true_intents)
    m_f1         = macro_f1(per_class)
    ner_scores   = ner_precision_recall_f1(pred_entities_all, true_entities_all)

    has_retrieval = bool(retrieved_ids_all)
    p_at_k  = precision_at_k(retrieved_ids_all, relevant_ids_all, k=_RETRIEVAL_K) if has_retrieval else 0.0
    r_at_k  = recall_at_k(retrieved_ids_all,    relevant_ids_all, k=_RETRIEVAL_K) if has_retrieval else 0.0
    mrr     = mean_reciprocal_rank(retrieved_ids_all, relevant_ids_all)            if has_retrieval else 0.0
    ndcg    = ndcg_at_k(retrieved_ids_all,       relevant_ids_all, k=_RETRIEVAL_K) if has_retrieval else 0.0

    metrics = {
        "intent_accuracy":          round(intent_acc, 4),
        "intent_macro_f1":          m_f1,
        "intent_per_class":         per_class,
        "ner_precision":            ner_scores["precision"],
        "ner_recall":               ner_scores["recall"],
        "ner_f1":                   ner_scores["f1"],
        "retrieval_precision_at_k": p_at_k,
        "retrieval_recall_at_k":    r_at_k,
        "retrieval_mrr":            mrr,
        "retrieval_ndcg_at_k":      ndcg,
        "num_examples":             len(fixtures),
    }

    logger.info("Evaluation complete: %s", {
        k: v for k, v in metrics.items() if k != "intent_per_class"
    })
    return metrics
