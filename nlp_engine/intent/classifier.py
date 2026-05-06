"""
Intent Classifier — uses a multilingual/Arabic-capable BERT model.

Model priority:
  1. Fine-tuned model saved at  data/models/intent/    (best accuracy)
  2. Zero-shot pipeline via     joeddav/xlm-roberta-large-xnli
     (handles Arabic + English without any training data)

Why these models?
  • Fine-tuning base: CAMeL-Lab/bert-base-arabic-camelbert-mix
    — Arabic BERT trained on a large Arabic corpus, handles MSA and
      dialectal Arabic well; fallback to bert-base-multilingual-cased
      if unavailable.
  • Zero-shot:  joeddav/xlm-roberta-large-xnli
    — XLM-RoBERTa fine-tuned on XNLI; strong multilingual zero-shot
      classification, including Arabic.

Usage
-----
    from nlp_engine.intent.classifier import get_classifier
    clf = get_classifier()
    result = clf.predict("ما هي رسوم التسجيل في قسم هندسة الحاسوب؟")
    # {"intent": "faq_fees", "confidence": 0.91, "all_scores": {...}}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

# Paths
_LABELS_PATH       = Path(__file__).parent / "labels.json"
_FINE_TUNED_DIR    = Path(__file__).parent.parent.parent / "data" / "models" / "intent"

# Arabic-first base model for fine-tuning
_ARABIC_BASE_MODEL  = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
_MULTILINGUAL_BASE  = "bert-base-multilingual-cased"

# Zero-shot fallback — strong multilingual, including Arabic
_ZERO_SHOT_MODEL    = "joeddav/xlm-roberta-large-xnli"


class IntentClassifier:
    """
    Loads and caches the intent classification model.

    Priority:
      1. Fine-tuned model at ``data/models/intent/``
      2. Multilingual zero-shot via xlm-roberta-large-xnli
    """

    def __init__(self):
        with open(_LABELS_PATH, "r", encoding="utf-8") as f:
            self.labels: list[str] = json.load(f)
        self.id2label: Dict[int, str] = {i: lbl for i, lbl in enumerate(self.labels)}
        self.label2id: Dict[str, int] = {lbl: i for i, lbl in enumerate(self.labels)}

        self._pipe       = None
        self._tokenizer  = None
        self._model      = None
        self._mode       = None

        self._load()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if _FINE_TUNED_DIR.exists() and any(_FINE_TUNED_DIR.iterdir()):
            logger.info("Loading fine-tuned intent model from %s", _FINE_TUNED_DIR)
            self._tokenizer = AutoTokenizer.from_pretrained(str(_FINE_TUNED_DIR))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(_FINE_TUNED_DIR)
            )
            self._model.eval()
            self._mode = "finetuned"
        else:
            logger.warning(
                "No fine-tuned model found at %s.\n"
                "Falling back to zero-shot XLM-RoBERTa (%s).\n"
                "Run nlp_engine/intent/fine_tune.py to train a dedicated Arabic model.",
                _FINE_TUNED_DIR,
                _ZERO_SHOT_MODEL,
            )
            self._pipe = pipeline(
                "zero-shot-classification",
                model=_ZERO_SHOT_MODEL,
                device=-1,   # CPU; set to 0 for GPU
            )
            self._mode = "zero_shot"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict:
        """
        Predict intent for *text* (Arabic or English).

        Returns
        -------
        dict:
            intent      : str   — top intent label
            confidence  : float — probability of top intent
            all_scores  : dict  — {label: score} for all intents
        """
        if not text or not text.strip():
            return {"intent": "unknown", "confidence": 0.0, "all_scores": {}}

        if self._mode == "finetuned":
            return self._predict_finetuned(text)
        return self._predict_zero_shot(text)

    def _predict_finetuned(self, text: str) -> Dict:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs   = torch.softmax(logits, dim=-1).squeeze()
        top_idx = int(torch.argmax(probs))
        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(self.labels))}
        return {
            "intent":     self.id2label[top_idx],
            "confidence": float(probs[top_idx]),
            "all_scores": all_scores,
        }

    def _predict_zero_shot(self, text: str) -> Dict:
        result = self._pipe(text, candidate_labels=self.labels, multi_label=False)
        scores = dict(zip(result["labels"], result["scores"]))
        return {
            "intent":     result["labels"][0],
            "confidence": float(result["scores"][0]),
            "all_scores": scores,
        }


# Module-level singleton
_classifier: IntentClassifier | None = None


def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
