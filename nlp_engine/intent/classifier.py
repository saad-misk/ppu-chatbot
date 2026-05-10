"""
Intent Classifier — uses a multilingual/Arabic-capable BERT model.

Model priority:
  1. Fine-tuned model saved at  data/models/intent/    (best accuracy)
  2. Zero-shot pipeline via     joeddav/xlm-roberta-base-xnli
     (handles Arabic + English without any training data)
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

# Paths
_LABELS_PATH = Path(__file__).parent / "labels.json"
_FINE_TUNED_DIR = Path(__file__).parent.parent.parent / "data" / "models" / "intent"
_INTENT_METADATA_PATH = Path(__file__).parent / "intent_metadata.json"

# Arabic-first base model for fine-tuning
_ARABIC_BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
_MULTILINGUAL_BASE = "bert-base-multilingual-cased"

# Zero-shot fallback — base model is faster than large
_ZERO_SHOT_MODEL = "joeddav/xlm-roberta-base-xnli"  # 270M params (down from 560M)

# Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
MAX_INPUT_LENGTH = 512
MIN_INPUT_LENGTH = 3


def _get_device() -> int:
    """Detect best available device."""
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return 0
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Using Apple MPS (Metal)")
        return 0  # MPS uses device 0 in HuggingFace
    logger.info("Using CPU")
    return -1


class IntentClassifier:
    """
    Loads and caches the intent classification model.

    Priority:
      1. Fine-tuned model at ``data/models/intent/``
      2. Multilingual zero-shot via xlm-roberta-base-xnli
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        
        # Load intent labels
        with open(_LABELS_PATH, "r", encoding="utf-8") as f:
            self.labels: List[str] = json.load(f)
        self.id2label: Dict[int, str] = {i: lbl for i, lbl in enumerate(self.labels)}
        self.label2id: Dict[str, int] = {lbl: i for i, lbl in enumerate(self.labels)}
        
        # Load intent metadata (optional)
        self.intent_metadata = self._load_intent_metadata()
        
        # Initialize model components
        self._pipe = None
        self._tokenizer = None
        self._model = None
        self._mode = None
        self._device = _get_device()
        
        # Statistics
        self.prediction_stats = Counter()
        
        # Load model
        self._load()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_intent_metadata(self) -> Dict:
        """Load intent metadata if available."""
        if _INTENT_METADATA_PATH.exists():
            try:
                with open(_INTENT_METADATA_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load intent metadata: {e}")
        return {}

    def _load(self) -> None:
        """Load the best available model."""
        if not self._load_finetuned_model():
            self._load_zero_shot_model()

    def _load_finetuned_model(self) -> bool:
        """Attempt to load fine-tuned model. Returns True if successful."""
        if not _FINE_TUNED_DIR.exists() or not any(_FINE_TUNED_DIR.iterdir()):
            return False
        
        try:
            logger.info("Loading fine-tuned intent model from %s", _FINE_TUNED_DIR)
            self._tokenizer = AutoTokenizer.from_pretrained(str(_FINE_TUNED_DIR))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(_FINE_TUNED_DIR)
            )
            
            # Move to appropriate device
            if self._device >= 0:
                self._model = self._model.to(self._device)
            
            self._validate_labels()
            self._model.eval()
            self._mode = "finetuned"
            
            logger.info(
                "Fine-tuned model loaded successfully (%d labels)", 
                len(self.labels)
            )
            return True
            
        except Exception as e:
            logger.error("Failed to load fine-tuned model: %s", e)
            return False

    def _load_zero_shot_model(self) -> None:
        """Load zero-shot classification pipeline."""
        logger.warning(
            "No fine-tuned model found at %s.\n"
            "Falling back to zero-shot XLM-RoBERTa (%s).\n"
            "Run nlp_engine/intent/fine_tune.py to train a dedicated Arabic model.\n"
            "Prediction will be slow (~1-2s per query on CPU).",
            _FINE_TUNED_DIR,
            _ZERO_SHOT_MODEL,
        )
        
        self._pipe = pipeline(
            "zero-shot-classification",
            model=_ZERO_SHOT_MODEL,
            device=self._device,
        )
        self._mode = "zero_shot"

    def _validate_labels(self) -> None:
        """Ensure model labels match labels.json configuration."""
        if not hasattr(self._model.config, 'id2label'):
            logger.warning("Model doesn't have id2label mapping")
            return
        
        model_labels = self._model.config.id2label
        config_labels = self.labels
        
        if len(model_labels) != len(config_labels):
            logger.error(
                "Label count mismatch! Model has %d labels but labels.json has %d",
                len(model_labels), len(config_labels)
            )
        
        # Check individual label names
        for idx in range(min(len(model_labels), len(config_labels))):
            if model_labels[idx] != config_labels[idx]:
                logger.warning(
                    "Label mismatch at index %d: model='%s', config='%s'",
                    idx, model_labels[idx], config_labels[idx]
                )

    def _validate_input(self, text: str) -> Optional[str]:
        """Validate and preprocess input text. Returns None if invalid."""
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        if len(text) < MIN_INPUT_LENGTH:
            logger.debug("Input too short (%d chars): '%s'", len(text), text)
            # Still process very short inputs but log them
            return text
        
        if len(text) > MAX_INPUT_LENGTH:
            logger.warning(
                "Input truncated from %d to %d chars", 
                len(text), MAX_INPUT_LENGTH
            )
            text = text[:MAX_INPUT_LENGTH]
        
        return text

    # ------------------------------------------------------------------
    # Prediction Methods
    # ------------------------------------------------------------------

    def predict(
        self, 
        text: str, 
        confidence_threshold: Optional[float] = None
    ) -> Dict:
        """
        Predict intent for *text* (Arabic or English).

        Args:
            text: Input text to classify
            confidence_threshold: Override default confidence threshold

        Returns:
            dict with keys:
                - intent: str — top intent label (or 'fallback' if low confidence)
                - confidence: float — probability of top intent
                - all_scores: dict — {label: score} for all intents
                - below_threshold: bool — True if confidence < threshold
                - intent_type: str — type of intent (if metadata available)
        """
        # Validate input
        text = self._validate_input(text)
        if text is None:
            return {
                "intent": "unknown", 
                "confidence": 0.0, 
                "all_scores": {},
                "below_threshold": True,
            }
        
        # Predict
        if self._mode == "finetuned":
            result = self._predict_finetuned(text)
        else:
            result = self._predict_zero_shot(text)
        
        # Update statistics
        self.prediction_stats[result["intent"]] += 1
        
        # Apply confidence threshold
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        if result["confidence"] < threshold:
            result["intent"] = "fallback"
            result["below_threshold"] = True
        else:
            result["below_threshold"] = False
        
        # Add metadata
        if result["intent"] in self.intent_metadata:
            result.update(self.intent_metadata[result["intent"]])
        
        return result

    def _predict_finetuned(self, text: str) -> Dict:
        """Predict using fine-tuned model."""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        
        # Move inputs to correct device
        if self._device >= 0:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self._model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1).squeeze()
        top_idx = int(torch.argmax(probs))
        
        all_scores = {
            self.id2label[i]: float(probs[i]) 
            for i in range(len(self.labels))
        }
        
        return {
            "intent": self.id2label[top_idx],
            "confidence": float(probs[top_idx]),
            "all_scores": all_scores,
        }

    def _predict_zero_shot(self, text: str) -> Dict:
        """Predict using zero-shot classification."""
        result = self._pipe(
            text, 
            candidate_labels=self.labels, 
            multi_label=False
        )
        
        scores = dict(zip(result["labels"], result["scores"]))
        
        return {
            "intent": result["labels"][0],
            "confidence": float(result["scores"][0]),
            "all_scores": scores,
        }

    def predict_batch(
        self, 
        texts: List[str], 
        confidence_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Batch prediction for multiple texts.
        
        Args:
            texts: List of input texts
            confidence_threshold: Override default confidence threshold
            
        Returns:
            List of prediction dictionaries (same format as predict())
        """
        if not texts:
            return []
        
        # Validate inputs
        valid_texts = []
        results = []
        
        for text in texts:
            validated = self._validate_input(text)
            if validated is None:
                results.append({
                    "intent": "unknown",
                    "confidence": 0.0,
                    "all_scores": {},
                    "below_threshold": True,
                })
            else:
                valid_texts.append(validated)
                results.append(None)  # Placeholder
        
        if not valid_texts:
            return results
        
        # Batch predict valid texts
        if self._mode == "finetuned":
            batch_results = self._predict_batch_finetuned(valid_texts)
        else:
            # Zero-shot doesn't batch well, but we can process sequentially
            batch_results = [self._predict_zero_shot(t) for t in valid_texts]
        
        # Apply thresholds and fill in results
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        result_idx = 0
        
        for i, result in enumerate(results):
            if result is None:
                result = batch_results[result_idx]
                result_idx += 1
                
                # Update statistics
                self.prediction_stats[result["intent"]] += 1
                
                # Apply confidence threshold
                if result["confidence"] < threshold:
                    result["intent"] = "fallback"
                    result["below_threshold"] = True
                else:
                    result["below_threshold"] = False
                
                # Add metadata
                if result["intent"] in self.intent_metadata:
                    result.update(self.intent_metadata[result["intent"]])
                
                results[i] = result
        
        return results

    def _predict_batch_finetuned(self, texts: List[str]) -> List[Dict]:
        """Efficient batch prediction for fine-tuned model."""
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        
        # Move inputs to correct device
        if self._device >= 0:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self._model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1)
        
        results = []
        for i in range(len(texts)):
            top_idx = int(torch.argmax(probs[i]))
            all_scores = {
                self.id2label[j]: float(probs[i][j])
                for j in range(len(self.labels))
            }
            results.append({
                "intent": self.id2label[top_idx],
                "confidence": float(probs[i][top_idx]),
                "all_scores": all_scores,
            })
        
        return results

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def get_top_k_intents(self, text: str, k: int = 3) -> List[Dict]:
        """
        Get top-k intent predictions with scores.
        Useful for disambiguation.
        """
        result = self.predict(text)
        
        # Sort all scores and get top-k
        sorted_intents = sorted(
            result["all_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return [
            {"intent": intent, "confidence": score}
            for intent, score in sorted_intents
        ]

    def get_statistics(self) -> Dict:
        """Get prediction statistics."""
        total = sum(self.prediction_stats.values())
        return {
            "total_predictions": total,
            "distribution": dict(self.prediction_stats.most_common()),
            "unique_intents": len(self.prediction_stats),
            "unknown_rate": (
                self.prediction_stats.get("unknown", 0) / max(total, 1)
            ),
            "fallback_rate": (
                self.prediction_stats.get("fallback", 0) / max(total, 1)
            ),
            "mode": self._mode,
        }

    def clear_cache(self):
        """Clear prediction cache (useful for testing)."""
        self.prediction_stats.clear()
        if hasattr(self, '_predict_zero_shot') and hasattr(self._predict_zero_shot, 'cache_clear'):
            self._predict_zero_shot.cache_clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_classifier: Optional[IntentClassifier] = None


def get_classifier(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    force_reload: bool = False,
) -> IntentClassifier:
    """
    Get or create the intent classifier singleton.
    
    Args:
        confidence_threshold: Confidence threshold for fallback
        force_reload: If True, recreate the classifier
        
    Returns:
        IntentClassifier instance
    """
    global _classifier
    
    if _classifier is None or force_reload:
        _classifier = IntentClassifier(confidence_threshold=confidence_threshold)
    
    return _classifier


def reset_classifier():
    """Reset the classifier singleton (useful for testing)."""
    global _classifier
    _classifier = None