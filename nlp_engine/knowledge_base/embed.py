"""
Sentence embedding — multilingual model that handles Arabic and English.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  • Trained on 50+ languages including Arabic
  • 384-dimensional vectors, fast CPU inference
  • Strong semantic similarity for mixed-language university queries

Alternative (higher quality, slower):
  intfloat/multilingual-e5-base  — requires "query: " prefix for queries

Usage
-----
    from nlp_engine.knowledge_base.embed import get_embedder

    embedder = get_embedder()
    vectors  = embedder.embed(["ما هي رسوم التسجيل؟", "What are the fees?"])
"""
from __future__ import annotations

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from pathlib import Path


logger = logging.getLogger(__name__)

# Multilingual model — handles Arabic as primary language, English as secondary
_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_LOCAL_MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models" / "embedder"


class Embedder:
    """Thin wrapper around a multilingual SentenceTransformer model."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        # Use local model if it exists, otherwise fall back to HuggingFace download
        model_path = str(_LOCAL_MODEL_DIR) if _LOCAL_MODEL_DIR.exists() else model_name
        logger.info("Loading multilingual embedding model from: %s", model_path)
        self._model    = SentenceTransformer(model_path)
        self.model_name = model_path
        self.dimension  = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dimension)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts (Arabic, English, or mixed).

        Returns
        -------
        List[List[float]] — one embedding vector per input text.
        """
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # unit vectors → dot product = cosine sim
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> List[float]:
        """Convenience wrapper for a single text."""
        return self.embed([text])[0]


# Module-level singleton
_embedder: Embedder | None = None


def get_embedder(model_name: str = _DEFAULT_MODEL) -> Embedder:
    """Return (or create) the process-level Embedder singleton."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder(model_name)
    return _embedder

def embed_with_cache(self, texts: List[str]) -> List[List[float]]:
    """
    Embed texts with caching for repeated queries.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    from nlp_engine.knowledge_base.cache import get_cache
    
    cache = get_cache()
    vectors = []
    texts_to_embed = []
    cache_indices = []
    
    # Check cache first
    for i, text in enumerate(texts):
        cached = cache.get(text) if len(text) < 200 else None  # Only cache short texts
        if cached is not None:
            vectors.append(cached)
        else:
            vectors.append(None)  # Placeholder
            texts_to_embed.append(text)
            cache_indices.append(i)
    
    # Embed uncached texts
    if texts_to_embed:
        new_vectors = self.embed(texts_to_embed)
        
        for idx, vector in zip(cache_indices, new_vectors):
            vectors[idx] = vector
            # Cache the result
            cache.set(texts[idx], vector)
    
    return vectors