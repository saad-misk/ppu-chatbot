"""
Embedding cache to avoid re-embedding frequent queries.

Uses LRU cache for in-memory caching and optional disk persistence
for frequently asked questions (FAQ).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    LRU cache for query embeddings.
    
    Caches: query → embedding vector
    Reduces latency for repeated questions (common in chatbots).
    
    Args:
        max_size: Maximum cache entries (default: 1000)
        ttl: Time-to-live in seconds (default: 1 hour)
        persist_path: Optional path for disk persistence
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
        persist_path: Optional[str] = None,
    ):
        self.max_size = max_size
        self.ttl = ttl
        self.persist_path = Path(persist_path) if persist_path else None
        
        # OrderedDict for LRU behavior
        self._cache: OrderedDict[str, tuple[List[float], float]] = OrderedDict()
        
        # Stats
        self.hits = 0
        self.misses = 0
        
        # Load from disk if exists
        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()
    
    def get(self, query: str) -> Optional[List[float]]:
        """
        Get cached embedding for query.
        
        Args:
            query: Normalized query text
            
        Returns:
            Embedding vector or None if not cached
        """
        key = self._make_key(query)
        
        if key in self._cache:
            vector, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return vector
        
        self.misses += 1
        return None
    
    def set(self, query: str, embedding: List[float]) -> None:
        """
        Cache embedding for query.
        
        Args:
            query: Normalized query text
            embedding: Embedding vector
        """
        key = self._make_key(query)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = (embedding, time.time())
        
        # Persist periodically (every 100 writes)
        if self.persist_path and len(self._cache) % 100 == 0:
            self._save_to_disk()
    
    def _make_key(self, query: str) -> str:
        """Create cache key from query."""
        # Normalize for consistent caching
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.persist_path:
            return
        
        try:
            data = {
                "entries": [
                    {
                        "key": key,
                        "vector": vector,
                        "timestamp": timestamp,
                    }
                    for key, (vector, timestamp) in self._cache.items()
                ],
                "max_size": self.max_size,
                "ttl": self.ttl,
            }
            
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(data, f)
            
            logger.debug(f"Saved {len(self._cache)} entries to {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            
            now = time.time()
            loaded = 0
            
            for entry in data.get("entries", []):
                timestamp = entry["timestamp"]
                
                # Skip expired entries
                if now - timestamp > self.ttl:
                    continue
                
                self._cache[entry["key"]] = (entry["vector"], timestamp)
                loaded += 1
            
            logger.info(f"Loaded {loaded} cached embeddings from {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __repr__(self) -> str:
        return (
            f"EmbeddingCache(size={len(self._cache)}/{self.max_size}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


# Module-level singleton
_cache: Optional[EmbeddingCache] = None


def get_cache(persist_path: Optional[str] = None) -> EmbeddingCache:
    """Get or create the embedding cache singleton."""
    global _cache
    if _cache is None:
        # Default persist path in data directory
        if persist_path is None:
            persist_path = str(
                Path(__file__).parent.parent.parent / "data" / "cache" / "embeddings.json"
            )
        _cache = EmbeddingCache(persist_path=persist_path)
    return _cache