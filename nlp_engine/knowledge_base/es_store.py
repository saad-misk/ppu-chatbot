"""
Elasticsearch client wrapper with connection management and error handling.

Features:
  • Connection pooling with retry logic
  • Health check on initialization
  • Arabic text analysis
  • Bulk operations with chunking
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional

from elasticsearch import Elasticsearch, exceptions as es_exceptions

from shared.config.settings import settings

logger = logging.getLogger(__name__)

INDEX_NAME = "ppu_knowledge"
_MAX_RETRIES = 3
_BULK_CHUNK_SIZE = 500  # Documents per bulk request


class ESStore:
    """Elasticsearch wrapper with connection management."""
    
    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        max_retries: int = _MAX_RETRIES,
    ):
        hosts = hosts or [getattr(settings, "ES_HOST", "http://localhost:9200")]
        
        self._client = Elasticsearch(
            hosts=hosts,
            max_retries=max_retries,
            retry_on_timeout=True,
            request_timeout=30,
        )
        
        # Verify connection
        if not self._health_check():
            logger.warning("Elasticsearch not available - BM25 search will be disabled")
            self._available = False
        else:
            self._available = True
            self._ensure_index()
    
    def _health_check(self) -> bool:
        """Check if Elasticsearch is reachable."""
        try:
            return self._client.ping()
        except Exception:
            return False
    
    def is_available(self) -> bool:
        """Check if Elasticsearch is available."""
        return self._available
    
    def _ensure_index(self) -> None:
        """Create index if it doesn't exist."""
        if self._client.indices.exists(index=INDEX_NAME):
            return
        
        self._client.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "arabic_analyzer": {
                                "type": "arabic",
                            },
                            "arabic_english_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "arabic_normalization"],
                            },
                        },
                    },
                },
                "mappings": {
                    "properties": {
                        "document": {
                            "type": "text",
                            "analyzer": "arabic_english_analyzer",
                            "fields": {
                                "english": {
                                    "type": "text",
                                    "analyzer": "english",
                                },
                            },
                        },
                        "doc_name": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "chunk_index": {"type": "integer"},
                        "metadata": {"type": "object", "enabled": False},
                    },
                },
            },
        )
        logger.info(f"Created Elasticsearch index: {INDEX_NAME}")
    
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> bool:
        """
        Bulk index documents with automatic chunking.
        
        Args:
            ids: Document IDs
            documents: Document texts
            metadatas: Metadata dicts
            
        Returns:
            True if successful
        """
        if not self._available:
            logger.warning("Elasticsearch not available - skipping indexing")
            return False
        
        operations = []
        
        for i, doc_id in enumerate(ids):
            meta = metadatas[i]
            
            operations.append({
                "index": {"_index": INDEX_NAME, "_id": doc_id}
            })
            operations.append({
                "document": documents[i],
                "doc_name": meta.get("doc_name"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "metadata": meta,
            })
        
        # Bulk index with retry
        for attempt in range(_MAX_RETRIES):
            try:
                # Split into chunks if too large
                for j in range(0, len(operations), _BULK_CHUNK_SIZE * 2):
                    chunk = operations[j:j + _BULK_CHUNK_SIZE * 2]
                    self._client.bulk(operations=chunk, refresh=False)
                
                self._client.indices.refresh(index=INDEX_NAME)
                logger.info(f"Indexed {len(ids)} documents in Elasticsearch")
                return True
                
            except es_exceptions.ConnectionError as e:
                logger.warning(f"ES connection failed (attempt {attempt + 1}): {e}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"ES indexing failed: {e}")
                break
        
        return False
    
    def delete_by_doc(self, doc_name: str) -> bool:
        """Delete all chunks for a document."""
        if not self._available:
            return False
        
        try:
            self._client.delete_by_query(
                index=INDEX_NAME,
                body={
                    "query": {
                        "term": {"doc_name": doc_name}
                    }
                },
                refresh=True,
            )
            logger.info(f"Deleted chunks for: {doc_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from ES: {e}")
            return False
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        BM25 search with configurable fields.
        
        Args:
            query: Search query
            top_k: Number of results
            fields: Fields to search (defaults to ["document"])
            
        Returns:
            List of results with id, document, metadata, bm25_score
        """
        if not self._available:
            return []
        
        search_fields = fields or ["document", "document.english"]
        
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": search_fields,
                    "type": "best_fields",
                }
            },
        }
        
        try:
            response = self._client.search(index=INDEX_NAME, body=body)
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append({
                    "id": hit["_id"],
                    "document": source["document"],
                    "metadata": source["metadata"],
                    "bm25_score": hit["_score"],
                })
            
            logger.debug(f"ES search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"ES search failed: {e}")
            return []
    
    def count(self) -> int:
        """Get total document count."""
        if not self._available:
            return 0
        
        try:
            stats = self._client.count(index=INDEX_NAME)
            return stats["count"]
        except Exception:
            return 0


# Module-level singleton
_store: Optional[ESStore] = None


def get_es_store() -> ESStore:
    """Get or create the ES store singleton."""
    global _store
    if _store is None:
        _store = ESStore()
    return _store


# ---------------------------------------------------------------------------
# Compatibility functions (maintain existing API)
# ---------------------------------------------------------------------------

def create_index():
    """Create index (compatibility wrapper)."""
    get_es_store()


def add_documents(ids, documents, metadatas):
    """Add documents (compatibility wrapper)."""
    return get_es_store().add_documents(ids, documents, metadatas)


def delete_by_doc(doc_name):
    """Delete by doc (compatibility wrapper)."""
    return get_es_store().delete_by_doc(doc_name)


def bm25_search(query, top_k=10, fields=None):
    """BM25 search (compatibility wrapper)."""
    return get_es_store().search(query, top_k, fields)