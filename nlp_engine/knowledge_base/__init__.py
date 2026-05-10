"""
Knowledge Base module for PPU chatbot.

Components:
  • chroma_store: ChromaDB vector store
  • es_store: Elasticsearch BM25 store
  • embed: Multilingual embedding model
  • chunker: Text chunking strategies
  • cache: Embedding cache
  • ingest: PDF ingestion pipeline
"""
from nlp_engine.knowledge_base.chroma_store import ChromaStore, get_store
from nlp_engine.knowledge_base.es_store import ESStore, get_es_store, bm25_search
from nlp_engine.knowledge_base.embed import Embedder, get_embedder
from nlp_engine.knowledge_base.chunker import chunk_document, chunk_semantic
from nlp_engine.knowledge_base.cache import EmbeddingCache, get_cache
from nlp_engine.knowledge_base.ingest import ingest_pdf, ingest_file

__all__ = [
    "ChromaStore",
    "get_store",
    "ESStore",
    "get_es_store",
    "bm25_search",
    "Embedder",
    "get_embedder",
    "chunk_document",
    "chunk_semantic",
    "EmbeddingCache",
    "get_cache",
    "ingest_pdf",
    "ingest_file",
]