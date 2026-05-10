"""
RAG (Retrieval-Augmented Generation) module.

Components:
  - retriever: ChromaDB embedding retrieval
  - hybrid_retriever: BM25 + embedding hybrid retrieval
  - reranker: Score-based chunk reranking
  - generator: LLM answer generation
  - pipeline: Complete RAG orchestration
"""
from nlp_engine.rag.retriever import retrieve
from nlp_engine.rag.hybrid_retriever import hybrid_retrieve
from nlp_engine.rag.reranker import rerank
from nlp_engine.rag.generator import generate
from nlp_engine.rag.pipeline import RAGPipeline, get_pipeline, query_rag

__all__ = [
    "retrieve",
    "hybrid_retrieve", 
    "rerank",
    "generate",
    "RAGPipeline",
    "get_pipeline",
    "query_rag",
]