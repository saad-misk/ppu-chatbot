"""
RAG Pipeline — coordinates retrieval, reranking, truncation, and generation.

Orchestrates the complete flow:
  1. Hybrid retrieval (BM25 + embeddings)
  2. Score-based reranking
  3. Context window management
  4. Answer generation
  5. Source attribution

Usage
-----
    from nlp_engine.rag.pipeline import RAGPipeline
    
    pipeline = RAGPipeline()
    result = pipeline.query("ما هي رسوم CS401؟")
    # {
    #   "answer": "رسوم مساق CS401 هي...",
    #   "sources": [{"document": "fees.pdf", "page": "12", "relevance": 0.85}],
    #   "confidence": 0.78,
    #   "context_used": 3,
    # }
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from nlp_engine.rag.hybrid_retriever import hybrid_retrieve
from nlp_engine.rag.retriever import retrieve as embed_retrieve
from nlp_engine.rag.reranker import rerank
from nlp_engine.rag.generator import (
    generate as _generate_raw,
    _truncate_context_for_prompt,
)
from shared.utils.lang import is_arabic as _is_arabic

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_TOP_K = 10        # Initial retrieval
_DEFAULT_RERANK_K = 5      # After reranking
_DEFAULT_THRESHOLD = 0.25  # Minimum similarity


class RAGPipeline:
    """
    Complete RAG pipeline: retrieve → rerank → truncate → generate.
    
    Args:
        top_k: Initial retrieval count
        rerank_k: Chunks after reranking
        threshold: Minimum similarity threshold
        max_context_chars: Maximum context size for LLM
        use_hybrid: Whether to use hybrid retrieval (BM25 + embeddings)
    """
    
    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        rerank_k: int = _DEFAULT_RERANK_K,
        threshold: float = _DEFAULT_THRESHOLD,
        max_context_chars: int = 4000,
        use_hybrid: bool = True,
    ):
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.threshold = threshold
        self.max_context_chars = max_context_chars
        self.use_hybrid = use_hybrid
        
        logger.info(
            f"RAG Pipeline initialized: hybrid={use_hybrid}, "
            f"top_k={top_k}, rerank_k={rerank_k}, threshold={threshold}"
        )
    
    def query(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        metadata_filter: Optional[Dict] = None,
        add_attribution: bool = True,
        **generator_kwargs,
    ) -> Dict:
        """
        Complete RAG query.
        
        Args:
            query: User question
            history: Conversation history from ContextManager
            metadata_filter: Optional filter (e.g., {"department": "CS"})
            add_attribution: Add source citations to response
            **generator_kwargs: Passed to generator (temperature, max_new_tokens)
            
        Returns:
            Dict with:
                - answer: Generated text
                - sources: List of cited documents
                - confidence: Average retrieval score
                - context_used: Number of chunks provided to LLM
        """
        # 1. Retrieve
        chunks = self._retrieve(query, metadata_filter)
        
        if not chunks:
            return self._no_context_result(query)
        
        # 2. Rerank
        chunks = rerank(chunks, top_k=self.rerank_k, threshold=self.threshold)
        
        if not chunks:
            return self._no_context_result(query)
        
        # 3. Truncate context
        chunks = _truncate_context_for_prompt(chunks, self.max_context_chars)
        
        # 4. Generate
        answer = _generate_raw(
            query=query,
            context_chunks=chunks,
            history=history or [],
            **generator_kwargs,
        )
        
        # 5. Add attribution
        sources = []
        if add_attribution:
            answer, sources = self._add_attribution(answer, chunks)
        
        # 6. Calculate confidence
        confidence = (
            sum(c.get("score", 0) for c in chunks) / len(chunks)
            if chunks else 0.0
        )
        
        logger.info(
            f"RAG query complete: "
            f"retrieved={len(chunks)}, confidence={confidence:.2f}"
        )
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 3),
            "context_used": len(chunks),
        }
    
    def _retrieve(
        self, 
        query: str, 
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """Retrieve using hybrid or embedding-only search."""
        if self.use_hybrid:
            return hybrid_retrieve(
                query=query,
                n_results=self.top_k
                )
        else:
            return embed_retrieve(
                query=query,
                n_results=self.top_k,
                doc_filter=metadata_filter.get("doc_name") if metadata_filter else None,
            )
    
    def _add_attribution(
        self, 
        answer: str, 
        chunks: List[Dict],
    ) -> tuple[str, List[Dict]]:
        """Add source citations."""
        arabic = _is_arabic(answer)
        
        seen = set()
        sources = []
        
        for chunk in chunks:
            score = chunk.get("score", 0)
            if score < 0.3:
                continue
            
            meta = chunk.get("metadata", {})
            doc = meta.get("doc_name", "Unknown")
            page = meta.get("page", "")
            
            key = f"{doc}:{page}" if page else doc
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": doc,
                    "page": page if page else None,
                    "relevance": round(score, 2),
                })
        
        if sources:
            if arabic:
                answer += "\n\n---\n*المصادر:* "
            else:
                answer += "\n\n---\n*Sources:* "
            
            answer += " • ".join(
                f"{s['document']}" + 
                (f" (ص. {s['page']})" if s['page'] and arabic else
                 f" (p. {s['page']})" if s['page'] else "")
                for s in sources
            )
        
        return answer, sources
    
    def _no_context_result(self, query: str) -> Dict:
        """Return when no context found."""
        arabic = _is_arabic(query)
        
        if arabic:
            answer = (
                "عذراً، لم أتمكن من العثور على معلومات ذات صلة في قاعدة المعرفة. "
                "يرجى إعادة صياغة السؤال أو التواصل مع القسم المختص."
            )
        else:
            answer = (
                "Sorry, I couldn't find relevant information in the knowledge base. "
                "Please rephrase your question or contact the relevant department."
            )
        
        return {
            "answer": answer,
            "sources": [],
            "confidence": 0.0,
            "context_used": 0,
        }


# Module singleton
_pipeline: Optional[RAGPipeline] = None


def get_pipeline(**kwargs) -> RAGPipeline:
    """Get or create the RAG pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(**kwargs)
    return _pipeline


def query_rag(
    query: str,
    history: Optional[List[Dict]] = None,
    **kwargs,
) -> Dict:
    """
    Convenience function for complete RAG query.
    
    Usage:
        result = query_rag("What are the CS fees?")
        print(result["answer"])
    """
    pipeline = get_pipeline()
    return pipeline.query(query, history=history, **kwargs)