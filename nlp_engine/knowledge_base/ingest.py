"""
PDF ingestion pipeline.

Reads a PDF (from a file path or raw bytes), splits it into overlapping
text chunks, generates embeddings, and indexes them in ChromaDB.

Usage
-----
    from nlp_engine.knowledge_base.ingest import ingest_pdf

    ingest_pdf(file_bytes=pdf_bytes, doc_name="student_handbook.pdf")
"""
from __future__ import annotations

from nlp_engine.knowledge_base.es_store import (
    create_index,
    add_documents,
    delete_by_doc as es_delete_by_doc,
)

from nlp_engine.knowledge_base.chunker import chunk_document


import hashlib
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional

from pypdf import PdfReader

from nlp_engine.knowledge_base.embed import get_embedder
from nlp_engine.knowledge_base.chroma_store import get_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE    = 700   # characters per chunk
CHUNK_OVERLAP = 120    # overlapping characters between adjacent chunks

# Sentence boundary punctuation (English + Arabic)
_SENTENCE_BOUNDARY_RE = re.compile(r"[\.\!\?؟؛]")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_pages(pdf_bytes: bytes) -> List[Dict]:
    """
    Extract text from each page of a PDF.

    Returns
    -------
    List[dict] — [{"page": int, "text": str}, ...]
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", raw).strip()
        if cleaned:
            pages.append({"page": page_num, "text": cleaned})
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    strategy: str = "semantic",
) -> List[str]:
    """
    Split text into chunks using specified strategy.
    
    Args:
        text: Input text
        chunk_size: Target chunk size
        overlap: Overlap between chunks (char strategy only)
        strategy: 'char', 'sentence', or 'semantic'
        
    Returns:
        List of text chunks
    """
    return chunk_document(text, strategy=strategy, chunk_size=chunk_size, overlap=overlap)




# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_pdf(
    file_bytes: bytes,
    doc_name: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    chunking_strategy: str = "semantic",
) -> Dict:
    """
    Full ingestion pipeline: PDF bytes → chunks → embeddings → ChromaDB.

    Args:
        file_bytes: Raw PDF bytes
        doc_name: Document name
        chunk_size: Characters per chunk
        overlap: Overlap between chunks
        chunking_strategy: 'char', 'sentence', or 'semantic'

    Returns
    dict with keys: doc_name, pages, chunks, status
    """
    logger.info("Starting ingestion for: %s (%d bytes)", doc_name, len(file_bytes))

    create_index()

    # 1. Extract pages
    pages = _extract_pages(file_bytes)
    if not pages:
        logger.warning("No text extracted from %s", doc_name)
        return {"doc_name": doc_name, "pages": 0, "chunks": 0, "status": "empty"}

    # 2. Remove old chunks for this document
    store = get_store()
    store.delete_by_doc(doc_name)
    es_delete_by_doc(doc_name)

    # 3. Build chunks with metadata
    embedder = get_embedder()
    all_ids, all_embeddings, all_documents, all_metadatas = [], [], [], []
    total_chunks = 0

    for page_info in pages:
        page_num = page_info["page"]
        page_text = page_info["text"]
        chunks = _chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=chunking_strategy,
        )

        for chunk_idx, chunk in enumerate(chunks):
            # Deterministic ID: hash of (doc_name + page + chunk_idx)
            chunk_id = hashlib.md5(f"{doc_name}:p{page_num}:c{chunk_idx}".encode()).hexdigest()
            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_metadatas.append(
                {
                    "doc_name":    doc_name,
                    "page":        page_num,
                    "chunk_index": chunk_idx,
                }
            )
            total_chunks += 1

    # 4. Embed in one batch
    logger.info("Embedding %d chunks…", total_chunks)
    all_embeddings = embedder.embed(all_documents)

    # 5. Upsert into ChromaDB
    store.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_documents,
        metadatas=all_metadatas,
    )

    add_documents(
        ids=all_ids,
        documents=all_documents,
        metadatas=all_metadatas,
    )

    # Invalidate BM25 cache after ingestion so hybrid retrieval sees new docs
    try:
        from nlp_engine.rag.hybrid_retriever import invalidate_bm25_cache
        invalidate_bm25_cache()
    except Exception:
        logger.warning("Failed to invalidate BM25 cache after ingestion.")

    logger.info(
        "Ingestion complete for '%s': %d pages → %d chunks indexed.",
        doc_name,
        len(pages),
        total_chunks,
    )
    return {
        "doc_name": doc_name,
        "pages":    len(pages),
        "chunks":   total_chunks,
        "status":   "ok",
    }


def ingest_file(file_path: str | Path) -> Dict:
    """Convenience wrapper: read a PDF from disk and ingest it."""
    path = Path(file_path)
    return ingest_pdf(file_bytes=path.read_bytes(), doc_name=path.name)
