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

CHUNK_SIZE    = 400   # characters per chunk
CHUNK_OVERLAP = 80    # overlapping characters between adjacent chunks

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

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split *text* into overlapping character-level chunks.
    Tries to break at sentence boundaries ('. ') when possible.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence boundary within the last 20% of the chunk
        if end < len(text):
            window_start = int(chunk_size * 0.8)
            window = chunk[window_start:]
            matches = list(_SENTENCE_BOUNDARY_RE.finditer(window))
            if matches:
                last = matches[-1].start() + window_start
                end = start + last + 1  # include the boundary punctuation
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_pdf(
    file_bytes: bytes,
    doc_name: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Dict:
    """
    Full ingestion pipeline: PDF bytes → chunks → embeddings → ChromaDB.

    Parameters
    ----------
    file_bytes  : raw PDF bytes
    doc_name    : logical document name (used in metadata and for deletion)
    chunk_size  : characters per chunk
    overlap     : overlap between chunks

    Returns
    -------
    dict with keys: doc_name, pages, chunks, status
    """
    logger.info("Starting ingestion for: %s (%d bytes)", doc_name, len(file_bytes))

    # 1. Extract pages
    pages = _extract_pages(file_bytes)
    if not pages:
        logger.warning("No text extracted from %s", doc_name)
        return {"doc_name": doc_name, "pages": 0, "chunks": 0, "status": "empty"}

    # 2. Remove old chunks for this document
    store = get_store()
    store.delete_by_doc(doc_name)

    # 3. Build chunks with metadata
    embedder = get_embedder()
    all_ids, all_embeddings, all_documents, all_metadatas = [], [], [], []
    total_chunks = 0

    for page_info in pages:
        page_num = page_info["page"]
        page_text = page_info["text"]
        chunks = _chunk_text(page_text, chunk_size, overlap)

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
