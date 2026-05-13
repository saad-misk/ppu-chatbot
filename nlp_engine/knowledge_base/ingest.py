"""
nlp_engine/knowledge_base/ingest.py
====================================
PDF ingestion pipeline — v2
Reads a PDF (file path or raw bytes), extracts text across full document
(not page-by-page), chunks with the contextual strategy, embeds, and
dual-indexes into ChromaDB + Elasticsearch.

Key fixes over v1
-----------------
  1. Full-document chunking  — pages concatenated before chunking so
     cross-page paragraphs stay intact.
  2. Contextual strategy     — heading-injected chunks by default.
  3. Per-chunk quality gate  — short / non-content chunks dropped.
  4. Rich metadata           — lang, category, source, url on every chunk.
  5. Idempotency guard       — skips already-indexed docs (no silent delete).
  6. Scanned PDF detection   — warns when OCR yield is too low.
  7. Language detection      — simple heuristic, no extra dependency.

Usage
-----
    from nlp_engine.knowledge_base.ingest import ingest_pdf, ingest_file
    ingest_pdf(file_bytes=pdf_bytes, doc_name="student_handbook.pdf")
"""
from __future__ import annotations

import hashlib
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber

from nlp_engine.knowledge_base.chunker import chunk_document
from nlp_engine.knowledge_base.embed import get_embedder
from nlp_engine.knowledge_base.chroma_store import get_store
from nlp_engine.knowledge_base.es_store import (
    create_index,
    add_documents,
    delete_by_doc as es_delete_by_doc,
)

logger = logging.getLogger(__name__)

# ── Chunking defaults ─────────────────────────────────────────────────────────
CHUNK_SIZE      = 700     # target chars per chunk
CHUNK_OVERLAP   = 120     # overlap for char strategy (unused by contextual)
CHUNK_STRATEGY  = "contextual"   # FIXED: was "semantic"
MIN_CHUNK_CHARS = 100     # per-chunk quality floor (post-chunking gate)
MIN_PDF_YIELD   = 0.05    # chars-per-byte floor — below = likely scanned

# ── Heuristic category keywords (mirrors crawler logic) ──────────────────────
_CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    ("admissions",     ["admis", "registr", "apply", "tuition", "fees",
                        "scholarship", "financial"]),
    ("academics",      ["college", "faculty", "depart", "program", "course",
                        "curriculum", "syllabus", "degree", "catalog"]),
    ("research",       ["research", "publication", "journal", "thesis",
                        "dissertation", "graduate"]),
    ("student_life",   ["student", "club", "activit", "housing", "counseling",
                        "career", "alumni"]),
    ("administration", ["admin", "president", "board", "council", "senate",
                        "procurement", "human-resources", "financial"]),
    ("library",        ["library"]),
    ("jobs",           ["jobs", "careers", "vacancies"]),
    ("about",          ["about", "history", "vision", "mission", "structure"]),
    ("documents",      ["handbook", "guide", "regulation", "bylaw", "policy"]),
]


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _detect_lang(text: str) -> str:
    """
    Heuristic language detection — no external dependency.
    Counts Arabic Unicode characters in the first 400 chars.
    """
    sample = text[:400]
    ar = sum(1 for c in sample if "\u0600" <= c <= "\u06ff")
    return "ar" if ar > 20 else "en"


def _detect_category(doc_name: str, text_sample: str = "") -> str:
    """Infer category from filename + first 300 chars of text."""
    haystack = (doc_name + " " + text_sample[:300]).lower()
    for cat, keywords in _CATEGORY_RULES:
        if any(kw in haystack for kw in keywords):
            return cat
    return "general"


def _make_chunk_id(doc_name: str, chunk_idx: int) -> str:
    """Deterministic, collision-resistant chunk ID."""
    return hashlib.md5(f"{doc_name}:c{chunk_idx}".encode()).hexdigest()


def _is_quality_chunk(text: str, min_chars: int = MIN_CHUNK_CHARS) -> bool:
    """
    Drop chunks that are too short, purely numeric/symbolic, or
    consist only of nav/noise content.
    """
    t = text.strip()
    if len(t) < min_chars:
        return False
    # Reject if >60 % of chars are non-alphabetic (garbled OCR, tables of numbers)
    alpha = sum(1 for c in t if c.isalpha())
    if alpha / max(len(t), 1) < 0.40:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Arabic text fixing
# ══════════════════════════════════════════════════════════════════════════════

def _is_reversed_arabic(text: str) -> bool:
    """
    Detect Arabic text stored in visual/reversed order.
    Presentation-form characters (FE70–FEFF) are the reliable signal —
    they appear when PDF software stored glyphs in display order instead
    of logical Unicode order.
    """
    ar_chars   = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    pres_forms = sum(1 for c in text if "\uFE70" <= c <= "\uFEFF")
    if ar_chars == 0:
        return False
    return (pres_forms / ar_chars) > 0.30


def _fix_arabic(text: str) -> str:
    """
    Fix reversed/presentation-form Arabic extracted from legacy PDFs.
    Requires:  pip install arabic-reshaper python-bidi
    Falls back gracefully if libraries are missing.
    """
    if not _is_reversed_arabic(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except ImportError:
        logger.warning(
            "arabic-reshaper / python-bidi not installed — "
            "reversed Arabic chunks will be filtered out. "
            "Run: pip install arabic-reshaper python-bidi"
        )
        return ""   # empty → caught by quality gate → dropped


# ══════════════════════════════════════════════════════════════════════════════
# Text extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_full_text(pdf_bytes: bytes) -> Tuple[str, int, bool]:
    """
    Extract all text from a PDF as a single string using pdfplumber
    (better RTL/Arabic handling than pypdf).
    Pages are joined with double-newline so cross-page paragraphs
    remain intact for the contextual chunker.

    Returns
    -------
    (full_text, page_count, is_likely_scanned)
    """
    page_texts: List[str] = []
    page_count = 0

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            cleaned = re.sub(r"[ \t]+", " ", raw).strip()
            if not cleaned:
                continue
            # Fix reversed Arabic before anything else
            cleaned = _fix_arabic(cleaned)
            if cleaned:
                page_texts.append(cleaned)

    full_text   = "\n\n".join(page_texts)
    yield_ratio = len(full_text) / max(len(pdf_bytes), 1)
    is_scanned  = yield_ratio < MIN_PDF_YIELD

    return full_text, page_count, is_scanned


# ══════════════════════════════════════════════════════════════════════════════
# Main ingestion function
# ══════════════════════════════════════════════════════════════════════════════

def ingest_pdf(
    file_bytes: bytes,
    doc_name:   str,
    chunk_size: int  = CHUNK_SIZE,
    overlap:    int  = CHUNK_OVERLAP,
    chunking_strategy: str = CHUNK_STRATEGY,
    url:        str  = "",
    category:   str  = "",
    force_reingest: bool = False,
) -> Dict:
    """
    Full ingestion pipeline: PDF bytes → full text → chunks → embeddings
    → ChromaDB + Elasticsearch.

    Parameters
    ----------
    file_bytes        : Raw PDF bytes
    doc_name          : Unique document identifier (filename)
    chunk_size        : Target chars per chunk
    overlap           : Overlap for char-based strategy
    chunking_strategy : 'contextual' (default) | 'structural' | 'semantic' |
                        'char' | 'sentence'
    url               : Source URL (optional, enriches metadata)
    category          : Override auto-detected category
    force_reingest    : If True, delete existing chunks and re-index

    Returns
    -------
    dict — doc_name, pages, chunks, skipped_chunks, status, is_scanned
    """
    logger.info("ingest_pdf: %s (%d KB)", doc_name, len(file_bytes) // 1024)
    create_index()

    store = get_store()

    # ── Idempotency guard ─────────────────────────────────────────────────
    already_indexed = doc_name in store.list_documents()
    if already_indexed and not force_reingest:
        logger.info("Already indexed, skipping: %s", doc_name)
        return {
            "doc_name": doc_name,
            "pages":    0,
            "chunks":   0,
            "skipped_chunks": 0,
            "status":   "already_indexed",
            "is_scanned": False,
        }

    if already_indexed and force_reingest:
        logger.info("force_reingest=True — removing old chunks for: %s", doc_name)
        store.delete_by_doc(doc_name)
        es_delete_by_doc(doc_name)

    # ── Extract ───────────────────────────────────────────────────────────
    full_text, page_count, is_scanned = _extract_full_text(file_bytes)

    if not full_text.strip():
        logger.warning("No text extracted from %s — possibly image-only PDF", doc_name)
        return {
            "doc_name": doc_name, "pages": page_count,
            "chunks": 0, "skipped_chunks": 0,
            "status": "empty", "is_scanned": True,
        }

    if is_scanned:
        logger.warning(
            "Low text yield in %s (%.3f chars/byte) — likely scanned. "
            "Consider OCR pre-processing. Ingesting anyway.",
            doc_name, len(full_text) / max(len(file_bytes), 1),
        )

    # ── Detect language + category ────────────────────────────────────────
    lang     = _detect_lang(full_text)
    category = category or _detect_category(doc_name, full_text[:300])

    # ── Chunk full document (not per-page) ────────────────────────────────
    raw_chunks = chunk_document(
        full_text,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # Per-chunk quality gate
    good_chunks    = [c for c in raw_chunks if _is_quality_chunk(c)]
    skipped_chunks = len(raw_chunks) - len(good_chunks)

    if skipped_chunks:
        logger.info(
            "  Quality gate: dropped %d/%d chunks from %s",
            skipped_chunks, len(raw_chunks), doc_name,
        )

    if not good_chunks:
        logger.warning("No quality chunks produced for: %s", doc_name)
        return {
            "doc_name": doc_name, "pages": page_count,
            "chunks": 0, "skipped_chunks": skipped_chunks,
            "status": "no_quality_chunks", "is_scanned": is_scanned,
        }

    # ── Embed all chunks in one batch ─────────────────────────────────────
    logger.info("  Embedding %d chunks (strategy=%s, lang=%s, cat=%s)…",
                len(good_chunks), chunking_strategy, lang, category)
    embedder   = get_embedder()
    embeddings = embedder.embed(good_chunks)

    # ── Build metadata records ────────────────────────────────────────────
    ids, docs, metas = [], [], []
    for idx, (chunk, vector) in enumerate(zip(good_chunks, embeddings)):
        ids.append(_make_chunk_id(doc_name, idx))
        docs.append(chunk)
        metas.append({
            "doc_name":    doc_name,
            "chunk_index": idx,
            "total_chunks": len(good_chunks),
            "lang":        lang,
            "category":    category,
            "source":      "pdf_scanned" if is_scanned else "pdf",
            "url":         url,
            "is_scanned":  str(is_scanned),  # ChromaDB requires str/int/float
        })

    # ── Upsert into both stores ───────────────────────────────────────────
    store.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    add_documents(ids=ids, documents=docs, metadatas=metas)

    # ── Invalidate BM25 cache ─────────────────────────────────────────────
    try:
        from nlp_engine.rag.hybrid_retriever import invalidate_bm25_cache
        invalidate_bm25_cache()
    except Exception as e:
        logger.warning("BM25 cache invalidation failed (%s: %s) — "
                       "queries may use stale index.", type(e).__name__, e)

    logger.info(
        "ingest_pdf done: '%s' — %d pages, %d chunks indexed, %d dropped.",
        doc_name, page_count, len(good_chunks), skipped_chunks,
    )
    return {
        "doc_name":      doc_name,
        "pages":         page_count,
        "chunks":        len(good_chunks),
        "skipped_chunks": skipped_chunks,
        "status":        "ok",
        "is_scanned":    is_scanned,
    }


def ingest_file(file_path: str | Path, **kwargs) -> Dict:
    """Convenience wrapper: read a PDF from disk and ingest it."""
    path = Path(file_path)
    return ingest_pdf(
        file_bytes=path.read_bytes(),
        doc_name=path.name,
        **kwargs,
    )