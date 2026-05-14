"""
scripts/ingest_ppu_data.py  —  v3
==================================
Ingest all scraped PPU data into ChromaDB + Elasticsearch.

Source priority (highest quality first):
  1. _chunks.jsonl  — pre-chunked by crawler, richest metadata
  2. Scraped .txt   — fallback if JSONL not present or --txt-only
  3. Raw PDFs       — original university PDF guides

Run from project root:
    python scripts/ingest_ppu_data.py

Options:
    --jsonl-only    Only ingest from _chunks.jsonl
    --txt-only      Only ingest .txt files
    --pdf-only      Only ingest raw PDFs
    --dry-run       Count files/chunks without writing
    --reset         Wipe collection before ingesting
    --batch-size N  ChromaDB upsert batch size (default: 100)
    --lang LANG     Only ingest chunks where lang=LANG (ar | en)
    --category CAT  Only ingest chunks where category=CAT
    --skip-scanned  Skip PDF-sourced chunks flagged as likely scanned
    --force         Re-ingest even if doc already indexed (overwrites)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nlp_engine.knowledge_base.chroma_store import get_store
from nlp_engine.knowledge_base.es_store import create_index, add_documents
from nlp_engine.knowledge_base.chunker import chunk_document
from nlp_engine.knowledge_base.embed import get_embedder
from nlp_engine.knowledge_base.ingest import ingest_pdf, _is_quality_chunk

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "scripts" / "ingest.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
TXT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ppu_rag_data"
JSONL_FILE   = TXT_DATA_DIR / "_chunks.jsonl"
METADATA_DIR = TXT_DATA_DIR / "_metadata"
RAW_PDF_DIR  = PROJECT_ROOT / "data" / "raw"

# ── Ingestion config ──────────────────────────────────────────────────────────
CHUNK_SIZE      = 700
CHUNK_OVERLAP   = 120
CHUNK_STRATEGY  = "contextual"

# FIX: lowered from 100 to 40 — small but useful pages (staff profiles,
# contact info, short policy items) must not be silently dropped.
# The quality gate still rejects pure-noise chunks via alpha-ratio check.
MIN_CHUNK_CHARS = 40

# ── Category priority ─────────────────────────────────────────────────────────
# "administration" and "contact" added to high-value:
#   admin pages carry department info, contact pages carry staff details —
#   both are frequently queried in a university chatbot.
HIGH_VALUE_CATEGORIES = {
    "academics", "admissions", "about",
    "administration", "contact", "student_life",
    "documents",    # course catalogs, regulations, handbooks
    "it_services",
}
LOW_VALUE_CATEGORIES = {
    "news_events",  # dated, not reusable
    "community",    # community outreach, rarely queried
    "jobs",         # individual postings go stale quickly
}
# "general", "research", "library" -> "normal" priority

# ── Folders/files to skip inside ppu_rag_data ─────────────────────────────────
SKIP_FOLDERS = {"_metadata", "_pdfs", "_chunks.jsonl",
                "_progress.json", "_scraper.log"}


# ══════════════════════════════════════════════════════════════════════════════
# Arabic utilities
# ══════════════════════════════════════════════════════════════════════════════

def _is_reversed_arabic(text: str) -> bool:
    """Detect visual-order (reversed) Arabic — presentation forms are the signal."""
    ar_chars   = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    pres_forms = sum(1 for c in text if "\uFE70" <= c <= "\uFEFF")
    if ar_chars == 0:
        return False
    return (pres_forms / ar_chars) > 0.30


def _fix_arabic(text: str) -> str:
    """
    Fix reversed/presentation-form Arabic.
    Returns empty string if libraries missing (chunk will be dropped).
    """
    if not _is_reversed_arabic(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        log.warning(
            "arabic-reshaper/python-bidi not installed — "
            "reversed Arabic chunk dropped. "
            "Fix: pip install arabic-reshaper python-bidi"
        )
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_id(doc_name: str, chunk_idx: int) -> str:
    return hashlib.md5(f"{doc_name}:c{chunk_idx}".encode()).hexdigest()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _priority(category: str) -> str:
    if category in HIGH_VALUE_CATEGORIES:
        return "high"
    if category in LOW_VALUE_CATEGORIES:
        return "low"
    return "normal"


def _passes_filters(
    lang:            str,
    category:        str,
    source:          str,
    filter_lang:     Optional[str],
    filter_category: Optional[str],
    skip_scanned:    bool,
) -> bool:
    # "unknown" lang always passes — can't be sure it's the wrong language
    if filter_lang and lang not in ("unknown", filter_lang):
        return False
    if filter_category and category != filter_category:
        return False
    if skip_scanned and source == "pdf_scanned":
        return False
    return True


def _quality_check(text: str, min_chars: int) -> str:
    """
    Combined quality + Arabic fix. Returns cleaned text or empty string.

    Order matters:
      1. Fix reversed Arabic FIRST so length/alpha checks run on real text
      2. Minimum length check
      3. Alpha ratio check — LOW threshold (25%) so mixed content like
         "Dr. Ahmed Khalil — ext. 3042  قسم الهندسة" survives.
         We want staff profiles, short contact entries, office hour lines
         to all pass — these are genuinely useful for a university chatbot.
    """
    # Step 1: fix Arabic if needed
    text = _fix_arabic(text)
    if not text:
        return ""

    t = text.strip()

    # Step 2: length gate
    if len(t) < min_chars:
        return ""

    # Step 3: alpha ratio — LOW threshold to preserve short useful content
    alpha = sum(1 for c in t if c.isalpha())
    if alpha / max(len(t), 1) < 0.25:
        return ""

    return t


def _flush_batch(
    store,
    batch_ids:   List[str],
    batch_vecs:  List,
    batch_docs:  List[str],
    batch_metas: List[Dict],
    dry_run:     bool,
) -> int:
    """Upsert accumulated batch into ChromaDB + Elasticsearch."""
    if not batch_ids:
        return 0
    count = len(batch_ids)
    if dry_run:
        batch_ids.clear(); batch_vecs.clear()
        batch_docs.clear(); batch_metas.clear()
        return count
    try:
        store.add(
            ids=batch_ids,
            embeddings=batch_vecs,
            documents=batch_docs,
            metadatas=batch_metas,
        )
        add_documents(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
        )
    except Exception as e:
        log.error("Batch upsert failed: %s", e)
        count = 0
    batch_ids.clear(); batch_vecs.clear()
    batch_docs.clear(); batch_metas.clear()
    return count


def _invalidate_bm25():
    try:
        from nlp_engine.rag.hybrid_retriever import invalidate_bm25_cache
        invalidate_bm25_cache()
    except Exception as e:
        log.warning("BM25 cache invalidation failed (%s: %s) — "
                    "restart server to refresh BM25 index.",
                    type(e).__name__, e)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — _chunks.jsonl
# ══════════════════════════════════════════════════════════════════════════════

def _iter_jsonl() -> Iterator[Dict]:
    with open(JSONL_FILE, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Bad JSONL line %d: %s", lineno, e)


def ingest_from_jsonl(
    dry_run:         bool = False,
    batch_size:      int  = 100,
    filter_lang:     Optional[str] = None,
    filter_category: Optional[str] = None,
    skip_scanned:    bool = False,
    force:           bool = False,
) -> Dict:
    if not JSONL_FILE.exists():
        log.warning("_chunks.jsonl not found — skipping JSONL phase.")
        return {"chunks": 0, "skipped": 0, "status": "no_file"}

    store    = get_store()
    embedder = get_embedder()
    create_index()

    already_indexed: set[str] = set(store.list_documents()) if not force else set()
    seen_hashes:     set[str] = set()

    total_chunks = 0
    skipped      = 0
    batch_ids, batch_vecs, batch_docs, batch_metas = [], [], [], []

    for rec in _iter_jsonl():
        url       = rec.get("url", "")
        title     = rec.get("title", "")
        lang      = rec.get("lang", "unknown")
        category  = rec.get("category", "general")
        source    = rec.get("source", "html")
        raw_text  = rec.get("text", "")
        doc_name  = rec.get("id", "").split("-")[0]
        chunk_idx = rec.get("chunk_index", 0)

        # Filter gate
        if not _passes_filters(lang, category, source,
                               filter_lang, filter_category, skip_scanned):
            skipped += 1
            continue

        # Idempotency
        if doc_name and doc_name in already_indexed and not force:
            skipped += 1
            continue

        # Quality + Arabic fix (unified gate)
        text = _quality_check(raw_text, MIN_CHUNK_CHARS)
        if not text:
            skipped += 1
            continue

        # Content dedup
        ch = _content_hash(text)
        if ch in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(ch)

        cid = rec.get("id") or _chunk_id(url, chunk_idx)

        batch_ids.append(cid)
        batch_docs.append(text)
        batch_metas.append({
            "doc_name":    doc_name or _content_hash(url)[:8],
            "url":         url,
            "title":       title[:200],
            "lang":        lang,
            "category":    category,
            "source":      source,
            "chunk_index": chunk_idx,
            "priority":    _priority(category),
            "scraped_at":  rec.get("scraped_at", ""),
        })

        # FIX: removed spurious embedder.embed_one() call that was firing
        # on the first item. Embedding only happens at flush time.
        if len(batch_ids) >= batch_size:
            vecs = embedder.embed(batch_docs) if not dry_run else []
            total_chunks += _flush_batch(
                store, batch_ids, vecs, batch_docs, batch_metas, dry_run
            )

    # Final partial batch
    if batch_ids:
        vecs = embedder.embed(batch_docs) if not dry_run else []
        total_chunks += _flush_batch(
            store, batch_ids, vecs, batch_docs, batch_metas, dry_run
        )

    _invalidate_bm25()
    log.info("JSONL phase done — %d indexed, %d skipped.", total_chunks, skipped)
    return {"chunks": total_chunks, "skipped": skipped,
            "status": "ok" if not dry_run else "dry_run"}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — .txt files (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_txt_header(text: str) -> Tuple[Dict, str]:
    pattern = re.compile(
        r"^URL:\s*(?P<url>.+)\n"
        r"TITLE:\s*(?P<title>.*)\n"
        r"LANGUAGE:\s*(?P<lang>\w+)\n"
        r"CATEGORY:\s*(?P<category>\w+)\n"
        r"(?:SCRAPED_AT:\s*(?P<scraped_at>.+)\n)?"
        r"(?:SOURCE:\s*(?P<source>.+)\n)?"
        r"(?:HEADINGS:\s*(?P<headings>.*)\n)?"
        r"=+\n",
        re.MULTILINE,
    )
    m = pattern.search(text)
    if m:
        meta    = {k: (v or "").strip() for k, v in m.groupdict().items()}
        content = text[m.end():].strip()
        return meta, content
    return {}, text.strip()


def _collect_txt_files(filter_category: Optional[str] = None) -> List[Path]:
    if not TXT_DATA_DIR.exists():
        log.error("TXT data dir not found: %s", TXT_DATA_DIR)
        return []
    files = []
    for path in sorted(TXT_DATA_DIR.rglob("*.txt")):
        rel   = path.relative_to(TXT_DATA_DIR)
        parts = rel.parts
        if any(part in SKIP_FOLDERS for part in parts):
            continue
        category = parts[0] if len(parts) >= 2 else "general"
        if filter_category and category != filter_category:
            continue
        files.append(path)
    log.info("Found %d .txt files.", len(files))
    return files


def ingest_txt_files(
    dry_run:         bool = False,
    batch_size:      int  = 100,
    filter_lang:     Optional[str] = None,
    filter_category: Optional[str] = None,
    skip_scanned:    bool = False,
    force:           bool = False,
) -> Dict:
    files = _collect_txt_files(filter_category)
    if not files:
        return {"files": 0, "chunks": 0, "skipped": 0, "status": "no_files"}

    store    = get_store()
    embedder = get_embedder()
    create_index()

    already_indexed = set(store.list_documents()) if not force else set()
    seen_hashes:    set[str] = set()

    total_chunks  = 0
    total_files   = 0
    skipped_files = 0

    batch_ids, batch_docs, batch_metas = [], [], []

    def flush():
        nonlocal total_chunks
        if not batch_ids:
            return
        vecs = embedder.embed(batch_docs) if not dry_run else []
        total_chunks += _flush_batch(
            store, batch_ids, vecs, batch_docs, batch_metas, dry_run
        )

    for file_path in files:
        doc_name = file_path.name

        if doc_name in already_indexed and not force:
            skipped_files += 1
            continue

        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.warning("Cannot read %s: %s", file_path, e)
            skipped_files += 1
            continue

        meta, content = _parse_txt_header(raw)

        parts    = file_path.relative_to(TXT_DATA_DIR).parts
        category = meta.get("category") or (parts[0] if len(parts) >= 2 else "general")
        lang     = meta.get("lang")     or (parts[1] if len(parts) >= 3 else "unknown")
        url      = meta.get("url", "")
        title    = meta.get("title", doc_name)
        source   = meta.get("source", "html")
        headings = meta.get("headings", "")

        if not _passes_filters(lang, category, source,
                               filter_lang, filter_category, skip_scanned):
            skipped_files += 1
            continue

        # Fix Arabic at document level before chunking
        if _is_reversed_arabic(content):
            content = _fix_arabic(content)
        if not content:
            skipped_files += 1
            continue

        # FIX: short documents (staff pages, contact pages, short policy items)
        # get "structural" strategy instead of "contextual".
        # Reason: contextual requires headings to inject context prefix — short
        # documents often have no headings so contextual produces 0 chunks and
        # the entire document is silently discarded.
        # "structural" just respects paragraph boundaries, which is correct
        # for a 3-line staff profile.
        is_short_doc  = len(content) < 500
        strategy      = "structural" if is_short_doc else CHUNK_STRATEGY
        # Lower quality floor for short docs so a 2-line staff entry
        # ("Dr. X | Computer Science | ext. 302") doesn't get dropped.
        doc_min_chars = 20 if is_short_doc else MIN_CHUNK_CHARS

        raw_chunks = chunk_document(
            content,
            strategy=strategy,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        # Safety net: if chunker still produced nothing for a non-empty doc,
        # treat the whole content as one chunk rather than discarding it.
        if not raw_chunks and len(content.strip()) >= doc_min_chars:
            raw_chunks = [content.strip()]

        good_chunks, dropped = [], 0
        for chunk in raw_chunks:
            cleaned = _quality_check(chunk, doc_min_chars)
            if not cleaned:
                dropped += 1
                continue
            ch = _content_hash(cleaned)
            if ch in seen_hashes:
                dropped += 1
                continue
            seen_hashes.add(ch)
            good_chunks.append(cleaned)

        if not good_chunks:
            log.debug("No chunks produced for: %s (%d chars)", doc_name, len(content))
            skipped_files += 1
            continue

        if dropped:
            log.debug("  Dropped %d/%d chunks from %s",
                      dropped, len(raw_chunks), doc_name)

        log.info("[%s/%s] %s → %d chunk(s)%s",
                 category, lang, doc_name[:55], len(good_chunks),
                 " [short-doc]" if is_short_doc else "")

        priority = _priority(category)
        for idx, chunk in enumerate(good_chunks):
            batch_ids.append(_chunk_id(doc_name, idx))
            batch_docs.append(chunk)
            batch_metas.append({
                "doc_name":     doc_name,
                "url":          url,
                "title":        title[:200],
                "lang":         lang,
                "category":     category,
                "source":       source,
                "chunk_index":  idx,
                "headings":     headings[:400],
                "priority":     priority,
                "is_short_doc": str(is_short_doc),
            })
            if len(batch_ids) >= batch_size:
                flush()

        total_files += 1

    flush()
    _invalidate_bm25()

    log.info("TXT phase done — %d files, %d chunks, %d skipped.",
             total_files, total_chunks, skipped_files)
    return {
        "files":   total_files,
        "chunks":  total_chunks,
        "skipped": skipped_files,
        "status":  "ok" if not dry_run else "dry_run",
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Raw PDFs
# ══════════════════════════════════════════════════════════════════════════════

def ingest_raw_pdfs(dry_run: bool = False, force: bool = False) -> Dict:
    if not RAW_PDF_DIR.exists():
        log.error("Raw PDF dir not found: %s", RAW_PDF_DIR)
        return {"files": 0, "chunks": 0, "skipped": 0, "status": "no_dir"}

    pdfs  = sorted(RAW_PDF_DIR.glob("*.pdf"))
    store = get_store()
    already_indexed = set(store.list_documents())

    total_files = total_chunks = skipped_files = 0
    log.info("Found %d raw PDFs.", len(pdfs))

    for pdf_path in pdfs:
        doc_name = pdf_path.name
        if doc_name in already_indexed and not force:
            log.info("Already indexed, skipping: %s", doc_name)
            skipped_files += 1
            continue

        size_kb = pdf_path.stat().st_size // 1024
        log.info("Ingesting PDF: %s (%d KB)", doc_name, size_kb)

        if dry_run:
            total_files += 1
            continue

        try:
            result = ingest_pdf(
                file_bytes=pdf_path.read_bytes(),
                doc_name=doc_name,
                chunking_strategy=CHUNK_STRATEGY,
                force_reingest=force,
            )
            status = result.get("status", "?")
            if status in ("ok", "already_indexed"):
                total_files  += 1
                total_chunks += result.get("chunks", 0)
                scanned_flag  = "  ⚠ scanned" if result.get("is_scanned") else ""
                log.info("  ✓ %s — %d pages, %d chunks%s",
                         doc_name, result.get("pages", 0),
                         result.get("chunks", 0), scanned_flag)
            else:
                log.warning("  ✗ %s — status: %s", doc_name, status)
                skipped_files += 1
        except Exception as e:
            log.error("Failed to ingest %s: %s", doc_name, e)
            skipped_files += 1

    log.info("PDF phase done — %d files, %d chunks, %d skipped.",
             total_files, total_chunks, skipped_files)
    return {
        "files":   total_files,
        "chunks":  total_chunks,
        "skipped": skipped_files,
        "status":  "ok" if not dry_run else "dry_run",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ingest PPU scraped data into ChromaDB + Elasticsearch"
    )
    parser.add_argument("--jsonl-only",   action="store_true")
    parser.add_argument("--txt-only",     action="store_true")
    parser.add_argument("--pdf-only",     action="store_true")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--reset",        action="store_true")
    parser.add_argument("--batch-size",   type=int, default=100)
    parser.add_argument("--lang",         type=str, default=None)
    parser.add_argument("--category",     type=str, default=None)
    parser.add_argument("--skip-scanned", action="store_true")
    parser.add_argument("--force",        action="store_true")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("PPU Data Ingestion Pipeline  v3")
    log.info("Project root : %s", PROJECT_ROOT)
    log.info("JSONL file   : %s  [%s]",
             JSONL_FILE, "found" if JSONL_FILE.exists() else "missing")
    log.info("Dry run      : %s | Force: %s | Reset: %s",
             args.dry_run, args.force, args.reset)
    log.info("Filters      : lang=%s  category=%s  skip_scanned=%s",
             args.lang, args.category, args.skip_scanned)
    log.info("=" * 65)

    store = get_store()
    create_index()

    if args.reset and not args.dry_run:
        count = store.count()
        log.warning("--reset: deleting all %d chunks...", count)
        store._client.delete_collection("ppu_knowledge")
        store._collection = store._client.get_or_create_collection(
            name="ppu_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        log.info("Collection reset complete.")

    results = {}
    kwargs  = dict(
        dry_run=args.dry_run,
        filter_lang=args.lang,
        filter_category=args.category,
        skip_scanned=args.skip_scanned,
        force=args.force,
    )

    if not args.txt_only and not args.pdf_only:
        if JSONL_FILE.exists():
            log.info("\n  Phase 1: _chunks.jsonl")
            results["jsonl"] = ingest_from_jsonl(
                batch_size=args.batch_size, **kwargs)
        else:
            log.info("\n  Phase 1: no JSONL found - falling back to .txt")
            args.txt_only = True

    if args.txt_only or (not args.jsonl_only and not args.pdf_only
                         and not JSONL_FILE.exists()):
        log.info("\n  Phase 2: .txt files")
        results["txt"] = ingest_txt_files(
            batch_size=args.batch_size, **kwargs)

    if not args.jsonl_only and not args.txt_only:
        log.info("\n  Phase 3: raw PDFs")
        results["pdf"] = ingest_raw_pdfs(
            dry_run=args.dry_run, force=args.force)

    log.info("\n" + "=" * 65)
    log.info("INGESTION SUMMARY")
    log.info("=" * 65)
    total_new = 0
    for source, r in results.items():
        log.info("  %-8s -> %5d chunks | %4d skipped | %s",
                 source.upper(),
                 r.get("chunks", 0),
                 r.get("skipped", 0),
                 r.get("status", "?"))
        total_new += r.get("chunks", 0)

    if not args.dry_run:
        log.info("  ChromaDB total : %d documents", store.count())
    log.info("  New this run   : %d chunks", total_new)
    log.info("=" * 65)


if __name__ == "__main__":
    main()