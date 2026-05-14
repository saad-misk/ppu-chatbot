"""
scripts/sync_chunks_to_files.py
================================
Remove chunks from _chunks.jsonl whose source files no longer exist.

Works by checking two things per chunk:
  1. If source is "pdf"  — the corresponding .pdf must exist in _pdfs/
  2. If source is "html" — the corresponding .txt must exist in ppu_rag_data/
  3. URL-based removal   — drop any chunk whose URL matches a deleted domain
                           or pattern you specify in REMOVE_URL_PATTERNS below.

Run:
    python scripts/sync_chunks_to_files.py

Add --dry-run to preview without writing.
"""
import argparse
import json
import re
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter
from urllib.parse import urlparse

# ── Paths ─────────────────────────────────────────────────────
JSONL_PATH   = Path("data/raw/ppu_rag_data/_chunks.jsonl")
TXT_DATA_DIR = Path("data/raw/ppu_rag_data")
PDF_DIR      = Path("data/raw/ppu_rag_data/_pdfs")
RAW_PDF_DIR  = Path("data/raw")

# ── Manual removal rules (edit as needed) ─────────────────────
# Any chunk whose URL matches one of these patterns will be removed.
# Add patterns for domains/paths you deleted manually.
REMOVE_URL_PATTERNS = [
    # r"library\.ppu\.edu",      # uncomment to remove all library chunks
    # r"download\.ppu\.edu",
    # r"/news/",
    # r"/jobs/\d+",
]

# ── Helpers ───────────────────────────────────────────────────
def make_txt_filename(url: str) -> str:
    """Reconstruct the .txt filename the crawler would have saved for a URL."""
    import hashlib
    p = urlparse(url).path.strip("/").replace("/", "_")
    p = re.sub(r'[^\w\-_.]', '_', p)[:80]
    h = hashlib.md5(url.encode()).hexdigest()[:6]
    return f"{p}_{h}.txt"

def make_pdf_filename(url: str) -> str:
    import hashlib
    p = urlparse(url).path.strip("/").replace("/", "_")
    p = re.sub(r'[^\w\-_.]', '_', p)[:80]
    h = hashlib.md5(url.encode()).hexdigest()[:6]
    return f"{p}_{h}.pdf"

def url_matches_removal_pattern(url: str) -> bool:
    return any(re.search(pat, url, re.I) for pat in REMOVE_URL_PATTERNS)

def chunk_file_exists(rec: dict) -> bool:
    """
    Returns True if the source file for this chunk still exists on disk.
    """
    source = rec.get("source", "html")
    url    = rec.get("url", "")

    if source in ("pdf", "pdf_scanned"):
        # Check in _pdfs/ folder
        pdf_name = make_pdf_filename(url)
        if (PDF_DIR / pdf_name).exists():
            return True
        # Also check raw PDF dir (manually placed PDFs)
        doc_name = rec.get("doc_name") or rec.get("title", "")
        if doc_name and (RAW_PDF_DIR / doc_name).exists():
            return True
        return False

    elif source in ("html", "scraped_html", "api_json", "json"):
        txt_name = make_txt_filename(url)
        # Search all subdirs of TXT_DATA_DIR for this filename
        matches = list(TXT_DATA_DIR.rglob(txt_name))
        return len(matches) > 0

    # Unknown source — keep it (don't delete what we can't verify)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview removals without writing")
    args = parser.parse_args()

    if not JSONL_PATH.exists():
        print(f"Not found: {JSONL_PATH}")
        return

    # Backup
    if not args.dry_run:
        backup = JSONL_PATH.with_suffix(
            f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        shutil.copy(JSONL_PATH, backup)
        print(f"Backup: {backup}")

    total   = 0
    kept    = []
    removed = Counter()

    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            url = rec.get("url", "")

            # Rule 1: manual URL pattern removal
            if url_matches_removal_pattern(url):
                removed["url_pattern"] += 1
                if args.dry_run:
                    print(f"  [URL_PATTERN] {url[:90]}")
                continue

            # Rule 2: source file no longer exists
            if not chunk_file_exists(rec):
                removed["file_deleted"] += 1
                if args.dry_run:
                    print(f"  [FILE_GONE]  src={rec.get('source','?'):12s}  "
                          f"{url[:70]}")
                continue

            kept.append(rec)

    total_removed = sum(removed.values())

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Results:")
    print(f"  Total chunks      : {total:,}")
    print(f"  Kept              : {len(kept):,}")
    print(f"  Removed (pattern) : {removed['url_pattern']:,}")
    print(f"  Removed (no file) : {removed['file_deleted']:,}")
    print(f"  Total removed     : {total_removed:,}")

    if args.dry_run:
        print("\nDry run — no changes written.")
        return

    if total_removed == 0:
        print("\nNothing to remove — JSONL already in sync.")
        return

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✓ _chunks.jsonl updated: {total:,} → {len(kept):,} chunks")


if __name__ == "__main__":
    main()