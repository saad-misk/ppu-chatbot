# Full ingest (both txt + PDFs)

python scripts/ingest_ppu_data.py

# Preview what would be ingested (no writes)

python scripts/ingest_ppu_data.py --dry-run

# Wipe ChromaDB and re-index everything from scratch

python scripts/ingest_ppu_data.py --reset

# Only the scraped pages (skip PDFs)

python scripts/ingest_ppu_data.py --txt-only

# Only the raw PDF guides

python scripts/ingest_ppu_data.py --pdf-only

# Larger batches for faster upsert

python scripts/ingest_ppu_data.py --batch-size 200
