**# detecting the reversed arabic chunks:** 

python .\scripts\detecting_arabic_Reversed_chunks.py

**# Step 1 — fix the existing _chunks.jsonl (one-time cleanup)**
python scripts/fix_reversed_arabic.py

**# Step 2 — now ingest the clean data**
python scripts/ingest_ppu_data.py
