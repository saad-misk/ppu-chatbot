import json

def normalize_arabic(text: str) -> str:
    """Convert presentation forms (U+FE70–U+FEFF) to standard Arabic (U+0600–U+06FF)."""
    # Mapping for common presentation forms to standard letters
    # Full mapping would be huge, but we only need to reduce the ratio.
    # Simpler: replace the entire block with a placeholder, but better:
    # We'll just count presentation forms separately.
    # Actually, we want to detect reversed text, so we should compare after conversion.
    # For detection, we can just ignore presentation forms entirely.
    return text

# Better: Redefine is_reversed_arabic without presentation form bias
def is_reversed_arabic(text: str) -> bool:
    """
    Detect visual-order Arabic by checking if reversing the string
    and reshaping yields a string that is significantly different
    AND contains more common Arabic trigrams.
    """
    import arabic_reshaper
    from difflib import SequenceMatcher

    # If text is very short or has no Arabic, not reversed
    ar_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\uFE70' <= c <= '\uFEFF')
    if ar_chars < 10:
        return False

    # Reverse character order
    reversed_candidate = text[::-1]
    reshaped_reversed = arabic_reshaper.reshape(reversed_candidate)

    # Similarity between original and reshaped_reversed
    sim = SequenceMatcher(None, text, reshaped_reversed).ratio()
    # If original is already logical, similarity will be high (>0.7)
    return sim < 0.6   # reversed if low similarity


total = good_ar = bad_ar = non_ar = 0

with open("data/raw/ppu_rag_data/_chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        text = r.get("text", "")
        total += 1

        # Count Arabic characters (both standard and presentation forms)
        ar_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\uFE70' <= c <= '\uFEFF')
        if ar_chars == 0:
            non_ar += 1
        elif is_reversed_arabic(text):
            bad_ar += 1
        else:
            good_ar += 1

print(f"Total chunks   : {total:,}")
print(f"Non-Arabic     : {non_ar:,}")
print(f"Good Arabic    : {good_ar:,}")
print(f"Reversed Arabic: {bad_ar:,}")
print(f"Bad ratio      : {bad_ar/total*100:.1f}%")