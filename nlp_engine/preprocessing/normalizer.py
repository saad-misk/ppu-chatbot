"""
Text normalizer — supports both Arabic (primary) and English.

Arabic normalization steps:
    1. Remove tashkeel (diacritics / harakat)
    2. Remove tatweel (kashida — elongation character ـ)
    3. Normalize alef variants (أ إ آ ا → ا)
    4. Normalize teh marbuta (ة → ه)
    5. Normalize waw (ؤ → و) and yeh (ئ → ي / ى → ي)
    6. Normalize Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩ → 0123456789)
    7. Collapse repeated Arabic letters (e.g., مررررحبا → مرحبا)

English normalization steps:
    1. Unicode NFC normalization
    2. Collapse whitespace
    3. Lowercase
    4. Optional punctuation / digit removal
"""
import re
import unicodedata

from shared.utils.lang import is_arabic as _is_arabic_shared


# ---------------------------------------------------------------------------
# Arabic-specific constants
# ---------------------------------------------------------------------------

# Diacritics (tashkeel) Unicode range
_ARABIC_DIACRITICS = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670]')

# Tatweel / kashida
_TATWEEL = re.compile(r'\u0640')

# Bidi / direction marks that sometimes appear in scraped PDFs
_BIDI_MARKS = re.compile(r'[\u200e\u200f\u202a-\u202e]')

# Alef variants → bare alef
_ALEF_VARIANTS = str.maketrans({
    '\u0623': '\u0627',   # أ → ا
    '\u0625': '\u0627',   # إ → ا
    '\u0622': '\u0627',   # آ → ا
    '\u0671': '\u0627',   # ٱ → ا
})

# Teh marbuta → heh (optional, helps recall)
_TEH_MARBUTA = str.maketrans({'\u0629': '\u0647'})   # ة → ه

# Waw with hamza → waw
_WAW_HAMZA = str.maketrans({'\u0624': '\u0648'})     # ؤ → و

# Yeh variants → yeh
_YEH_VARIANTS = str.maketrans({
    '\u0626': '\u064A',   # ئ → ي
    '\u0649': '\u064A',   # ى → ي
})

# Arabic-Indic and Eastern Arabic-Indic digits → Western
_ARABIC_DIGITS = str.maketrans({
    '\u0660': '0', '\u0661': '1', '\u0662': '2', '\u0663': '3', '\u0664': '4',
    '\u0665': '5', '\u0666': '6', '\u0667': '7', '\u0668': '8', '\u0669': '9',
    '\u06F0': '0', '\u06F1': '1', '\u06F2': '2', '\u06F3': '3', '\u06F4': '4',
    '\u06F5': '5', '\u06F6': '6', '\u06F7': '7', '\u06F8': '8', '\u06F9': '9',
})

# Collapse exaggerated letter repetition (2+ repeats → 2)
_ARABIC_REPEAT = re.compile(r'([\u0621-\u064A])\1{2,}')


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def is_arabic(text: str) -> bool:
    """Return True if the text contains any Arabic Unicode characters."""
    return _is_arabic_shared(text)


def normalize_arabic(text: str) -> str:
    """
    Apply Arabic-specific normalization to *text*.

    Safe to call on mixed Arabic/English strings — only Arabic characters
    are affected by the substitution rules.
    """
    if not text:
        return ""
    # Remove bidi marks, diacritics, and tatweel
    text = _BIDI_MARKS.sub('', text)
    text = _ARABIC_DIACRITICS.sub('', text)
    text = _TATWEEL.sub('', text)
    # Normalize digits and collapse repeated letters
    text = text.translate(_ARABIC_DIGITS)
    text = _ARABIC_REPEAT.sub(r'\1\1', text)
    # Normalize character variants
    text = text.translate(_ALEF_VARIANTS)
    text = text.translate(_TEH_MARBUTA)
    text = text.translate(_WAW_HAMZA)
    text = text.translate(_YEH_VARIANTS)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Unified public API
# ---------------------------------------------------------------------------

def normalize(
    text: str,
    remove_punctuation: bool = False,
    remove_digits: bool = False,
) -> str:
    """
    Normalize *text* for the PPU chatbot (Arabic-primary, English-secondary).

    Steps applied to all text:
      1. Unicode NFC
      2. Arabic normalization (diacritics, tatweel, alef/teh/yeh variants)
      3. Collapse whitespace
      4. Lowercase (affects only Latin characters)
      5. Optional: remove punctuation
      6. Optional: remove digit strings
    """
    if not text:
        return ""

    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Arabic normalization
    text = normalize_arabic(text)

    # 3. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. Lowercase Latin characters (Arabic has no case)
    text = text.lower()

    # 5. Optional: remove punctuation
    if remove_punctuation:
        # Strip punctuation while preserving letters/digits across scripts
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r'\s+', ' ', text).strip()

    # 6. Optional: remove digits (Arabic-Indic + Western digits)
    if remove_digits:
        text = re.sub(r'[\d\u0660-\u0669]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_for_classification(text: str) -> str:
    """
    Pre-classification normalization: remove punctuation, keep digits
    (student IDs, course codes, credit hours).
    """
    return normalize(text, remove_punctuation=True, remove_digits=False)


def normalize_for_display(text: str) -> str:
    """
    Light normalization for display — preserves punctuation and casing.
    """
    return normalize(text, remove_punctuation=False, remove_digits=False)
