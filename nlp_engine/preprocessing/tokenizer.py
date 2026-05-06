"""
Tokenizer — bilingual Arabic/English tokenization.

Strategy:
    • Arabic text  → light normalization + regex tokenization
                                     + optional stop-word filtering + light affix stripping
    • English text → spaCy tokenizer + NLTK stop-word filtering
    • Mixed text   → handled by the Arabic path (safe for mixed strings)

Language detection uses a fast character heuristic with an optional
langdetect fallback for ambiguous mixed-language inputs.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List

import nltk

from nlp_engine.preprocessing.normalizer import normalize_arabic
from shared.utils.lang import is_arabic as _is_arabic


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arabic stop words (university-query domain curated list)
# ---------------------------------------------------------------------------

ARABIC_STOPWORDS: set[str] = {
    # Pronouns
    'انا', 'انت', 'انتي', 'هو', 'هي', 'نحن', 'هم', 'هن', 'انتم',
    # Prepositions / conjunctions
    'في', 'من', 'الى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'بين', 'خلال',
    'عند', 'حتى', 'لكن', 'واو', 'او', 'ام', 'اذا', 'لو', 'ان', 'اذ',
    # Articles / determiners
    'ال', 'هذا', 'هذه', 'ذلك', 'تلك', 'كل', 'بعض', 'اي',
    # Common verbs (forms of "to be", "want", "have")
    'هل', 'كان', 'يكون', 'اريد', 'اود', 'احتاج', 'عندي', 'لدي',
    'يوجد', 'توجد', 'موجود', 'موجوده',
    # Question words
    'ما', 'ماذا', 'من', 'متى', 'اين', 'كيف', 'لماذا', 'لم', 'هل',
    'كم', 'ماهو', 'ماهي',
    # Filler / greetings handled by state machine — strip here for retrieval
    'مرحبا', 'السلام', 'عليكم', 'اهلا', 'وسهلا', 'شكرا', 'جزيلا',
    # Common Arabic words with no semantic value for retrieval
    'يمكن', 'ممكن', 'برجاء', 'رجاء', 'من فضلك', 'لو سمحت',
    'اخبرني', 'اخبر', 'قل', 'وضح', 'شرح',
}

_ARABIC_STOPWORD_PHRASES = {s for s in ARABIC_STOPWORDS if " " in s}
_ARABIC_STOPWORD_TOKENS = {s for s in ARABIC_STOPWORDS if " " not in s}

_ARABIC_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0621-\u064A\u0660-\u0669\u06F0-\u06F9]+")

_ARABIC_PREFIXES = (
    "وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ك", "ل", "س"
)
_ARABIC_SUFFIXES = (
    "كما", "كم", "كن", "هم", "هن", "نا", "ها", "ات", "ون", "ين", "ان", "ية", "يه", "ة"
)


# ---------------------------------------------------------------------------
# Lazy spaCy loader (English path)
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception:
            logger.warning("spaCy 'en_core_web_sm' not found — English tokenization will use whitespace split.")
            _NLP = False   # sentinel: unavailable
    return _NLP


@lru_cache(maxsize=1)
def _get_english_stopwords() -> set:
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_arabic(text: str) -> bool:  # keep local alias so rest of file doesn't change
    from shared.utils.lang import is_arabic
    return is_arabic(text)


def _arabic_tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Light normalization + regex tokenization for Arabic."""
    if not text:
        return []

    text = normalize_arabic(text)

    if remove_stopwords and _ARABIC_STOPWORD_PHRASES:
        for phrase in _ARABIC_STOPWORD_PHRASES:
            text = re.sub(rf"(?<!\S){re.escape(phrase)}(?!\S)", " ", text)

    tokens = _ARABIC_TOKEN_RE.findall(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _ARABIC_STOPWORD_TOKENS]
        tokens = [_strip_arabic_affixes(t) for t in tokens]
        tokens = [t for t in tokens if t and t not in _ARABIC_STOPWORD_TOKENS]

    return tokens


def _english_tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """spaCy tokenization for English (falls back to whitespace)."""
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        tokens = [tok.text for tok in doc if not tok.is_space]
        if remove_stopwords:
            sw = _get_english_stopwords()
            tokens = [t for t in tokens if not (t.lower() in sw or len(t) == 1)]
    else:
        tokens = text.split()
        if remove_stopwords:
            sw = _get_english_stopwords()
            tokens = [t for t in tokens if t.lower() not in sw]
    return tokens


def _strip_arabic_affixes(token: str) -> str:
    """Remove frequent Arabic clitics (light stemming)."""
    if not token or token.isdigit() or len(token) <= 2:
        return token

    if not re.search(r'[\u0600-\u06FF]', token):
        return token

    for prefix in _ARABIC_PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            token = token[len(prefix):]
            break

    for suffix in _ARABIC_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            token = token[:-len(suffix)]
            break

    return token


def light_arabic_stem(text: str) -> str:
    """Return a whitespace-joined light stemmed Arabic string for retrieval."""
    if not text:
        return ""

    if not _is_arabic(text):
        return text.strip()

    text = normalize_arabic(text)
    tokens = _ARABIC_TOKEN_RE.findall(text)
    stemmed = [_strip_arabic_affixes(t) for t in tokens]
    stemmed = [t for t in stemmed if t]
    return " ".join(stemmed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Tokenize *text* — Arabic or English, detected automatically.

    Returns a list of token strings without filtering stop words.
    """
    if not text:
        return []
    if _is_arabic(text):
        return _arabic_tokenize(text, remove_stopwords=False)
    return _english_tokenize(text, remove_stopwords=False)


def tokenize_no_stopwords(text: str) -> List[str]:
    """
    Tokenize and remove stop words (Arabic or English, auto-detected).
    Best for bag-of-words / TF-IDF use cases.
    """
    if not text:
        return []
    if _is_arabic(text):
        return _arabic_tokenize(text, remove_stopwords=True)
    return _english_tokenize(text, remove_stopwords=True)


def lemmatize(text: str) -> List[str]:
    """
    Lemmatize *text*.
    • English → spaCy lemmas
    • Arabic  → light normalization (no heavy morphological analyzer)
    """
    if not text:
        return []
    if _is_arabic(text):
        return _arabic_tokenize(text, remove_stopwords=False)
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        return [tok.lemma_.lower() for tok in doc if not tok.is_space and not tok.is_punct]
    return text.split()


def detect_language(text: str) -> str:
    """
    Detect whether the message is primarily 'arabic' or 'english'.
    Uses a character-level heuristic with optional langdetect fallback.
    """
    if not text:
        return 'unknown'

    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))

    if arabic_chars == 0 and latin_chars == 0:
        return 'unknown'
    if arabic_chars == 0:
        return 'english'
    if latin_chars == 0:
        return 'arabic'

    ratio = arabic_chars / (arabic_chars + latin_chars)
    if ratio >= 0.6:
        return 'arabic'
    if ratio <= 0.4:
        return 'english'

    try:
        from langdetect import detect
        lang = detect(text)
        return 'arabic' if lang in ('ar', 'ur', 'fa') else 'english'
    except Exception:
        return 'arabic' if ratio >= 0.5 else 'english'
