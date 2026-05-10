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
import time
from functools import lru_cache, wraps
from typing import List, NamedTuple, Optional, Set, Tuple

import nltk

from nlp_engine.preprocessing.normalizer import normalize_arabic
from shared.utils.lang import is_arabic as _is_arabic


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class Token(NamedTuple):
    """Enhanced token with optional metadata."""
    text: str
    lemma: Optional[str] = None
    pos: Optional[str] = None  # Part of speech
    is_stopword: bool = False


# ---------------------------------------------------------------------------
# Arabic stop words (university-query domain curated list)
# ---------------------------------------------------------------------------

ARABIC_STOPWORDS: Set[str] = {
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

# Separate single tokens from multi-word phrases
_ARABIC_STOPWORD_PHRASES: Set[str] = {s for s in ARABIC_STOPWORDS if " " in s}
_ARABIC_STOPWORD_TOKENS: Set[str] = {s for s in ARABIC_STOPWORDS if " " not in s}

# Extended character ranges for Arabic (covering all common variants)
_ARABIC_CHARS = (
    r'\u0621-\u063A'  # Arabic letters
    r'\u0641-\u064A'  # More Arabic letters
    r'\u0660-\u0669'  # Arabic-Indic digits
    r'\u0670-\u06FF'  # Extended Arabic (includes some Persian/Urdu chars)
)

# Token regex: captures Arabic text, Latin text, digits, and protected patterns
_ARABIC_TOKEN_RE = re.compile(
    rf'[{_ARABIC_CHARS}]+'  # Arabic tokens
    r'|[A-Za-z]+'            # Latin tokens
    r'|[0-9]+'               # Digit sequences
)

# Protected patterns that should not be stemmed or modified
_PROTECTED_PATTERNS = re.compile(
    r'^[A-Z]{2,4}\d{3,4}$'   # Course codes: CS101, MATH202
    r'|^\d{6,10}$'            # Student IDs
    r'|^[A-Z]+-\d+$'          # Section codes: CS-101
)

# Arabic affixes for light stemming
_ARABIC_PREFIXES: Tuple[str, ...] = (
    "وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ك", "ل", "س"
)
_ARABIC_SUFFIXES: Tuple[str, ...] = (
    "كما", "كم", "كن", "هم", "هن", "نا", "ها", "ات", "ون", "ين", "ان", "ية", "يه", "ة"
)

# Minimum token length after affix stripping
_MIN_TOKEN_LENGTH: int = 2


# ---------------------------------------------------------------------------
# Performance monitoring decorator
# ---------------------------------------------------------------------------

def log_slow_calls(threshold: float = 0.05):
    """Decorator to log function calls that exceed a time threshold."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if elapsed > threshold:
                logger.debug(
                    f"{func.__name__} took {elapsed:.3f}s "
                    f"(args length: {len(str(args[0])) if args else 0})"
                )
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Lazy spaCy loader (English path)
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    """Lazy-load spaCy model with fallback to whitespace tokenization."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("spaCy English model loaded successfully")
        except Exception:
            logger.warning(
                "spaCy 'en_core_web_sm' not found — "
                "English tokenization will use whitespace split."
            )
            _NLP = False  # sentinel: unavailable
    return _NLP if _NLP is not False else None


@lru_cache(maxsize=1)
def _get_english_stopwords() -> Set[str]:
    """Load NLTK English stopwords with caching."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Detect whether the message is primarily 'arabic' or 'english'.
    Uses a character-level heuristic with optional langdetect fallback.
    
    Args:
        text: Input text to analyze
        
    Returns:
        'arabic', 'english', or 'unknown'
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

    # Ambiguous case — try langdetect
    try:
        from langdetect import detect
        lang = detect(text)
        return 'arabic' if lang in ('ar', 'ur', 'fa') else 'english'
    except Exception:
        logger.debug(f"langdetect failed for text: {text[:50]}...")
        return 'arabic' if ratio >= 0.5 else 'english'


def is_arabic_text(text: str) -> bool:
    """Check if text contains any Arabic characters."""
    from shared.utils.lang import is_arabic
    return is_arabic(text)


# ---------------------------------------------------------------------------
# Arabic Affix Stripping (Light Stemming)
# ---------------------------------------------------------------------------

def _is_protected_token(token: str) -> bool:
    """Check if token is a protected pattern (course code, ID, etc.)."""
    return bool(_PROTECTED_PATTERNS.match(token))


@log_slow_calls(threshold=0.01)
def _strip_arabic_affixes(token: str, aggressive: bool = False) -> str:
    """
    Remove frequent Arabic clitics (light stemming).
    
    Args:
        token: Arabic token to stem
        aggressive: If True, iteratively strip multiple layers of affixes
        
    Returns:
        Stemmed token, or original if stripping would make it too short
    """
    if not token or token.isdigit() or len(token) <= _MIN_TOKEN_LENGTH:
        return token

    # Don't stem protected patterns
    if _is_protected_token(token):
        return token

    # Only stem Arabic text
    if not re.search(r'[\u0600-\u06FF]', token):
        return token

    original = token
    stemmed = token

    if aggressive:
        # Iteratively strip multiple prefixes
        changed = True
        while changed and len(stemmed) > _MIN_TOKEN_LENGTH:
            changed = False
            for prefix in _ARABIC_PREFIXES:
                if stemmed.startswith(prefix) and len(stemmed) - len(prefix) >= _MIN_TOKEN_LENGTH:
                    stemmed = stemmed[len(prefix):]
                    changed = True
                    break

        # Iteratively strip multiple suffixes
        changed = True
        while changed and len(stemmed) > _MIN_TOKEN_LENGTH:
            changed = False
            for suffix in _ARABIC_SUFFIXES:
                if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= _MIN_TOKEN_LENGTH:
                    stemmed = stemmed[:-len(suffix)]
                    changed = True
                    break
    else:
        # Strip only first matching prefix and suffix
        for prefix in _ARABIC_PREFIXES:
            if stemmed.startswith(prefix) and len(stemmed) - len(prefix) >= _MIN_TOKEN_LENGTH:
                stemmed = stemmed[len(prefix):]
                break

        for suffix in _ARABIC_SUFFIXES:
            if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= _MIN_TOKEN_LENGTH:
                stemmed = stemmed[:-len(suffix)]
                break

    # If stripping was too aggressive, return original
    if len(stemmed) < _MIN_TOKEN_LENGTH:
        logger.debug(f"Affix stripping too aggressive for '{original}' → '{stemmed}', keeping original")
        return original

    return stemmed


# ---------------------------------------------------------------------------
# Core Tokenization Functions
# ---------------------------------------------------------------------------

def _remove_arabic_stopword_phrases(text: str) -> str:
    """
    Remove multi-word Arabic stop phrases from text.
    Handles punctuation around phrases.
    
    Args:
        text: Normalized Arabic text
        
    Returns:
        Text with stop phrases removed
    """
    if not _ARABIC_STOPWORD_PHRASES:
        return text

    # First, normalize punctuation/whitespace around phrases
    text = ' ' + text + ' '
    
    for phrase in _ARABIC_STOPWORD_PHRASES:
        # Escape for regex, but allow flexible whitespace
        pattern = re.escape(phrase).replace(r'\ ', r'\s+')
        # Remove phrase with optional surrounding punctuation
        text = re.sub(rf'[\s\.,!?;:\(\)\[\]\{{\}}]' + pattern + r'[\s\.,!?;:\(\)\[\]\{{\}}]', ' ', text)
    
    return text.strip()


@log_slow_calls(threshold=0.02)
def _arabic_tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """
    Light normalization + regex tokenization for Arabic.
    
    Args:
        text: Arabic text to tokenize
        remove_stopwords: Whether to filter stopwords and apply stemming
        
    Returns:
        List of token strings
    """
    if not text:
        return []

    # Normalize Arabic text
    text = normalize_arabic(text)

    if remove_stopwords:
        # Remove multi-word phrases first
        text = _remove_arabic_stopword_phrases(text)

    # Extract tokens
    tokens = _ARABIC_TOKEN_RE.findall(text)

    if remove_stopwords:
        # Filter single-word stopwords
        tokens = [t for t in tokens if t not in _ARABIC_STOPWORD_TOKENS]
        
        # Apply light stemming
        tokens = [_strip_arabic_affixes(t) for t in tokens]
        
        # Filter again after stemming (in case stemming revealed a stopword)
        tokens = [t for t in tokens if t and t not in _ARABIC_STOPWORD_TOKENS]

    return tokens


@log_slow_calls(threshold=0.02)
def _english_tokenize(text: str, remove_stopwords: bool = False, lowercase: bool = True) -> List[str]:
    """
    spaCy tokenization for English (falls back to whitespace).
    
    Args:
        text: English text to tokenize
        remove_stopwords: Whether to filter stopwords
        lowercase: Whether to lowercase tokens
        
    Returns:
        List of token strings
    """
    nlp = _get_nlp()
    
    if nlp:
        # spaCy tokenization
        doc = nlp(text)
        tokens = [tok.text for tok in doc if not tok.is_space]
        
        if lowercase:
            tokens = [t.lower() for t in tokens]
        
        if remove_stopwords:
            sw = _get_english_stopwords()
            tokens = [
                t for t in tokens 
                if t not in sw and len(t) > 1 and not t.isdigit()
            ]
    else:
        # Fallback: whitespace tokenization
        tokens = text.split()
        
        if lowercase:
            tokens = [t.lower() for t in tokens]
        
        if remove_stopwords:
            sw = _get_english_stopwords()
            tokens = [t for t in tokens if t not in sw]

    return tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize *text* — Arabic or English, detected automatically.
    
    Args:
        text: Input text to tokenize
        lowercase: Whether to lowercase English tokens
        
    Returns:
        List of token strings without filtering stop words
        
    Examples:
        >>> tokenize("Hello world")
        ['hello', 'world']
        
        >>> tokenize("مرحبا بالعالم")
        ['مرحبا', 'بالعالم']
    """
    if not text:
        return []
    
    if is_arabic_text(text):
        return _arabic_tokenize(text, remove_stopwords=False)
    
    return _english_tokenize(text, remove_stopwords=False, lowercase=lowercase)


def tokenize_no_stopwords(text: str, aggressive_stemming: bool = False) -> List[str]:
    """
    Tokenize and remove stop words (Arabic or English, auto-detected).
    Best for bag-of-words / TF-IDF use cases.
    
    Args:
        text: Input text to tokenize
        aggressive_stemming: If True, iteratively strip multiple affix layers
        
    Returns:
        List of token strings with stopwords removed
        
    Examples:
        >>> tokenize_no_stopwords("I want to know about CS101")
        ['want', 'know', 'cs101']
        
        >>> tokenize_no_stopwords("اريد معرفة المزيد عن القسم")
        ['معرفة', 'مزيد', 'قسم']
    """
    if not text:
        return []
    
    if is_arabic_text(text):
        return _arabic_tokenize(text, remove_stopwords=True)
    
    return _english_tokenize(text, remove_stopwords=True, lowercase=True)


def tokenize_with_metadata(text: str) -> List[Token]:
    """
    Tokenize with additional metadata (lemma, POS).
    Currently fully supported for English only.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of Token namedtuples with metadata
    """
    if not text:
        return []
    
    if is_arabic_text(text):
        # Arabic: basic metadata
        tokens = _arabic_tokenize(text, remove_stopwords=False)
        return [
            Token(
                text=t,
                lemma=_strip_arabic_affixes(t) if not _is_protected_token(t) else t,
                is_stopword=t in _ARABIC_STOPWORD_TOKENS
            )
            for t in tokens
        ]
    
    # English: spaCy metadata
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        sw = _get_english_stopwords()
        return [
            Token(
                text=tok.text.lower(),
                lemma=tok.lemma_.lower(),
                pos=tok.pos_,
                is_stopword=tok.text.lower() in sw
            )
            for tok in doc
            if not tok.is_space
        ]
    
    # Fallback
    tokens = text.lower().split()
    sw = _get_english_stopwords()
    return [
        Token(text=t, is_stopword=t in sw)
        for t in tokens
    ]


def lemmatize(text: str) -> List[str]:
    """
    Lemmatize *text*.
    • English → spaCy lemmas
    • Arabic  → light normalization (no heavy morphological analyzer)
    
    Args:
        text: Input text to lemmatize
        
    Returns:
        List of lemmatized tokens
    """
    if not text:
        return []
    
    if is_arabic_text(text):
        # Arabic: light stemming instead of full lemmatization
        tokens = _arabic_tokenize(text, remove_stopwords=False)
        return [_strip_arabic_affixes(t) for t in tokens]
    
    # English: spaCy lemmatization
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        return [
            tok.lemma_.lower() 
            for tok in doc 
            if not tok.is_space and not tok.is_punct
        ]
    
    # Fallback
    return text.lower().split()


def light_arabic_stem(text: str, aggressive: bool = False) -> str:
    """
    Return a whitespace-joined light stemmed Arabic string for retrieval.
    
    Args:
        text: Arabic text to stem
        aggressive: If True, apply iterative affix stripping
        
    Returns:
        Stemmed text as a single string
        
    Examples:
        >>> light_arabic_stem("الجامعة الإسلامية")
        'جامعة إسلامية'
    """
    if not text:
        return ""

    if not is_arabic_text(text):
        return text.strip()

    # Normalize and tokenize
    text = normalize_arabic(text)
    tokens = _ARABIC_TOKEN_RE.findall(text)
    
    # Apply stemming
    stemmed = [_strip_arabic_affixes(t, aggressive=aggressive) for t in tokens]
    
    # Remove empty tokens
    stemmed = [t for t in stemmed if t]
    
    return " ".join(stemmed)


def debug_tokenization(text: str) -> dict:
    """
    Debug helper to show tokenization steps and differences.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with original, tokenized, and diagnostic information
    """
    if not text:
        return {"original": "", "tokens": [], "stemmed": []}
    
    lang = detect_language(text)
    tokens = tokenize(text)
    tokens_no_sw = tokenize_no_stopwords(text)
    
    result = {
        "original": text,
        "language": lang,
        "token_count": len(tokens),
        "tokens": tokens,
        "tokens_no_stopwords": tokens_no_sw,
        "stopwords_removed": len(tokens) - len(tokens_no_sw),
    }
    
    if lang == "arabic":
        result["stemmed"] = light_arabic_stem(text)
        
        # Show affix stripping details
        affix_details = []
        for token in tokens:
            stemmed = _strip_arabic_affixes(token)
            if stemmed != token:
                affix_details.append(f"{token} → {stemmed}")
        result["affix_changes"] = affix_details
    
    return result


# ---------------------------------------------------------------------------
# Batch processing utilities
# ---------------------------------------------------------------------------

def tokenize_batch(texts: List[str], remove_stopwords: bool = False) -> List[List[str]]:
    """
    Tokenize multiple texts efficiently.
    
    Args:
        texts: List of input texts
        remove_stopwords: Whether to filter stopwords
        
    Returns:
        List of token lists
    """
    # Pre-load resources to avoid repeated checks
    _get_nlp()
    _get_english_stopwords()
    
    results = []
    for text in texts:
        if remove_stopwords:
            results.append(tokenize_no_stopwords(text))
        else:
            results.append(tokenize(text))
    
    return results


def add_custom_stopwords(words: List[str], language: str = "arabic"):
    """
    Add domain-specific stopwords at runtime.
    
    Args:
        words: List of words to add as stopwords
        language: 'arabic' or 'english'
        
    Examples:
        >>> add_custom_stopwords(["دكتور", "جامعة"], "arabic")
    """
    global ARABIC_STOPWORDS, _ARABIC_STOPWORD_PHRASES, _ARABIC_STOPWORD_TOKENS
    
    if language == "arabic":
        ARABIC_STOPWORDS.update(words)
        _ARABIC_STOPWORD_PHRASES = {s for s in ARABIC_STOPWORDS if " " in s}
        _ARABIC_STOPWORD_TOKENS = {s for s in ARABIC_STOPWORDS if " " not in s}
        logger.info(f"Added {len(words)} Arabic stopwords")
    
    elif language == "english":
        # For English, we rely on NLTK, but can extend
        sw = _get_english_stopwords()
        sw.update(words)
        logger.info(f"Added {len(words)} English stopwords")