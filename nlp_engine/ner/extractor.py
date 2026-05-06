"""
Named Entity Extractor — bilingual Arabic/English for PPU queries.

Combines:
  • Regex patterns — course codes, student IDs (work in any language)
  • Arabic keyword lists — department names, course names, semesters in Arabic
  • English keyword lists — same in English
  • spaCy NER — dates and person names (English path)

Usage
-----
    from nlp_engine.ner.extractor import extract_entities
    entities = extract_entities("اريد معرفة رسوم مادة CS401 في الفصل الثاني")
    # [
    #   {"type": "COURSE_CODE", "value": "CS401", ...},
    #   {"type": "SEMESTER",    "value": "الفصل الثاني", ...},
    # ]
"""
from __future__ import annotations

import re
import logging
from functools import lru_cache
from typing import List, Dict

from nlp_engine.ner.entities import EntityType
from nlp_engine.preprocessing.normalizer import normalize_arabic
from shared.utils.lang import is_arabic as _is_arabic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy loader (English path only)
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            logger.warning("spaCy 'en_core_web_sm' not found; NER will be regex/keyword-only.")
            _NLP = False
    return _NLP


# ---------------------------------------------------------------------------
# Regex patterns (language-agnostic)
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: List[tuple] = [
    # Course code: 2-4 uppercase letters + 3-4 digits (CS401, ENGL1001, IT302)
    (EntityType.COURSE_CODE, r'\b([A-Z]{2,4}\s?\d{3,4})\b', True),   # uppercase input
    # PPU Student ID: 7+ digits, optionally hyphenated
    (EntityType.STUDENT_ID,  r'\b(\d{3}-?\d{4,5})\b', False),
    # Arabic-Indic student ID: ١٢٠٢٣٤٥
    (EntityType.STUDENT_ID,  r'[\u0660-\u0669]{7,}', False),
    # Credit hours / fee amounts
    (EntityType.AMOUNT,      r'\b(\d+(?:\.\d+)?)\s*(?:دينار|دنانير|شيكل|NIS|ILS|JD|\$|ساعه|ساعة معتمدة)\b', False),
]

# ---------------------------------------------------------------------------
# Arabic keyword lists (post-normalization — no diacritics, ا not أ etc.)
# ---------------------------------------------------------------------------

_AR_DEPARTMENTS: List[str] = [
    # Engineering
    'هندسة الحاسوب', 'هندسة الكهرباء', 'هندسة المدني', 'هندسة الميكانيك',
    'هندسة الاتصالات', 'هندسة الكيمياء',
    # CS / IT
    'علم الحاسوب', 'علوم الحاسوب', 'تكنولوجيا المعلومات', 'نظم المعلومات',
    'الذكاء الاصطناعي',
    # Business / other
    'ادارة الاعمال', 'المحاسبة', 'التسويق', 'الاقتصاد',
    'الصيدلة', 'التمريض', 'الطب',
    # Architecture
    'العمارة', 'الهندسة المعمارية',
    # Math / Science
    'الرياضيات', 'الفيزياء', 'الكيمياء', 'الاحياء',
]

_AR_SEMESTERS: List[str] = [
    'الفصل الاول', 'الفصل الثاني', 'الفصل الثالث',
    'الفصل الصيفي', 'الفصل الدراسي الاول', 'الفصل الدراسي الثاني',
    'الربيع', 'الخريف', 'الصيف',
    'فصل الربيع', 'فصل الخريف', 'فصل الصيف',
    'فصل اول', 'فصل ثاني', 'فصل ثالث',
    'ترم اول', 'ترم ثاني', 'ترم ثالث',
    'الترم الاول', 'الترم الثاني', 'الترم الثالث',
]

_AR_COURSES: List[str] = [
    'هندسة البرمجيات', 'قواعد البيانات', 'هياكل البيانات',
    'الخوارزميات', 'انظمة التشغيل', 'شبكات الحاسوب',
    'الذكاء الاصطناعي', 'تعلم الالة', 'معالجة اللغات الطبيعية',
    'الدوائر المنطقية', 'البرمجة الشيئية', 'برمجة الشبكات',
    'امن المعلومات', 'الحوسبة السحابية', 'الجبر الخطي',
    'التفاضل والتكامل', 'الاحصاء', 'المنطق الرياضي',
]

# ---------------------------------------------------------------------------
# English keyword lists
# ---------------------------------------------------------------------------

_EN_DEPARTMENTS: List[str] = [
    "computer science", "computer engineering", "electrical engineering",
    "civil engineering", "mechanical engineering", "business administration",
    "information technology", "architecture", "pharmacy", "nursing",
    "applied mathematics", "physics", "chemistry", "artificial intelligence",
]

_EN_SEMESTERS: List[str] = [
    "spring semester", "fall semester", "summer semester",
    "first semester", "second semester", "third semester",
    "spring", "fall", "summer",
]

_EN_COURSES: List[str] = [
    "software engineering", "data structures", "algorithms",
    "operating systems", "database systems", "computer networks",
    "artificial intelligence", "machine learning", "natural language processing",
    "digital logic", "calculus", "linear algebra", "discrete mathematics",
    "object oriented programming", "information security",
]

# ---------------------------------------------------------------------------
# Arabic normalization helpers for keyword matching
# ---------------------------------------------------------------------------

_ARABIC_NOISE = r'[\u0610-\u061A\u064B-\u065F\u0670\u0640]*'
_ARABIC_VARIANTS = {
    'ا': '[اأإآٱ]',
    'ه': '[هة]',
    'ة': '[هة]',
    'ي': '[يىئ]',
    'ى': '[يىئ]',
    'و': '[وؤ]',
    'ء': '[ءٔ]',
}


def _normalize_keywords(values: List[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for val in values:
        norm = normalize_arabic(val)
        if norm and norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


_AR_DEPARTMENTS_NORM = _normalize_keywords(_AR_DEPARTMENTS)
_AR_SEMESTERS_NORM   = _normalize_keywords(_AR_SEMESTERS)
_AR_COURSES_NORM     = _normalize_keywords(_AR_COURSES)


@lru_cache(maxsize=512)
def _arabic_fuzzy_pattern(keyword: str) -> re.Pattern:
    parts: List[str] = []
    for ch in keyword:
        if ch.isspace():
            parts.append(r"\s+")
            continue
        parts.append(_ARABIC_VARIANTS.get(ch, re.escape(ch)))
        parts.append(_ARABIC_NOISE)
    return re.compile("".join(parts))


def _find_arabic_span(text: str, keyword: str) -> tuple[int, int] | None:
    if not text or not keyword:
        return None
    for variant in _keyword_variants(keyword):
        idx = text.find(variant)
        if idx != -1:
            return idx, idx + len(variant)
        pattern = _arabic_fuzzy_pattern(variant)
        match = pattern.search(text)
        if match:
            return match.start(), match.end()
    return None


def _keyword_variants(keyword: str) -> List[str]:
    """Generate Arabic keyword variants for attached prepositions (e.g., للفصل)."""
    if not keyword:
        return []

    variants = [keyword]
    if keyword.startswith("ال") and len(keyword) > 2:
        base = keyword[2:]
        for prefix in ("لل", "بال", "كال", "وال", "فال"):
            variants.append(prefix + base)
    return variants


def _find_keyword_in_normalized(normalized_text: str, keyword: str) -> tuple[str, int] | None:
    """Find keyword or its variants in normalized Arabic text."""
    for variant in _keyword_variants(keyword):
        idx = normalized_text.find(variant)
        if idx != -1:
            return variant, idx
    return None


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------


def _keyword_search(
    text: str,
    keywords: List[str],
    entity_type: str,
    results: List[Dict],
    seen: set,
) -> None:
    """Search for keyword list entries in text and record matches."""
    lower = text.lower()
    for kw in keywords:
        idx = lower.find(kw.lower())
        if idx != -1:
            end = idx + len(kw)
            span = (idx, end)
            if span not in seen:
                seen.add(span)
                results.append({
                    "type":  entity_type,
                    "value": text[idx:end],
                    "start": idx,
                    "end":   end,
                })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> List[Dict]:
    """
    Extract named entities from *text* (Arabic or English).

    Returns
    -------
    List[dict] — each item: {type, value, start, end}
                 sorted by start position.
    """
    entities: List[Dict] = []
    seen_spans: set = set()

    def _add(entity_type: str, value: str, start: int, end: int):
        span = (start, end)
        if span not in seen_spans:
            seen_spans.add(span)
            entities.append({"type": entity_type, "value": value, "start": start, "end": end})

    # --- 1. Regex patterns (language-agnostic) ---
    upper = text.upper()
    for entity_type, pattern, use_upper in _REGEX_PATTERNS:
        source = upper if use_upper else text
        for m in re.finditer(pattern, source, re.IGNORECASE):
            raw = text[m.start():m.end()]
            _add(entity_type, raw.strip(), m.start(), m.end())

    # --- 2. Arabic semester and department keywords ---
    if _is_arabic(text):
        normalized_text = normalize_arabic(text)

        # Sort by length descending so longer phrases match first
        for kw in sorted(_AR_SEMESTERS_NORM, key=len, reverse=True):
            match = _find_keyword_in_normalized(normalized_text, kw)
            if match:
                variant, idx = match
                span = _find_arabic_span(text, variant)
                if span:
                    _add(EntityType.SEMESTER, text[span[0]:span[1]], span[0], span[1])
                else:
                    _add(EntityType.SEMESTER, variant, idx, idx + len(variant))

        for kw in sorted(_AR_DEPARTMENTS_NORM, key=len, reverse=True):
            match = _find_keyword_in_normalized(normalized_text, kw)
            if match:
                variant, idx = match
                span = _find_arabic_span(text, variant)
                if span:
                    _add(EntityType.DEPARTMENT, text[span[0]:span[1]], span[0], span[1])
                else:
                    _add(EntityType.DEPARTMENT, variant, idx, idx + len(variant))

        for kw in sorted(_AR_COURSES_NORM, key=len, reverse=True):
            match = _find_keyword_in_normalized(normalized_text, kw)
            if match:
                variant, idx = match
                span = _find_arabic_span(text, variant)
                if span:
                    _add(EntityType.COURSE_NAME, text[span[0]:span[1]], span[0], span[1])
                else:
                    _add(EntityType.COURSE_NAME, variant, idx, idx + len(variant))

    # --- 3. English keyword matching ---
    lower = text.lower()
    for kw in sorted(_EN_SEMESTERS, key=len, reverse=True):
        idx = lower.find(kw)
        if idx != -1:
            _add(EntityType.SEMESTER, text[idx:idx+len(kw)], idx, idx+len(kw))

    for kw in sorted(_EN_DEPARTMENTS, key=len, reverse=True):
        idx = lower.find(kw)
        if idx != -1:
            _add(EntityType.DEPARTMENT, text[idx:idx+len(kw)], idx, idx+len(kw))

    for kw in sorted(_EN_COURSES, key=len, reverse=True):
        idx = lower.find(kw)
        if idx != -1:
            _add(EntityType.COURSE_NAME, text[idx:idx+len(kw)], idx, idx+len(kw))

    # --- 4. spaCy NER — always run (handles mixed Arabic+English queries) ---
    # Running on all text ensures English dates/persons are captured even when
    # the query contains Arabic characters (common at PPU: "CS401 due May 12").
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        spacy_map = {
            "DATE":   EntityType.DATE,
            "PERSON": EntityType.PERSON,
            "ORG":    EntityType.DEPARTMENT,
            "MONEY":  EntityType.AMOUNT,
        }
        for ent in doc.ents:
            mapped = spacy_map.get(ent.label_)
            if mapped:
                _add(mapped, ent.text, ent.start_char, ent.end_char)

    # --- 5. Numeric date regex — always run (DD/MM/YYYY works in any script) ---
    _date_re = r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})'
    for m in re.finditer(_date_re, text):
        _add(EntityType.DATE, m.group(), m.start(), m.end())

    # Year-only mentions (e.g. 2025, 2026) — only when Arabic text is present
    # to avoid false-positives against 4-digit student IDs in English text.
    if _is_arabic(text):
        for m in re.finditer(r'\b(20\d{2}|19\d{2})\b', text):
            _add(EntityType.DATE, m.group(), m.start(), m.end())

    entities.sort(key=lambda e: e["start"])
    return entities


def entities_to_dict(entities: List[Dict]) -> Dict[str, List[str]]:
    """Flatten entity list → {type: [value, ...]}."""
    result: Dict[str, List[str]] = {}
    for ent in entities:
        result.setdefault(ent["type"], []).append(ent["value"])
    return result
