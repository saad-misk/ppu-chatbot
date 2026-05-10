"""
Named Entity Extractor — bilingual Arabic/English for PPU queries.

Combines:
  • Regex patterns — course codes, student IDs, course numbers
  • Arabic keyword lists — department names, course names, semesters
  • English keyword lists — same in English
  • spaCy NER — dates and person names (English path)
  • Arabic name patterns — titles + person names
  • PPU-specific: student ID format (YYDSSS), course numbers, grades, status

Student ID Format (PPU):
  YYDSSS where:
    YY = Last 2 digits of enrollment year (10-30 representing 2010-2030)
    D  = Department code (1-9)
    SSS = Student sequence number in department (000-999)
  Example: 221100 = enrolled 2022, CS department, student #100

Course Numbers:
  4-digit internal PPU codes (4447, 4620, etc.)
  Can appear with or without course code prefix
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from nlp_engine.ner.entities import (
    ARABIC_ACADEMIC_STATUS,
    ENGLISH_ACADEMIC_STATUS,
    ENTITY_PRIORITY,
    GRADE_VALUES,
    PPU_DEPARTMENT_CODES,
    STUDENT_ID_DEPT_CODES,
    STUDENT_ID_LENGTH,
    STUDENT_ID_SEQ_MAX,
    STUDENT_ID_SEQ_MIN,
    STUDENT_ID_YEAR_MAX,
    STUDENT_ID_YEAR_MIN,
    VALID_COURSE_PREFIXES,
    EntityType,
)
from nlp_engine.preprocessing.normalizer import normalize_arabic
from shared.utils.lang import is_arabic as _is_arabic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy loader (English path only)
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    """Lazy-load spaCy with fallback."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
            logger.info("spaCy English model loaded for NER")
        except Exception:
            logger.warning(
                "spaCy 'en_core_web_sm' not found; NER will be regex/keyword-only."
            )
            _NLP = False
    return _NLP if _NLP is not False else None


# ---------------------------------------------------------------------------
# Regex patterns — ordered by specificity
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: List[Tuple[EntityType, str, bool]] = [
    # PPU Student ID: exactly 6 digits (YYDSSS format)
    # Must match PPU format: 2-digit year (10-30), 1-digit dept (1-9), 3-digit seq (000-999)
    # Negative lookahead prevents matching when followed by currency/credit terms
    (
        EntityType.STUDENT_ID,
        r'\b(\d{6})\b(?!\s*(?:دينار|دنانير|شيكل|NIS|ILS|JD|\$|ساعة|ساعات|معتمدة|علامة))',
        False
    ),
    
    # PPU Course Number: 4-digit internal codes (4447, 4620, 5055, etc.)
    # Must be 4 digits, typically starting with 4,5,8,3
    (
        EntityType.COURSE_NUMBER,
        r'\b([4385]\d{3})\b(?!\s*(?:دينار|دنانير|شيكل|NIS|ILS|JD|\$|ساعة|ساعات|معتمدة))',
        False
    ),
    
    # Course code: 2-4 uppercase letters + space? + 3-4 digits (CS401, IT377, AC 111)
    (
        EntityType.COURSE_CODE,
        r'\b([A-Z]{2,4}\s?\d{3,4})\b',
        True
    ),
    
    # Credit hours with context (more specific than general amount)
    (
        EntityType.CREDIT_HOURS,
        r'\b(\d{1,2})\s*(?:ساعة|ساعات|ساعه|س\.م\.?)\s*(?:معتمدة)?\b',
        False
    ),
    
    # Credit hours in English context
    (
        EntityType.CREDIT_HOURS,
        r'\b(\d{1,2})\s*(?:credit|credits|cr\.?|hrs?\.?|hours?)\b',
        False
    ),
    
    # Grade: number after "العلامة" or standalone grade
    (
        EntityType.GRADE,
        r'(?:العلامة|علامة|درجة|grade|mark)\s*:?\s*(\d{1,3})\b',
        False
    ),
    
    # Grade: P/F format
    (
        EntityType.GRADE,
        r'\b([PN]P?)\b(?!\s*[A-Z])',  # P or NP but not part of longer text
        False
    ),
    
    # Amount with currency
    (
        EntityType.AMOUNT,
        r'\b(\d+(?:\.\d+)?)\s*(?:دينار|دنانير|شيكل|NIS|ILS|JD|\$)\b',
        False
    ),
]

# Academic status patterns
_ARABIC_STATUS_PATTERN = '|'.join(re.escape(s) for s in ARABIC_ACADEMIC_STATUS)
_ENGLISH_STATUS_PATTERN = '|'.join(re.escape(s) for s in ENGLISH_ACADEMIC_STATUS)

# Arabic person name patterns (titles + names)
_AR_PERSON_PATTERNS = [
    # Dr. First Last
    r'(?:د\.?\s*|دكتور\s+|الدكتور\s+)([\u0621-\u064A]{3,}(?:\s+[\u0621-\u064A]{3,}){1,3})',
    # Prof. First Last
    r'(?:أ\.?\s*|أستاذ\s+|الاستاذ\s+)([\u0621-\u064A]{3,}(?:\s+[\u0621-\u064A]{3,}){1,3})',
    # Mr./Ms./Eng. First Last
    r'(?:م\.?\s*|مهندس\s+|المهندس\s+|أ\.\s*|د\.\s*)([\u0621-\u064A]{3,}(?:\s+[\u0621-\u064A]{3,}){1,3})',
]

# Date patterns
_DATE_REGEX = r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})'
# Year-only pattern: avoid matching parts of student IDs
_YEAR_REGEX = r'(?<!\d)(20\d{2}|19\d{2})(?!\d)'

# Semester patterns like "الفصل الأول" or "الفصل الثاني  ( 13ساعة)"
_SEMESTER_HEADER_PATTERN = re.compile(
    r'(الفصل\s+(?:الاول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|'
    r'الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|'
    r'الدراسي\s+(?:الاول|الثاني|الثالث)|'
    r'الصيفي|الربيعي|الخريفي))'
    r'(?:\s*\(\s*\d+\s*ساعة\s*\))?'
)

# ---------------------------------------------------------------------------
# Arabic keyword lists (post-normalization)
# ---------------------------------------------------------------------------

_AR_DEPARTMENTS: List[str] = [
    # Engineering
    'هندسة الحاسوب', 'هندسة الكهرباء', 'هندسة المدني', 'هندسة الميكانيك',
    'هندسة الاتصالات', 'هندسة الكيمياء',
    # CS / IT
    'علم الحاسوب', 'علوم الحاسوب', 'تكنولوجيا المعلومات', 'نظم المعلومات',
    'الذكاء الاصطناعي', 'علم البيانات', 'امن المعلومات',
    # Business / other
    'ادارة الاعمال', 'المحاسبة', 'التسويق', 'الاقتصاد',
    'الصيدلة', 'التمريض', 'الطب',
    # Architecture
    'العمارة', 'الهندسة المعمارية',
    # Math / Science
    'الرياضيات', 'الفيزياء', 'الكيمياء', 'الاحياء',
]

_AR_SEMESTERS: List[str] = [
    'الفصل الاول', 'الفصل الثاني', 'الفصل الثالث', 'الفصل الرابع',
    'الفصل الخامس', 'الفصل السادس', 'الفصل السابع', 'الفصل الثامن',
    'الفصل الصيفي', 'الفصل الدراسي الاول', 'الفصل الدراسي الثاني',
    'الربيع', 'الخريف', 'الصيف',
    'فصل الربيع', 'فصل الخريف', 'فصل الصيف',
    'فصل اول', 'فصل ثاني', 'فصل ثالث', 'فصل رابع',
    'ترم اول', 'ترم ثاني', 'ترم ثالث',
    'الترم الاول', 'الترم الثاني', 'الترم الثالث',
]

# Real PPU course names from the provided data
_AR_COURSES: List[str] = [
    # CS/IT courses (real PPU names)
    'أنظمة الوكلاء المتنقلة', 'الحقيقة الافتراضية',
    'توجهات حديثة في تكنولوجيا المعلومات', 'مواضيع خاصة',
    'تفاعل الإنسان والحاسوب', 'أمن المعلومات',
    'هندسة برمجيات متقدمة', 'مبادىء الرسم الحاسوبي',
    'الأنظمة الموزعة', 'مبادىء المترجمات',
    'البرمجة المرئية', 'برمجة النظم',
    'تقنيات برمجية عصرية', 'النمذجة والمحاكاة',
    'مقدمة في انظمة التشفير', 'تطوير تطبيقات المحمول',
    'الحوسبة السحابية', 'الحوسبة المتوازية',
    'الأنظمة الخبيرة', 'الويب الدلالي',
    'معالجة الصور', 'نظم المعلومات الحيوية',
    'تطوير تطبيقات المحمول المتقدمة', 'استرجاع المعلومات',
    'إظهار البيانات', 'مخزن البيانات',
    'معالجة اللغة الطبيعية', 'مقدمة الى التعلم العميق',
    'مقدمة الى البيانات الضخمة', 'استرجاع المعلومات للوسائط المتعددة',
    'مواضيع خاصة في علم البيانات', 'تطوير الويب لعلم البيانات',
    
    # Foundation courses
    'إنجليزي A2', 'إنجليزي B1', 'إنجليزي B2',
    'لغة عربية', 'تفاضل وتكامل 1', 'تفاضل وتكامل 2',
    'فيزياء 1', 'فيزياء 2', 'مختبر فيزياء 1', 'مختبر فيزياء 2',
    'الأحياء العامة 1', 'الاحياء العامة2',
    'مختبر احياء عامة 1', 'مختبر الاحياء العامة2',
    'التدريب الميداني 1', 'التدريب الميداني 2',
    'الحاسوب واساسيات البرمجة', 'كيمياء عامة 1', 'كيمياء عامة 2',
    'مختبر كيمياء عامة 1', 'مختبر كيمياء عامة 2',
    'المهارات الحياتية', 'التربية الرياضية',
    'مختبر الحاسوب واساسيات البرمجة',
    'تنمية تطوير الأعمال المستدام', 'الريادة المستدامة',
    
    # Core CS courses
    'الرياضيات المتقطعة', 'تاريخ فلسطين الحديث', 'برمجة الحاسوب',
    'مختبر تطبيقات أنظمة التشغيل', 'ثقافة اسلامية', 'جبر خطي 1',
    'لغة عبرية', 'التنميه في الوطن العربي', 'مشكلات معاصرة',
    'لغة فرنسية', 'الاتصال الفعال', 'لغة اسبانية', 'لغة المانية',
    'القانون في خدمة المجتمع', 'الديمقراطية وحقوق الإنسان',
    'تاريخ العلوم عند العرب والمسلمين', 'تركيب البيانات',
    'الحديقه المنزليه', 'اللغة التركية',
    'أخلاقيات الحاسوب وأمن المعلومات', 'المنطق الرقمي',
    'النزاهة والشفافية ومكافحة الفساد', 'الحركة الأسيرة الفلسطينية',
    'علم النفس العام',
    
    # Advanced CS
    'الاحتمالات والاحصاء', 'تصميم وتحليل الخوارزميات',
    'برمجة الكيانات', 'تنظيم وعمارة الحاسوب',
    'الاستخدام الفعال للغة الانجليزية', 'الريادة',
    'اساليب البحث العلمي', 'تحليل عددي', 'نظم قواعد البيانات',
    'لغات البرمجة', 'ادارة واقتصاد', 'مقدمة الى علم البيانات',
    'مبادىء الذكاء الاصطناعي', 'برمجة قواعد البيانات',
    'برمجة الانترنت', 'نظم التشغيل', 'هندسة البرمجيات',
    'نظرية الحوسبة', 'تعلم الالة', 'قواعد البيانات غير العلائقية',
    'مقدمة مشروع التخرج', 'شبكات الحاسوب',
    'مختبر هندسة البرمجيات', 'مشروع التخرج', 'مساق حر',
]

# ---------------------------------------------------------------------------
# English keyword lists
# ---------------------------------------------------------------------------

_EN_DEPARTMENTS: List[str] = [
    "computer science", "computer engineering", "electrical engineering",
    "civil engineering", "mechanical engineering", "business administration",
    "information technology", "architecture", "pharmacy", "nursing",
    "applied mathematics", "physics", "chemistry", "artificial intelligence",
    "data science", "information security",
]

_EN_SEMESTERS: List[str] = [
    "spring semester", "fall semester", "summer semester",
    "first semester", "second semester", "third semester",
    "fourth semester", "fifth semester", "sixth semester",
    "seventh semester", "eighth semester",
    "spring", "fall", "summer",
]

_EN_COURSES: List[str] = [
    "software engineering", "data structures", "algorithms",
    "operating systems", "database systems", "computer networks",
    "artificial intelligence", "machine learning", "natural language processing",
    "digital logic", "calculus", "linear algebra", "discrete mathematics",
    "object oriented programming", "information security",
    "mobile application development", "cloud computing",
    "data visualization", "data mining", "deep learning",
    "big data", "information retrieval", "semantic web",
    "computer graphics", "distributed systems", "compilers",
    "visual programming", "cryptography", "expert systems",
    "image processing", "bioinformatics", "human computer interaction",
]

# ---------------------------------------------------------------------------
# Arabic normalization helpers
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
_ARABIC_VARIANT_TO_CANONICAL = {
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
    'ة': 'ه',
    'ى': 'ي', 'ئ': 'ي',
    'ؤ': 'و',
    'ٔ': 'ء',
}


def _normalize_keywords(values: List[str]) -> List[str]:
    """Normalize Arabic keywords for matching."""
    normalized: List[str] = []
    seen = set()
    for val in values:
        norm = normalize_arabic(val)
        if norm and norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


# Pre-normalize all Arabic keywords
_AR_DEPARTMENTS_NORM = _normalize_keywords(_AR_DEPARTMENTS)
_AR_SEMESTERS_NORM = _normalize_keywords(_AR_SEMESTERS)
_AR_COURSES_NORM = _normalize_keywords(_AR_COURSES)


def _fast_normalize_arabic(text: str) -> str:
    """Fast Arabic normalization for keyword matching."""
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u0640]', '', text)
    for variant, canonical in _ARABIC_VARIANT_TO_CANONICAL.items():
        text = text.replace(variant, canonical)
    return text


@lru_cache(maxsize=512)
def _arabic_fuzzy_pattern(keyword: str) -> re.Pattern:
    """Create fuzzy regex pattern for Arabic keyword matching."""
    parts: List[str] = []
    for ch in keyword:
        if ch.isspace():
            parts.append(r"\s+")
            continue
        parts.append(_ARABIC_VARIANTS.get(ch, re.escape(ch)))
        parts.append(_ARABIC_NOISE)
    return re.compile("".join(parts))


def _find_arabic_span(text: str, keyword: str) -> Optional[Tuple[int, int]]:
    """Find keyword span in original Arabic text using three-tier approach."""
    if not text or not keyword:
        return None
    
    variants = _keyword_variants(keyword)
    
    for variant in variants:
        # 1. Exact match
        idx = text.find(variant)
        if idx != -1:
            return idx, idx + len(variant)
        
        # 2. Fast normalized match
        clean_text = _fast_normalize_arabic(text)
        clean_variant = _fast_normalize_arabic(variant)
        idx = clean_text.find(clean_variant)
        if idx != -1:
            return idx, idx + len(variant)
        
        # 3. Fuzzy regex
        pattern = _arabic_fuzzy_pattern(variant)
        match = pattern.search(text)
        if match:
            return match.start(), match.end()
    
    return None


def _keyword_variants(keyword: str) -> List[str]:
    """Generate Arabic keyword variants for attached prepositions."""
    if not keyword or len(keyword) <= 2:
        return [keyword] if keyword else []

    variants = [keyword]
    
    if keyword.startswith("ال") and len(keyword) > 3:
        base = keyword[2:]
        for prefix in ("لل", "بال", "كال", "وال", "فال"):
            variants.append(prefix + base)
    else:
        for prefix in ("لل", "بال", "ل", "ب", "و", "ف", "ك"):
            variants.append(prefix + keyword)
    
    return variants


def _find_keyword_in_normalized(
    normalized_text: str, keyword: str
) -> Optional[Tuple[str, int]]:
    """Find keyword or its variants in normalized Arabic text."""
    for variant in _keyword_variants(keyword):
        idx = normalized_text.find(variant)
        if idx != -1:
            return variant, idx
    return None


# ---------------------------------------------------------------------------
# Entity validation (PPU-specific)
# ---------------------------------------------------------------------------

def _validate_ppu_student_id(sid: str) -> Optional[Dict]:
    """
    Validate PPU student ID format (YYDSSS).
    
    Returns:
        Dict with parsed components if valid, None if invalid
    """
    sid = sid.strip()
    if not sid.isdigit() or len(sid) != STUDENT_ID_LENGTH:
        return None
    
    year = int(sid[:2])
    dept = sid[2]
    seq = int(sid[3:])
    
    if not (STUDENT_ID_YEAR_MIN <= year <= STUDENT_ID_YEAR_MAX):
        return None
    if dept not in STUDENT_ID_DEPT_CODES:
        return None
    if not (STUDENT_ID_SEQ_MIN <= seq <= STUDENT_ID_SEQ_MAX):
        return None
    
    enrollment_year = 2000 + year
    dept_name = PPU_DEPARTMENT_CODES.get(dept, f"Department {dept}")
    
    return {
        "student_id": sid,
        "enrollment_year": enrollment_year,
        "department_code": dept,
        "department_name": dept_name,
        "sequence_number": seq,
    }


def _validate_course_code(code: str) -> bool:
    """Validate course code format for PPU."""
    code = code.replace(" ", "").upper()
    match = re.match(r'^([A-Z]{2,4})(\d{3,4})$', code)
    if match:
        prefix = match.group(1)
        return prefix in VALID_COURSE_PREFIXES
    return False


def _validate_course_number(number: str) -> bool:
    """Validate PPU course number (4-digit internal code)."""
    number = number.strip()
    if not number.isdigit() or len(number) != 4:
        return False
    # PPU course numbers typically start with 3,4,5,8
    return number[0] in '3458'


def _is_credit_hours_context(text: str, entity_end: int) -> bool:
    """Check if a numeric entity is in credit hours context."""
    after = text[entity_end:entity_end + 30].strip()
    credit_terms = [
        'ساعة', 'ساعات', 'ساعه', 'معتمدة', 'معتمد', 'س.م',
        'credit', 'credits', 'hour', 'hours', 'hrs', 'cr',
    ]
    return any(term in after.lower() for term in credit_terms)


def _is_grade_context(text: str, entity_start: int) -> bool:
    """Check if a number appears to be a grade."""
    # Look at text before the entity
    before = text[max(0, entity_start - 20):entity_start].strip()
    grade_terms = ['العلامة', 'علامة', 'grade', 'mark', 'درجة']
    return any(term in before.lower() for term in grade_terms)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_entities(text: str, validate: bool = True) -> List[Dict]:
    """
    Extract named entities from *text* (Arabic or English).

    Args:
        text: Input text to extract entities from
        validate: If True, validate extracted entities (recommended)

    Returns:
        List[dict] — each item: {type, value, start, end, metadata?}
                    sorted by start position.
    """
    entities: List[Dict] = []
    seen_spans: Dict[Tuple[int, int], int] = {}  # span -> priority

    def _add(entity_type: EntityType, value: str, start: int, end: int, 
             metadata: Optional[Dict] = None):
        """Add entity with priority-based overlap resolution."""
        span = (start, end)
        priority = ENTITY_PRIORITY.get(entity_type, 10)
        
        if span in seen_spans:
            if priority < seen_spans[span]:
                for i, e in enumerate(entities):
                    if (e["start"], e["end"]) == span:
                        entities[i] = {
                            "type": entity_type.value,
                            "value": value,
                            "start": start,
                            "end": end,
                        }
                        if metadata:
                            entities[i]["metadata"] = metadata
                        seen_spans[span] = priority
                        break
            return
        
        seen_spans[span] = priority
        entity_dict = {
            "type": entity_type.value,
            "value": value,
            "start": start,
            "end": end,
        }
        if metadata:
            entity_dict["metadata"] = metadata
        entities.append(entity_dict)

    # --- 1. Regex patterns ---
    upper = text.upper()
    for entity_type, pattern, use_upper in _REGEX_PATTERNS:
        source = upper if use_upper else text
        for m in re.finditer(pattern, source, re.IGNORECASE):
            raw = text[m.start():m.end()].strip()
            
            if validate:
                if entity_type == EntityType.STUDENT_ID:
                    metadata = _validate_ppu_student_id(raw)
                    if metadata is None:
                        continue
                    _add(entity_type, raw, m.start(), m.end(), metadata=metadata)
                    continue
                
                if entity_type == EntityType.COURSE_CODE and not _validate_course_code(raw):
                    continue
                
                if entity_type == EntityType.COURSE_NUMBER and not _validate_course_number(raw):
                    continue
            
            _add(entity_type, raw, m.start(), m.end())

    # --- 2. Academic status patterns ---
    for m in re.finditer(_ARABIC_STATUS_PATTERN, text):
        _add(EntityType.ACADEMIC_STATUS, m.group(), m.start(), m.end())
    
    lower = text.lower()
    for m in re.finditer(_ENGLISH_STATUS_PATTERN, lower, re.IGNORECASE):
        _add(EntityType.ACADEMIC_STATUS, text[m.start():m.end()], m.start(), m.end())

    # --- 3. Semester headers ---
    for m in _SEMESTER_HEADER_PATTERN.finditer(text):
        _add(EntityType.SEMESTER, m.group(), m.start(), m.end())

    # --- 4. Arabic keyword matching ---
    if _is_arabic(text):
        normalized_text = normalize_arabic(text)
        
        # Semesters (longer phrases first)
        for kw in sorted(_AR_SEMESTERS_NORM, key=len, reverse=True):
            match = _find_keyword_in_normalized(normalized_text, kw)
            if match:
                variant, idx = match
                span = _find_arabic_span(text, variant)
                if span:
                    _add(EntityType.SEMESTER, text[span[0]:span[1]], span[0], span[1])
                else:
                    _add(EntityType.SEMESTER, variant, idx, idx + len(variant))

        # Departments
        for kw in sorted(_AR_DEPARTMENTS_NORM, key=len, reverse=True):
            match = _find_keyword_in_normalized(normalized_text, kw)
            if match:
                variant, idx = match
                span = _find_arabic_span(text, variant)
                if span:
                    _add(EntityType.DEPARTMENT, text[span[0]:span[1]], span[0], span[1])
                else:
                    _add(EntityType.DEPARTMENT, variant, idx, idx + len(variant))

        # Courses
        for kw in sorted(_AR_COURSES_NORM, key=len, reverse=True):
            match = _find_keyword_in_normalized(normalized_text, kw)
            if match:
                variant, idx = match
                span = _find_arabic_span(text, variant)
                if span:
                    _add(EntityType.COURSE_NAME, text[span[0]:span[1]], span[0], span[1])
                else:
                    _add(EntityType.COURSE_NAME, variant, idx, idx + len(variant))

        # Arabic person names
        for pattern in _AR_PERSON_PATTERNS:
            for m in re.finditer(pattern, text):
                _add(EntityType.PERSON, m.group(1), m.start(1), m.end(1))

    # --- 5. English keyword matching ---
    for kw in sorted(_EN_SEMESTERS, key=len, reverse=True):
        idx = lower.find(kw)
        if idx != -1:
            _add(EntityType.SEMESTER, text[idx:idx + len(kw)], idx, idx + len(kw))

    for kw in sorted(_EN_DEPARTMENTS, key=len, reverse=True):
        idx = lower.find(kw)
        if idx != -1:
            _add(EntityType.DEPARTMENT, text[idx:idx + len(kw)], idx, idx + len(kw))

    for kw in sorted(_EN_COURSES, key=len, reverse=True):
        idx = lower.find(kw)
        if idx != -1:
            _add(EntityType.COURSE_NAME, text[idx:idx + len(kw)], idx, idx + len(kw))

    # --- 6. spaCy NER ---
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        spacy_map = {
            "DATE": EntityType.DATE,
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.DEPARTMENT,
            "MONEY": EntityType.AMOUNT,
        }
        for ent in doc.ents:
            mapped = spacy_map.get(ent.label_)
            if mapped:
                _add(mapped, ent.text, ent.start_char, ent.end_char)

    # --- 7. Date patterns ---
    for m in re.finditer(_DATE_REGEX, text):
        _add(EntityType.DATE, m.group(), m.start(), m.end())

    if _is_arabic(text):
        for m in re.finditer(_YEAR_REGEX, text):
            _add(EntityType.DATE, m.group(), m.start(), m.end())

    # --- 8. Post-processing: context disambiguation ---
    entities = _disambiguate_entities(entities, text)

    # Sort by start position
    entities.sort(key=lambda e: (e["start"], -e["end"]))
    
    return entities


def _disambiguate_entities(entities: List[Dict], text: str) -> List[Dict]:
    """Resolve ambiguous entities using context."""
    for i, entity in enumerate(entities):
        if entity["type"] == EntityType.AMOUNT.value:
            if _is_credit_hours_context(text, entity["end"]):
                entity["type"] = EntityType.CREDIT_HOURS.value
        
        elif entity["type"] == EntityType.CREDIT_HOURS.value:
            if _is_grade_context(text, entity["start"]):
                # Could be a grade, not credit hours
                pass
        
        elif entity["type"] == EntityType.DATE.value:
            if entity["value"].isdigit():
                before = text[max(0, entity["start"] - 3):entity["start"]]
                after = text[entity["end"]:entity["end"] + 3]
                if before.isdigit() or after.isdigit():
                    entities[i] = None
        
        elif entity["type"] == EntityType.COURSE_NUMBER.value:
            # Course numbers could also be student IDs if 6 digits
            val = entity["value"].strip()
            if len(val) == 6 and val.isdigit():
                # Check if context suggests student
                around = text[max(0, entity["start"]-20):entity["end"]+20].lower()
                if any(term in around for term in ['student', 'طالب', 'رقم', 'id']):
                    entity["type"] = EntityType.STUDENT_ID.value
    
    return [e for e in entities if e is not None]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def entities_to_dict(entities: List[Dict]) -> Dict[str, List[str]]:
    """Flatten entity list → {type: [value, ...]}."""
    result: Dict[str, List[str]] = {}
    for ent in entities:
        entity_type = ent["type"]
        if entity_type not in result:
            result[entity_type] = []
        if ent["value"] not in result[entity_type]:
            result[entity_type].append(ent["value"])
    return result


def extract_entities_batch(texts: List[str], validate: bool = True) -> List[List[Dict]]:
    """Extract entities from multiple texts efficiently."""
    _get_nlp()
    return [extract_entities(text, validate=validate) for text in texts]


def parse_student_id(sid: str) -> Optional[Dict]:
    """
    Parse a PPU student ID into its components.
    
    Args:
        sid: Student ID string (e.g., "221100")
        
    Returns:
        Dict with parsed components or None if invalid
    """
    return _validate_ppu_student_id(sid)


def get_department_from_student_id(sid: str) -> Optional[str]:
    """
    Get department name from PPU student ID.
    
    Args:
        sid: Student ID string (e.g., "221100")
        
    Returns:
        Department name or None if invalid
    """
    result = _validate_ppu_student_id(sid)
    if result:
        return result["department_name"]
    return None


def debug_extraction(text: str) -> Dict:
    """Debug helper to show entity extraction details."""
    entities = extract_entities(text)
    
    return {
        "text": text,
        "language": "arabic" if _is_arabic(text) else "english",
        "entity_count": len(entities),
        "entities": entities,
        "by_type": entities_to_dict(entities),
        "unique_types": list(set(e["type"] for e in entities)),
        "student_ids": [
            e for e in entities 
            if e["type"] == EntityType.STUDENT_ID.value
        ],
        "course_codes": [
            e for e in entities 
            if e["type"] in (EntityType.COURSE_CODE.value, EntityType.COURSE_NUMBER.value)
        ],
    }