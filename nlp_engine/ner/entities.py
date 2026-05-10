"""
Entity type definitions for PPU NER.

Based on actual PPU data structures:
- Student IDs: 221100 format (2 digit year, 1 digit department, 3 digit sequence)
- Course codes: CS331, IT377, etc. (2-4 letters + 3 digits)
- Course numbers: 4-digit internal codes (4447, 4620, etc.)
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple


class EntityType(str, Enum):
    """Canonical entity types recognized by the PPU NER extractor."""

    COURSE_NAME = "COURSE_NAME"          # e.g. "Software Engineering", "NLP"
    COURSE_CODE = "COURSE_CODE"          # e.g. "CS401", "ENGL101"
    COURSE_NUMBER = "COURSE_NUMBER"      # e.g. "4447", "4620" (internal PPU code)
    STUDENT_ID  = "STUDENT_ID"           # e.g. "221100" (YYDSSS format)
    DEPARTMENT  = "DEPARTMENT"           # e.g. "Computer Engineering"
    DATE        = "DATE"                 # e.g. "May 12", "next Monday"
    SEMESTER    = "SEMESTER"             # e.g. "spring 2025", "الفصل الثاني"
    PERSON      = "PERSON"               # staff / instructor names
    AMOUNT      = "AMOUNT"               # fee amounts in currency
    CREDIT_HOURS = "CREDIT_HOURS"        # credit hours (س.م. / ساعة معتمدة)
    GRADE       = "GRADE"                # student grades (علامة)
    ACADEMIC_STATUS = "ACADEMIC_STATUS"  # ناجح, راسب, etc.


# Human-readable descriptions
ENTITY_DESCRIPTIONS: Dict[str, str] = {
    EntityType.COURSE_NAME: "Name of an academic course",
    EntityType.COURSE_CODE: "Alphanumeric course code (e.g. CS401)",
    EntityType.COURSE_NUMBER: "Internal PPU 4-digit course number",
    EntityType.STUDENT_ID:  "PPU student ID (YYDSSS format: 221100)",
    EntityType.DEPARTMENT:  "University department or faculty",
    EntityType.DATE:        "A calendar date or relative time expression",
    EntityType.SEMESTER:    "Academic semester reference",
    EntityType.PERSON:      "Name of a person (staff, instructor, etc.)",
    EntityType.AMOUNT:      "Monetary amount",
    EntityType.CREDIT_HOURS: "Number of credit hours for a course",
    EntityType.GRADE:       "Course grade (numeric or P/F)",
    EntityType.ACADEMIC_STATUS: "Student academic status (ناجح, راسب, etc.)",
}

# Priority ordering for entity disambiguation (lower = higher priority)
ENTITY_PRIORITY: Dict[EntityType, int] = {
    EntityType.COURSE_CODE: 1,         # Very specific: CS331
    EntityType.COURSE_NUMBER: 1,       # Very specific: 4-digit codes
    EntityType.STUDENT_ID: 1,          # Very specific: 6-digit IDs
    EntityType.DATE: 2,                # Specific date formats
    EntityType.CREDIT_HOURS: 3,        # Specific credit context
    EntityType.GRADE: 4,               # Numeric grades in context
    EntityType.PERSON: 5,              # Names with titles
    EntityType.ACADEMIC_STATUS: 6,     # Status keywords
    EntityType.SEMESTER: 7,            # Semester phrases
    EntityType.COURSE_NAME: 8,         # Course name phrases
    EntityType.DEPARTMENT: 8,          # Department phrases
    EntityType.AMOUNT: 9,              # Least specific number pattern
}

# Valid PPU department codes (the middle digit in student ID)
PPU_DEPARTMENT_CODES: Dict[str, str] = {
    '1': 'Computer Science',
    '2': 'Computer Engineering',
    '3': 'Information Technology',
    '4': 'Electrical Engineering',
    '5': 'Mechanical Engineering',
    '6': 'Civil Engineering',
    '7': 'Business Administration',
    '8': 'Architecture',
    '9': 'Pharmacy',
}

# Valid PPU department prefixes for course code validation
VALID_COURSE_PREFIXES = {
    'CS', 'CIS', 'IT', 'ENGL', 'MATH', 'PHYS', 'CHEM', 'BIO',
    'BUS', 'NURS', 'PHAR', 'ARCH', 'CE', 'EE', 'ME', 'IE',
    'SE', 'AI', 'DS', 'NET', 'SEC', 'SC', 'GE', 'AB', 'ARE', 'AC',
}

# PPU student ID validation
STUDENT_ID_LENGTH = 6  # YYDSSS format exactly 6 digits
STUDENT_ID_YEAR_MIN = 10  # Year 2010 and later
STUDENT_ID_YEAR_MAX = 30  # Up to 2030
STUDENT_ID_DEPT_CODES = set('123456789')  # Valid department digits
STUDENT_ID_SEQ_MIN = 0
STUDENT_ID_SEQ_MAX = 999

# Academic status terms in Arabic
ARABIC_ACADEMIC_STATUS = [
    'ناجح', 'راسب', 'غير مجتاز', 'مسجل حالياً', 'لم يسجل',
    'منسحب', 'مؤجل', 'محول', 'مفصول',
]

# English academic status terms
ENGLISH_ACADEMIC_STATUS = [
    'passed', 'failed', 'not passed', 'currently enrolled',
    'not registered', 'withdrawn', 'deferred', 'transferred',
    'dismissed', 'pass', 'fail', 'incomplete',
]

# Grade patterns
GRADE_VALUES = {
    # Numeric grades
    **{str(i): i for i in range(0, 101)},
    # Letter grades
    'A+': 4.0, 'A': 4.0, 'A-': 3.7,
    'B+': 3.3, 'B': 3.0, 'B-': 2.7,
    'C+': 2.3, 'C': 2.0, 'C-': 1.7,
    'D+': 1.3, 'D': 1.0, 'D-': 0.7,
    'F': 0.0,
    # Special
    'P': -1,   # Pass
    'NP': -2,  # Not Pass
    'I': -3,   # Incomplete
}