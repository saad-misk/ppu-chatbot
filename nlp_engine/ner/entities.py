"""
Entity type definitions for PPU NER.

These constants are used across the extractor and any downstream consumers
so that entity type strings are never magic literals.
"""
from enum import Enum


class EntityType(str, Enum):
    """Canonical entity types recognized by the PPU NER extractor."""

    COURSE_NAME = "COURSE_NAME"          # e.g. "Software Engineering", "NLP"
    COURSE_CODE = "COURSE_CODE"          # e.g. "CS401", "ENGL101"
    STUDENT_ID  = "STUDENT_ID"           # e.g. "1202345", "120-2345"
    DEPARTMENT  = "DEPARTMENT"           # e.g. "Computer Engineering"
    DATE        = "DATE"                 # e.g. "May 12", "next Monday"
    SEMESTER    = "SEMESTER"             # e.g. "spring 2025", "second semester"
    PERSON      = "PERSON"               # staff / instructor names (from spaCy)
    AMOUNT      = "AMOUNT"               # fee / credit-hour amounts


# Human-readable descriptions (useful in admin UI / debugging)
ENTITY_DESCRIPTIONS: dict[str, str] = {
    EntityType.COURSE_NAME: "Name of an academic course",
    EntityType.COURSE_CODE: "Alphanumeric course code (e.g. CS401)",
    EntityType.STUDENT_ID:  "PPU student ID number",
    EntityType.DEPARTMENT:  "University department or faculty",
    EntityType.DATE:        "A calendar date or relative time expression",
    EntityType.SEMESTER:    "Academic semester reference",
    EntityType.PERSON:      "Name of a person (staff, instructor, etc.)",
    EntityType.AMOUNT:      "Monetary amount or credit-hour count",
}
