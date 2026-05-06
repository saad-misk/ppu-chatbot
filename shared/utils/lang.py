"""
shared.utils.lang — language-detection utilities used across the entire codebase.

This module is the single source of truth for `is_arabic`.  All modules that
previously defined their own copy of the regex should import from here.
"""
from __future__ import annotations

import re

# Unicode block for Arabic script (Basic Arabic + Arabic Supplement + Presentation Forms)
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def is_arabic(text: str) -> bool:
    """Return True if *text* contains any Arabic Unicode characters."""
    if not text:
        return False
    return bool(_ARABIC_RE.search(text))
