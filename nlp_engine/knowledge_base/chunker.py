"""
Text chunking strategies for multilingual documents (Arabic + English).

Philosophy
----------
Chunks should reflect *meaning boundaries*, not fixed character counts.
A paragraph about "admission requirements" should stay together even if
it's 900 chars; a one-liner navigation label should be dropped entirely.

Strategies provided
-------------------
1. chunk_by_characters   — original fast fallback (kept for compatibility)
2. chunk_by_sentences    — sentence-grouped (original, kept for compatibility)
3. chunk_semantic        — original semantic (kept for compatibility)
4. chunk_structural      — NEW: paragraph/section-aware, respects document
                           structure (headers → paragraphs → sentences)
5. chunk_contextual      — NEW: structural + sliding context window that
                           prepends the nearest heading to every chunk so
                           retrieval always knows "where" the chunk came from
6. chunk_document        — dispatcher (now defaults to "contextual")
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

_MIN_CHUNK_CHARS  = 80    # drop chunks shorter than this
_MAX_CHUNK_CHARS  = 1200  # hard ceiling — split even mid-paragraph if exceeded
_TARGET_CHUNK     = 700   # preferred chunk size

# Heading patterns (Arabic + English, numbered or decorated)
_HEADING_RE = re.compile(
    r"^(?:"
    # Arabic section markers
    r"(?:القسم|الفصل|المادة|البند|أولاً|ثانياً|ثالثاً|رابعاً|خامساً"
    r"|أ\.|ب\.|ج\.|د\.)\s*.+"
    r"|"
    # English numbered headings: "1.", "1.2", "A.", "Chapter 1"
    r"(?:Chapter|Section|Article|Part|Appendix)\s+\w+"
    r"|"
    r"(?:\d+\.)+\s+\S.{0,60}"          # "1.2.3 Some heading"
    r"|"
    r"[A-Z][A-Z\s]{4,50}$"             # ALL CAPS SHORT LINE
    r")",
    re.MULTILINE | re.UNICODE,
)

# Sentence boundaries — English + Arabic punctuation
_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?؟؛])\s+"
    r"|(?<=[\n])\s*(?=\S)",
    re.UNICODE,
)

# Blank-line paragraph separator
_PARA_SEP = re.compile(r"\n\s*\n+", re.UNICODE)

# Lines that are pure noise (navigation, page numbers, repeated labels)
_NOISE_LINE = re.compile(
    r"^(?:"
    r"\d+\s*$"                          # lone page number
    r"|[|·•\-–—=_]{3,}"                # decorative separators
    r"|(?:home|menu|search|login|contact|sitemap|copyright|©)"
    r"|(?:الرئيسية|القائمة|بحث|دخول|اتصل|خريطة الموقع)"
    r")\s*$",
    re.IGNORECASE | re.UNICODE,
)

# Repeated whitespace normalizer
_WS = re.compile(r"[ \t]+", re.UNICODE)


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextBlock:
    """A single logical block extracted from the document."""
    text:       str
    kind:       str          # "heading" | "paragraph" | "list" | "table_row"
    heading:    str = ""     # nearest preceding heading (for context injection)
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


# ══════════════════════════════════════════════════════════════════════════════
# Pre-processing
# ══════════════════════════════════════════════════════════════════════════════

def _clean_line(line: str) -> str:
    """Normalize whitespace, strip, return empty string if noise."""
    line = _WS.sub(" ", line).strip()
    if _NOISE_LINE.match(line):
        return ""
    return line


def _is_heading(line: str) -> bool:
    """Heuristic: is this line a section heading?"""
    if not line or len(line) > 120:
        return False
    if _HEADING_RE.match(line):
        return True
    # Short line ending without normal sentence punctuation → likely a heading
    if len(line) < 80 and not re.search(r"[.،,;]$", line):
        # And starts with uppercase or Arabic letter
        if re.match(r"^[\u0600-\u06FF\u0041-\u005A]", line):
            return True
    return False


def _is_list_item(line: str) -> bool:
    return bool(re.match(r"^\s*[-•*·◦▪▸]\s+\S", line)
                or re.match(r"^\s*\d+[.)]\s+\S", line)
                or re.match(r"^\s*[\u0661-\u0669][.)]\s+\S", line))  # Arabic numerals


# ══════════════════════════════════════════════════════════════════════════════
# Block extractor
# ══════════════════════════════════════════════════════════════════════════════

def _extract_blocks(text: str) -> List[TextBlock]:
    """
    Split document text into typed TextBlocks.

    Order of priority:
      1. Blank-line paragraph boundaries
      2. Heading detection within paragraphs
      3. List item grouping
    """
    blocks: List[TextBlock] = []
    current_heading = ""

    # Split on blank lines first
    raw_paras = _PARA_SEP.split(text)

    for para in raw_paras:
        lines = [_clean_line(l) for l in para.splitlines()]
        lines = [l for l in lines if l]   # drop noise/empty

        if not lines:
            continue

        # Single-line paragraph — heading or standalone sentence
        if len(lines) == 1:
            line = lines[0]
            if _is_heading(line):
                current_heading = line
                blocks.append(TextBlock(text=line, kind="heading",
                                        heading=current_heading))
            else:
                blocks.append(TextBlock(text=line, kind="paragraph",
                                        heading=current_heading))
            continue

        # Multi-line paragraph — scan for embedded headings
        buffer: List[str] = []
        list_buffer: List[str] = []

        def flush_buffer():
            if buffer:
                merged = " ".join(buffer)
                blocks.append(TextBlock(text=merged, kind="paragraph",
                                        heading=current_heading))
                buffer.clear()

        def flush_list():
            if list_buffer:
                merged = "\n".join(list_buffer)
                blocks.append(TextBlock(text=merged, kind="list",
                                        heading=current_heading))
                list_buffer.clear()

        for line in lines:
            if _is_heading(line):
                flush_buffer()
                flush_list()
                current_heading = line
                blocks.append(TextBlock(text=line, kind="heading",
                                        heading=current_heading))
            elif _is_list_item(line):
                flush_buffer()
                list_buffer.append(line)
            else:
                flush_list()
                buffer.append(line)

        flush_buffer()
        flush_list()

    return blocks


# ══════════════════════════════════════════════════════════════════════════════
# Block → chunk assembly
# ══════════════════════════════════════════════════════════════════════════════

def _split_oversized(text: str, max_size: int, min_size: int) -> List[str]:
    """
    Break a single block that exceeds max_size at sentence boundaries.
    Used only when a paragraph is too long to keep whole.
    """
    sentences = _SENT_BOUNDARY.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        slen = len(sent)
        if current_len + slen > max_size and current:
            chunk = " ".join(current)
            if len(chunk) >= min_size:
                parts.append(chunk)
            current = []
            current_len = 0
        current.append(sent)
        current_len += slen

    if current:
        chunk = " ".join(current)
        if len(chunk) >= min_size:
            parts.append(chunk)

    return parts or [text[:max_size]]


def _assemble_chunks(
    blocks: List[TextBlock],
    target: int,
    max_size: int,
    min_size: int,
    inject_heading: bool,
) -> List[str]:
    """
    Greedily merge TextBlocks into chunks respecting these rules:

    - Headings are never merged into a preceding chunk; they start a new one
      (and their text is injected as context prefix into subsequent chunks).
    - Paragraphs that fit within max_size are kept whole.
    - Paragraphs that exceed max_size are split at sentence boundaries.
    - Adjacent small blocks are merged until the target size is reached.
    - List blocks are kept together (they're already short).
    """
    chunks: List[str] = []
    acc_texts: List[str] = []
    acc_len   = 0
    ctx_heading = ""   # heading to prepend when inject_heading=True

    def flush(force_heading: str = ""):
        nonlocal acc_len
        if not acc_texts:
            return
        body = " ".join(acc_texts).strip()
        if len(body) < min_size:
            acc_texts.clear()
            acc_len = 0
            return
        if inject_heading and (force_heading or ctx_heading):
            h = force_heading or ctx_heading
            # Only prepend if heading text isn't already the start of body
            if not body.startswith(h):
                body = f"{h}\n{body}"
        chunks.append(body)
        acc_texts.clear()
        acc_len = 0

    for block in blocks:
        # Headings: flush current accumulator, then start fresh context
        if block.kind == "heading":
            flush(force_heading="")
            ctx_heading = block.text
            # Don't add the heading as a standalone chunk —
            # it will be prepended to the next real chunk instead.
            continue

        text = block.text

        # Block too large — split it, then treat each piece as a mini-block
        if len(text) > max_size:
            flush()
            for part in _split_oversized(text, max_size, min_size):
                if inject_heading and ctx_heading and not part.startswith(ctx_heading):
                    chunks.append(f"{ctx_heading}\n{part}")
                else:
                    chunks.append(part)
            continue

        # Block fits — accumulate
        if acc_len + len(text) <= target:
            acc_texts.append(text)
            acc_len += len(text)
        else:
            # Adding this block would exceed target
            if acc_len >= min_size:
                flush()
            acc_texts.append(text)
            acc_len = len(text)

        # Hard ceiling — flush immediately
        if acc_len >= max_size:
            flush()

    flush()
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Public strategies
# ══════════════════════════════════════════════════════════════════════════════

def chunk_structural(
    text: str,
    target_size: int = _TARGET_CHUNK,
    max_size: int = _MAX_CHUNK_CHARS,
    min_size: int = _MIN_CHUNK_CHARS,
) -> List[str]:
    """
    Structure-aware chunking.

    Respects paragraph and section boundaries extracted from the document.
    Adjacent small paragraphs are merged until *target_size* is reached.
    Large paragraphs are split at sentence boundaries only if they exceed
    *max_size* — they are never cut at arbitrary character positions.

    Best for: well-structured university pages (admin, academics, about).
    """
    blocks = _extract_blocks(text)
    return _assemble_chunks(
        blocks,
        target=target_size,
        max_size=max_size,
        min_size=min_size,
        inject_heading=False,
    )


def chunk_contextual(
    text: str,
    target_size: int = _TARGET_CHUNK,
    max_size: int = _MAX_CHUNK_CHARS,
    min_size: int = _MIN_CHUNK_CHARS,
) -> List[str]:
    """
    Contextual chunking — structural + heading injection.

    Every chunk is prefixed with its nearest section heading so that
    retrieval always carries the "where am I in the document" signal.

    Example output chunk:
        شروط القبول والتسجيل          ← injected heading
        يجب على الطالب تقديم...       ← paragraph content

    This dramatically improves RAG retrieval precision because the
    embedding captures both the topic label and the content together.

    Best for: all PPU content — recommended default.
    """
    blocks = _extract_blocks(text)
    return _assemble_chunks(
        blocks,
        target=target_size,
        max_size=max_size,
        min_size=min_size,
        inject_heading=True,
    )


# ── Originals kept for backward compatibility ──────────────────────────────

_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?؟؛])\s+|'
    r'(?<=[။။])\s+|'
    r'\n+',
    re.UNICODE,
)


def chunk_by_characters(
    text: str,
    chunk_size: int = _TARGET_CHUNK,
    overlap: int = 120,
) -> List[str]:
    """Original character-level chunker (kept for compatibility)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            window_start = int(chunk_size * 0.8)
            window = chunk[window_start:]
            matches = list(_SENTENCE_BOUNDARY.finditer(window))
            if matches:
                break_point = window_start + matches[-1].start()
                end = start + break_point + 1
                chunk = text[start:end]
        chunk = chunk.strip()
        if len(chunk) >= _MIN_CHUNK_CHARS:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def chunk_by_sentences(
    text: str,
    max_sentences: int = 5,
    min_sentences: int = 2,
) -> List[str]:
    """Original sentence-grouped chunker (kept for compatibility)."""
    sentences = _SENTENCE_BOUNDARY.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence)
        if len(current_chunk) >= max_sentences or current_length >= 1000:
            chunk = " ".join(current_chunk)
            if len(chunk) >= _MIN_CHUNK_CHARS:
                chunks.append(chunk)
            current_chunk, current_length = [], 0
    if len(current_chunk) >= min_sentences:
        chunk = " ".join(current_chunk)
        if len(chunk) >= _MIN_CHUNK_CHARS:
            chunks.append(chunk)
    return chunks


def chunk_semantic(
    text: str,
    target_size: int = _TARGET_CHUNK,
    max_size: int = _MAX_CHUNK_CHARS,
) -> List[str]:
    """Original semantic chunker (kept for compatibility)."""
    sentences = _SENTENCE_BOUNDARY.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, current_chunk, current_size = [], [], 0
    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > max_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_size = [], 0
        current_chunk.append(sentence)
        current_size += sentence_size
        if current_size >= target_size:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_size = [], 0
    if current_chunk:
        chunk = " ".join(current_chunk)
        if len(chunk) >= _MIN_CHUNK_CHARS:
            chunks.append(chunk)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def chunk_document(
    text: str,
    strategy: str = "contextual",
    chunk_size: int = _TARGET_CHUNK,
    overlap: int = 120,
) -> List[str]:
    """
    Chunk text using the specified strategy.

    Strategies
    ----------
    contextual  (default) — structural + heading injection. Best for RAG.
    structural            — paragraph/section-aware, no heading prefix.
    char                  — fixed character windows with overlap.
    sentence              — sentence-count windows.
    semantic              — sentence boundaries + size limits (original).

    Parameters
    ----------
    text        : Input document text
    chunk_size  : Target chunk size in characters (contextual/structural/semantic)
    overlap     : Overlap in characters (char strategy only)
    """
    if not text or not text.strip():
        return []

    strategies = {
        "contextual": lambda: chunk_contextual(text, target_size=chunk_size),
        "structural": lambda: chunk_structural(text, target_size=chunk_size),
        "char":       lambda: chunk_by_characters(text, chunk_size, overlap),
        "sentence":   lambda: chunk_by_sentences(text),
        "semantic":   lambda: chunk_semantic(text, chunk_size),
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {', '.join(strategies)}"
        )

    chunks = strategies[strategy]()

    logger.debug(
        "chunk_document: strategy=%s, input=%d chars → %d chunks",
        strategy, len(text), len(chunks),
    )
    return chunks