"""
Text chunking strategies for multilingual documents.

Provides multiple chunking approaches:
  • Character chunking with overlap (current)
  • Sentence-aware chunking
  • Semantic chunking (sentence boundary + size limits)
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Sentence boundary patterns (multilingual)
_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?؟؛])\s+|'  # Standard punctuation + Arabic
    r'(?<=[။။])\s+|'      # Additional scripts
    r'\n+'                  # Newlines
)

# Minimum chunk size to avoid tiny fragments
_MIN_CHUNK_SIZE = 100


def chunk_by_characters(
    text: str,
    chunk_size: int = 700,
    overlap: int = 120,
) -> List[str]:
    """
    Split text into overlapping character-level chunks.
    Basic approach - fast but may break mid-sentence.
    
    Args:
        text: Input text
        chunk_size: Characters per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary in last 20% of chunk
        if end < len(text):
            window_start = int(chunk_size * 0.8)
            window = chunk[window_start:]
            
            # Find last sentence boundary in window
            matches = list(_SENTENCE_BOUNDARY.finditer(window))
            if matches:
                last_match = matches[-1]
                break_point = window_start + last_match.start()
                end = start + break_point + 1
                chunk = text[start:end]
        
        chunk = chunk.strip()
        if len(chunk) >= _MIN_CHUNK_SIZE:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def chunk_by_sentences(
    text: str,
    max_sentences: int = 5,
    min_sentences: int = 2,
) -> List[str]:
    """
    Split text into sentence-based chunks.
    Better for maintaining semantic coherence.
    
    Args:
        text: Input text
        max_sentences: Maximum sentences per chunk
        min_sentences: Minimum sentences per chunk
        
    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = _SENTENCE_BOUNDARY.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence)
        
        # Create chunk when we have enough sentences or length
        if len(current_chunk) >= max_sentences or current_length >= 1000:
            chunk = " ".join(current_chunk)
            if len(chunk) >= _MIN_CHUNK_SIZE:
                chunks.append(chunk)
            current_chunk = []
            current_length = 0
    
    # Don't forget remaining sentences
    if len(current_chunk) >= min_sentences:
        chunk = " ".join(current_chunk)
        if len(chunk) >= _MIN_CHUNK_SIZE:
            chunks.append(chunk)
    
    return chunks


def chunk_semantic(
    text: str,
    target_size: int = 700,
    max_size: int = 1000,
) -> List[str]:
    """
    Semantic chunking: sentence boundaries + size constraints.
    
    Tries to keep semantically related content together while
    respecting size limits for embedding models.
    
    Args:
        text: Input text
        target_size: Target chunk size in characters
        max_size: Maximum chunk size
        
    Returns:
        List of text chunks
    """
    sentences = _SENTENCE_BOUNDARY.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence exceeds max, start new chunk
        if current_size + sentence_size > max_size and current_chunk:
            chunk = " ".join(current_chunk)
            chunks.append(chunk)
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
        
        # If we've reached target size, create chunk
        if current_size >= target_size:
            chunk = " ".join(current_chunk)
            chunks.append(chunk)
            current_chunk = []
            current_size = 0
    
    # Add remaining
    if current_chunk:
        chunk = " ".join(current_chunk)
        if len(chunk) >= _MIN_CHUNK_SIZE:
            chunks.append(chunk)
    
    return chunks


def chunk_document(
    text: str,
    strategy: str = "semantic",
    chunk_size: int = 700,
    overlap: int = 120,
) -> List[str]:
    """
    Chunk text using specified strategy.
    
    Args:
        text: Input text
        strategy: 'char', 'sentence', or 'semantic'
        chunk_size: Target size (for char and semantic)
        overlap: Overlap (for char only)
        
    Returns:
        List of text chunks
    """
    if strategy == "char":
        return chunk_by_characters(text, chunk_size, overlap)
    elif strategy == "sentence":
        return chunk_by_sentences(text)
    elif strategy == "semantic":
        return chunk_semantic(text, chunk_size)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")