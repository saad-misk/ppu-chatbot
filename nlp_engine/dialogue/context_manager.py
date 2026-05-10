"""
Conversation context manager — formats and trims multi-turn history
for injection into the RAG generator prompt with token-aware trimming.

Features:
  • Fixed-size turn window
  • Token counting and limit enforcement
  • System message support
  • Message truncation for very long messages
  • Flexible formatting for different LLM APIs

Usage
-----
    from nlp_engine.dialogue.context_manager import ContextManager

    ctx = ContextManager(max_turns=6, max_tokens=2000)
    ctx.set_system_message("You are a helpful PPU assistant.")
    ctx.add_turn("user", "What are the fees?")
    ctx.add_turn("assistant", "The fees are...")
    messages = ctx.get_messages()  # List[{"role": ..., "content": ...}]
"""
from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ContextManager:
    """
    In-memory store for a single session's conversation history.
    
    Manages conversation turns with:
    - Maximum turn count
    - Token-aware trimming
    - System message support
    - Message truncation for long individual messages
    
    The gateway persists full history in PostgreSQL; this class manages
    the in-flight window passed to the NLP engine on each request.
    
    Args:
        max_turns: Maximum number of turns to keep (user + assistant combined)
        max_tokens: Maximum total tokens (approximate). 0 = no limit.
        max_message_length: Maximum characters per message before truncation
        preserve_system: Whether to always keep system message at start
    """

    def __init__(
        self,
        max_turns: int = 6,
        max_tokens: int = 2000,
        max_message_length: int = 1000,
        preserve_system: bool = True,
    ):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_message_length = max_message_length
        self.preserve_system = preserve_system
        
        self._history: Deque[Dict[str, str]] = deque(maxlen=max_turns * 2)  # Extra room
        self._system_message: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def set_system_message(self, content: str) -> None:
        """
        Set the system message for role-setting.
        
        Args:
            content: System instruction (e.g., "You are a helpful PPU assistant")
        """
        self._system_message = {"role": "system", "content": content.strip()}

    def add_turn(self, role: str, content: str) -> None:
        """
        Append a single turn.
        
        Args:
            role: 'user', 'assistant', or 'system'
            content: Message content (will be trimmed if too long)
        """
        assert role in ("user", "assistant", "system"), f"Unknown role: {role}"
        
        # Truncate very long messages
        if len(content) > self.max_message_length * 2:
            logger.warning(
                f"Message too long ({len(content)} chars), truncating to {self.max_message_length}"
            )
            content = content[:self.max_message_length] + "... [truncated]"
        
        turn = {"role": role, "content": content.strip()}
        
        if role == "system":
            self.set_system_message(content)
            return
        
        self._history.append(turn)
        
        # Trim to max turns if needed
        while len(self._history) > self.max_turns:
            self._history.popleft()

    def load_history(self, history: List[Dict[str, str]]) -> None:
        """
        Populate from an external history list (e.g., sent by the gateway).
        Only the last max_turns items are kept.
        
        Args:
            history: List of {"role": ..., "content": ...} dictionaries
        """
        self._history.clear()
        
        for turn in history[-self.max_turns:]:
            if turn.get("role") == "system":
                self.set_system_message(turn["content"])
            else:
                self._history.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", "").strip(),
                })

    def clear(self) -> None:
        """Clear all history (keeps system message)."""
        self._history.clear()

    def clear_all(self) -> None:
        """Clear everything including system message."""
        self._history.clear()
        self._system_message = None

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough token estimation (4 chars ≈ 1 token for mixed Arabic/English).
        
        This is a simple heuristic. For production, use a proper tokenizer
        like tiktoken (OpenAI) or the model's native tokenizer.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Arabic characters are typically 1-2 tokens each in multilingual models
        # English is roughly 4 chars per token
        # This gives a rough upper bound
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(text) - arabic_chars
        
        # Conservative estimate
        return (arabic_chars // 1) + (latin_chars // 4)

    def get_total_tokens(self) -> int:
        """Estimate total tokens in current history (including system message)."""
        total = 0
        if self._system_message:
            total += self.estimate_tokens(self._system_message["content"])
        for turn in self._history:
            total += self.estimate_tokens(turn["content"])
        return total

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_messages(self, trim_to_tokens: bool = True) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM API, with optional token trimming.
        
        Args:
            trim_to_tokens: If True, remove oldest messages until under max_tokens
            
        Returns:
            List of message dictionaries ready for LLM API
        """
        messages: List[Dict[str, str]] = []
        
        # Always include system message if set
        if self._system_message:
            messages.append(self._system_message)
        
        # Add history
        messages.extend(list(self._history))
        
        # Trim by tokens if needed
        if trim_to_tokens and self.max_tokens > 0:
            messages = self._trim_to_token_limit(messages)
        
        return messages

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current history as a plain list (without system message)."""
        return list(self._history)

    def get_last_user_message(self) -> Optional[str]:
        """Return the most recent user message, or None."""
        for turn in reversed(self._history):
            if turn["role"] == "user":
                return turn["content"]
        return None

    def get_recent_context(self, n_turns: int = 2) -> str:
        """
        Get recent conversation as a formatted string for RAG context.
        
        Args:
            n_turns: Number of recent turns to include
            
        Returns:
            Formatted string of recent conversation
        """
        recent = list(self._history)[-n_turns:]
        lines = []
        for turn in recent:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content']}")
        return "\n".join(lines)

    def format_as_messages(self) -> str:
        """Format as a string for debugging."""
        lines = []
        if self._system_message:
            lines.append(f"System: {self._system_message['content']}")
        for turn in self._history:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim_to_token_limit(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Remove oldest non-system messages until under token limit.
        
        Args:
            messages: Full message list
            
        Returns:
            Trimmed message list
        """
        # Calculate current tokens
        total_tokens = sum(self.estimate_tokens(m["content"]) for m in messages)
        
        if total_tokens <= self.max_tokens:
            return messages
        
        # Build result: keep system message, trim history
        result = []
        if self._system_message:
            result.append(self._system_message)
            total_tokens -= self.estimate_tokens(self._system_message["content"])
        
        # Add history from newest to oldest while under limit
        history_messages = [m for m in messages if m.get("role") != "system"]
        trimmed_history = []
        
        for msg in reversed(history_messages):
            msg_tokens = self.estimate_tokens(msg["content"])
            if total_tokens + msg_tokens <= self.max_tokens:
                trimmed_history.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
        
        result.extend(trimmed_history)
        
        trimmed_count = len(history_messages) - len(trimmed_history)
        if trimmed_count > 0:
            logger.debug(f"Trimmed {trimmed_count} messages to fit token limit")
        
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._history)

    def __bool__(self) -> bool:
        return len(self._history) > 0 or self._system_message is not None

    def __repr__(self) -> str:
        return (
            f"ContextManager(turns={len(self._history)}, "
            f"tokens≈{self.get_total_tokens()}, "
            f"has_system={self._system_message is not None})"
        )