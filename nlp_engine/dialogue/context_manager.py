"""
Conversation context manager — formats and trims multi-turn history
for injection into the RAG generator prompt.

Usage
-----
    from nlp_engine.dialogue.context_manager import ContextManager

    ctx = ContextManager(max_turns=6)
    ctx.add_turn("user", "What are the fees?")
    ctx.add_turn("assistant", "The fees are...")
    history = ctx.get_history()   # List[{"role": ..., "content": ...}]
"""
from __future__ import annotations

from collections import deque
from typing import List, Dict, Deque


class ContextManager:
    """
    In-memory store for a single session's conversation history.

    Keeps the last *max_turns* turns (user + assistant combined).
    The gateway persists full history in PostgreSQL; this class manages
    the in-flight window passed to the NLP engine on each request.
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns: int = max_turns
        self._history: Deque[Dict[str, str]] = deque(maxlen=max_turns)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        """Append a single turn.  role must be 'user' or 'assistant'."""
        assert role in ("user", "assistant"), f"Unknown role: {role}"
        self._history.append({"role": role, "content": content.strip()})

    def load_history(self, history: List[Dict[str, str]]) -> None:
        """
        Populate from an external history list (e.g. sent by the gateway).
        Only the last *max_turns* items are kept.
        """
        self._history.clear()
        for turn in history[-self.max_turns:]:
            self._history.append(turn)

    def clear(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current history as a plain list."""
        return list(self._history)

    def get_last_user_message(self) -> str | None:
        """Return the most recent user message, or None."""
        for turn in reversed(self._history):
            if turn["role"] == "user":
                return turn["content"]
        return None

    def format_as_string(self) -> str:
        """Human-readable history string for debugging."""
        lines = []
        for turn in self._history:
            prefix = "User" if turn["role"] == "user" else "Bot"
            lines.append(f"{prefix}: {turn['content']}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._history)
