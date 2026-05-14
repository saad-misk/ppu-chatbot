"""
Chat repository — all DB operations for sessions, turns, and feedback.
Every write uses an explicit try/except with rollback so the caller always
gets a clean error rather than a half-committed state.
"""
import logging
import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DBSession

from gateway.storage.models import ChatSession, Feedback, Turn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def create_session(
    db: DBSession,
    channel: str = "web",
    user_id: str | None = None,
    preview: str | None = None,
) -> ChatSession:
    session = ChatSession(
        id=str(uuid.uuid4()),
        channel=channel,
        user_id=user_id,
        preview=preview,
    )
    try:
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("create_session failed: %s", exc)
        raise


def get_session(db: DBSession, session_id: str) -> ChatSession | None:
    try:
        return db.query(ChatSession).filter(ChatSession.id == session_id).first()
    except SQLAlchemyError as exc:
        logger.error("get_session failed: %s", exc)
        raise


def get_sessions_for_user(db: DBSession, user_id: str) -> list[ChatSession]:
    try:
        return (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
            .all()
        )
    except SQLAlchemyError as exc:
        logger.error("get_sessions_for_user failed: %s", exc)
        raise


def update_session_preview(db: DBSession, session_id: str, preview: str) -> ChatSession | None:
    """Update the human-readable preview/title for a session."""
    try:
        session = get_session(db, session_id)
        if not session:
            return None
        session.preview = preview[:120]
        db.commit()
        db.refresh(session)
        return session
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("update_session_preview failed: %s", exc)
        raise


def delete_session(db: DBSession, session_id: str) -> None:
    """
    Hard-delete a session and all its turns + feedback.
    Relies on CASCADE rules in the model, but also handles databases
    that don't enforce FK cascades (e.g., SQLite without PRAGMA).
    """
    try:
        # Fetch turn IDs first (needed if cascades aren't enforced)
        turn_ids = [
            row[0]
            for row in db.query(Turn.id).filter(Turn.session_id == session_id).all()
        ]
        if turn_ids:
            db.query(Feedback).filter(Feedback.turn_id.in_(turn_ids)).delete(
                synchronize_session=False
            )
            db.query(Turn).filter(Turn.session_id == session_id).delete(
                synchronize_session=False
            )
        db.query(ChatSession).filter(ChatSession.id == session_id).delete(
            synchronize_session=False
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("delete_session failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Turn helpers
# ---------------------------------------------------------------------------

def save_turn(
    db: DBSession,
    session_id: str,
    role: str,
    content: str,
    intent: str | None = None,
    confidence: float | None = None,
) -> Turn:
    turn = Turn(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        content=content,
        intent=intent,
        confidence=confidence,
    )
    try:
        db.add(turn)
        db.commit()
        db.refresh(turn)
        return turn
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("save_turn failed: %s", exc)
        raise


def get_history(db: DBSession, session_id: str) -> list[Turn]:
    try:
        return (
            db.query(Turn)
            .filter(Turn.session_id == session_id)
            .order_by(Turn.created_at)
            .all()
        )
    except SQLAlchemyError as exc:
        logger.error("get_history failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Feedback helpers
# ---------------------------------------------------------------------------

def save_feedback(
    db: DBSession,
    turn_id: str,
    rating: str,
    comment: str | None = None,
) -> Feedback:
    fb = Feedback(
        id=str(uuid.uuid4()),
        turn_id=turn_id,
        rating=rating,
        comment=comment,
    )
    try:
        db.add(fb)
        db.commit()
        db.refresh(fb)
        return fb
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("save_feedback failed: %s", exc)
        raise
