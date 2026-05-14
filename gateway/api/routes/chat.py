"""
Chat routes.

Design: ALL chat endpoints are open to guests AND authenticated users.
  - Guests get a session with user_id=None (anonymous).
  - Authenticated users have sessions linked to their user_id.
  - Session ownership is only enforced for authenticated users.
"""
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session as DBSession

from shared.config.settings import settings
from shared.schemas.message import ProcessRequest, ProcessResponse, FeedbackRequest
from gateway.storage.db import get_db
from gateway.storage import chat_repo
from gateway.storage.user_repo import get_user_by_email
from gateway.api.auth.jwt_handler import optional_token
from gateway.api.middleware.rate_limiter import check_rate_limit
from gateway.api.middleware.validator import validate_message

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_user_id(db: DBSession, token: dict | None) -> str | None:
    """Return the DB user-id from a decoded token, or None for guests."""
    if token is None:
        return None
    email = token.get("sub")
    if not email:
        return None
    user = get_user_by_email(db, email)
    return user.id if user else None


def _assert_session_owner(session, user_id: str | None) -> None:
    """
    Raise 403 only when BOTH the session AND the requester are authenticated
    AND they don't match.  Guests can always access anonymous sessions.
    """
    if session.user_id and user_id and session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user")


# ---------------------------------------------------------------------------
# NLP engine call
# ---------------------------------------------------------------------------

async def call_nlp_engine(request: ProcessRequest) -> ProcessResponse:
    timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.NLP_SERVER_URL}/process",
                json=request.model_dump(),
            )
            response.raise_for_status()
            return ProcessResponse(**response.json())
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=500, detail=f"NLP engine error: {exc.response.text}")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="The NLP engine took too long to respond.")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Error connecting to NLP engine: {exc}")


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/new")
def new_session(
    channel: str = "web",
    db: DBSession = Depends(get_db),
    token: dict | None = Depends(optional_token),
):
    """Create a session.  Works for guests (no token) and logged-in users."""
    user_id = _resolve_user_id(db, token)
    session = chat_repo.create_session(db, channel=channel, user_id=user_id)
    return {"session_id": session.id, "channel": session.channel}


@router.get("/sessions")
def list_sessions(
    db: DBSession = Depends(get_db),
    token: dict | None = Depends(optional_token),
):
    """
    Return sessions for the authenticated user.
    Guests get an empty list (they track their session_id client-side).
    """
    user_id = _resolve_user_id(db, token)
    if not user_id:
        return {"sessions": []}

    sessions = chat_repo.get_sessions_for_user(db, user_id)
    result = []
    for session in sessions:
        turns = chat_repo.get_history(db, session.id)
        first_user_turn = next((t for t in turns if t.role == "user"), None)
        auto_preview = (
            first_user_turn.content[:36] + ("…" if len(first_user_turn.content) > 36 else "")
            if first_user_turn
            else None
        )
        result.append(
            {
                "session_id": session.id,
                "channel": session.channel,
                "created_at": session.created_at,
                "preview": session.preview or auto_preview or "محادثة جديدة",
            }
        )
    return {"sessions": result}


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    db: DBSession = Depends(get_db),
    token: dict | None = Depends(optional_token),
):
    user_id = _resolve_user_id(db, token)
    session = chat_repo.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, user_id)
    chat_repo.delete_session(db, session_id)
    return {"message": "Session deleted"}


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

@router.post("/chat/message")
async def send_message(
    session_id: str,
    message: str,
    db: DBSession = Depends(get_db),
    _rate: None = Depends(check_rate_limit),
    token: dict | None = Depends(optional_token),
):
    """Send a message.  Works without authentication (guest / open access)."""
    message = validate_message(message)
    user_id = _resolve_user_id(db, token)

    session = chat_repo.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, user_id)

    history_turns = chat_repo.get_history(db, session_id)
    history = [{"role": t.role, "content": t.content} for t in history_turns]

    chat_repo.save_turn(db, session_id=session_id, role="user", content=message)

    nlp_request = ProcessRequest(
        session_id=session_id,
        message=message,
        history=history,
        channel=session.channel,
    )
    nlp_response = await call_nlp_engine(nlp_request)

    assistant_turn = chat_repo.save_turn(
        db,
        session_id=session_id,
        role="assistant",
        content=nlp_response.reply,
        intent=nlp_response.intent,
        confidence=nlp_response.confidence,
    )

    return {
        "turn_id": assistant_turn.id,
        "reply": nlp_response.reply,
        "intent": nlp_response.intent,
        "confidence": nlp_response.confidence,
        "sources": nlp_response.sources,
        "low_confidence": (nlp_response.confidence < settings.CONFIDENCE_THRESHOLD),
    }


@router.get("/chat/history/{session_id}")
def get_history(
    session_id: str,
    db: DBSession = Depends(get_db),
    token: dict | None = Depends(optional_token),
):
    user_id = _resolve_user_id(db, token)
    session = chat_repo.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, user_id)

    turns = chat_repo.get_history(db, session_id)
    return {
        "session_id": session_id,
        "turns": [
            {
                "id": t.id,
                "role": t.role,
                "content": t.content,
                "intent": t.intent,
                "confidence": t.confidence,
                "created_at": t.created_at,
            }
            for t in turns
        ],
    }


@router.post("/chat/feedback")
def submit_feedback(
    payload: FeedbackRequest = Body(...),
    db: DBSession = Depends(get_db),
):
    """Feedback is always open — no auth required."""
    fb = chat_repo.save_feedback(
        db,
        turn_id=payload.message_id,
        rating=payload.rating,
        comment=payload.comment,
    )
    return {"feedback_id": fb.id, "status": "saved"}


# ---------------------------------------------------------------------------
# Session rename
# ---------------------------------------------------------------------------

from pydantic import BaseModel as PydanticBase


class RenameRequest(PydanticBase):
    preview: str


@router.patch("/sessions/{session_id}/rename")
def rename_session(
    session_id: str,
    payload: RenameRequest = Body(...),
    db: DBSession = Depends(get_db),
    token: dict | None = Depends(optional_token),
):
    user_id = _resolve_user_id(db, token)
    session = chat_repo.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, user_id)

    updated = chat_repo.update_session_preview(db, session_id, payload.preview)
    return {"session_id": session_id, "preview": updated.preview if updated else payload.preview}