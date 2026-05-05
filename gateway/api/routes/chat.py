import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from gateway.storage.db import get_db
from gateway.storage import chat_repo

router = APIRouter()


def get_settings():
    from shared.config.settings import settings
    return settings


def get_schemas():
    from shared.schemas.message import (
        ProcessRequest,
        ProcessResponse,
        FeedbackRequest,
    )
    return ProcessRequest, ProcessResponse, FeedbackRequest


async def call_nlp_engine(request):
    ProcessRequest, ProcessResponse, _ = get_schemas()
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{settings.NLP_SERVER_URL}/process",
                json=request.model_dump(),
            )
            response.raise_for_status()
            return ProcessResponse(**response.json())

    except Exception:
        return ProcessResponse(
            reply="This is a mock reply. NLP engine is not connected yet.",
            intent="mock_intent",
            confidence=0.99,
            sources=[],
        )

@router.post("/sessions/new")
def new_session(channel: str = "web", db: DBSession = Depends(get_db)):
    session = chat_repo.create_session(db, channel=channel)
    return {"session_id": session.id, "channel": session.channel}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: DBSession = Depends(get_db)):
    session = chat_repo.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    chat_repo.delete_session(db, session_id)
    return {"message": "Session deleted"}


@router.post("/chat/message")
async def send_message(
    session_id: str,
    message: str,
    db: DBSession = Depends(get_db),
):
    ProcessRequest, _, _ = get_schemas()
    settings = get_settings()

    session = chat_repo.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

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
        "low_confidence": nlp_response.confidence
        < settings.CONFIDENCE_THRESHOLD,
    }

@router.get("/chat/history/{session_id}")
def get_history(session_id: str, db: DBSession = Depends(get_db)):
    session = chat_repo.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

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
def submit_feedback(payload, db: DBSession = Depends(get_db)):
    _, _, FeedbackRequest = get_schemas()

    fb = chat_repo.save_feedback(
        db,
        turn_id=payload.message_id,
        rating=payload.rating,
        comment=payload.comment,
    )

    return {"feedback_id": fb.id, "status": "saved"}