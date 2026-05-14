import httpx
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session as DBSession

from shared.config.settings import settings
from shared.schemas.message import ProcessRequest, ProcessResponse, FeedbackRequest
from gateway.storage.db import get_db
from gateway.storage import chat_repo
from gateway.storage.user_repo import get_user_by_email
from gateway.api.auth.jwt_handler import verify_token
from gateway.api.middleware.rate_limiter import check_rate_limit
from gateway.api.middleware.session import get_valid_session
from gateway.api.middleware.validator import validate_message

router = APIRouter()


def get_user_id_from_token(db: DBSession, token: dict) -> str:
    email = token.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user.id


def require_session_owner(session, user_id: str):
    if session.user_id and session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user")


async def call_nlp_engine(request: ProcessRequest) -> ProcessResponse:
    try:
        # Increased timeout settings for slower model responses
        timeout = httpx.Timeout(
            connect=5.0,   # connection timeout
            read=60.0,     # wait longer for model response
            write=10.0,
            pool=5.0,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.NLP_SERVER_URL}/process",
                json=request.model_dump(),
            )

            response.raise_for_status()

            return ProcessResponse(**response.json())

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=500,
            detail=f"NLP engine error: {e.response.text}",
        )

    except httpx.ReadTimeout:
        raise HTTPException(
            status_code=504,
            detail="The NLP engine took too long to respond.",
        )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error connecting to NLP engine: {e}",
        )


@router.post("/sessions/new")
def new_session(
    channel: str = "web",
    db: DBSession = Depends(get_db),
    token: dict = Depends(verify_token),
):
    user_id = get_user_id_from_token(db, token)
    session = chat_repo.create_session(db, channel=channel, user_id=user_id)

    return {
        "session_id": session.id,
        "channel": session.channel,
    }


@router.get("/sessions")
def list_sessions(
    db: DBSession = Depends(get_db),
    token: dict = Depends(verify_token),
):
    user_id = get_user_id_from_token(db, token)
    sessions = chat_repo.get_sessions_for_user(db, user_id)

    result = []
    for session in sessions:
        turns = chat_repo.get_history(db, session.id)
        first_user_turn = next((t for t in turns if t.role == "user"), None)
        preview = first_user_turn.content[:36] if first_user_turn else "New Chat"
        result.append(
            {
                "session_id": session.id,
                "channel": session.channel,
                "created_at": session.created_at,
                "preview": preview,
            }
        )

    return {"sessions": result}


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    db: DBSession = Depends(get_db),
    token: dict = Depends(verify_token),
):
    user_id = get_user_id_from_token(db, token)
    session = chat_repo.get_session(db, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    require_session_owner(session, user_id)
    chat_repo.delete_session(db, session_id)

    return {"message": "Session deleted"}


@router.post("/chat/message")
async def send_message(
    session_id: str,
    message: str,
    db: DBSession = Depends(get_db),
    _rate: None = Depends(check_rate_limit),
    _session=Depends(get_valid_session),
    token: dict = Depends(verify_token),
):
    message = validate_message(message)
    user_id = get_user_id_from_token(db, token)

    session = chat_repo.get_session(db, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    require_session_owner(session, user_id)

    history_turns = chat_repo.get_history(db, session_id)

    history = [
        {
            "role": t.role,
            "content": t.content,
        }
        for t in history_turns
    ]

    chat_repo.save_turn(
        db,
        session_id=session_id,
        role="user",
        content=message,
    )

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
        "low_confidence": (
            nlp_response.confidence < settings.CONFIDENCE_THRESHOLD
        ),
    }


@router.get("/chat/history/{session_id}")
def get_history(
    session_id: str,
    db: DBSession = Depends(get_db),
    token: dict = Depends(verify_token),
):
    user_id = get_user_id_from_token(db, token)
    session = chat_repo.get_session(db, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    require_session_owner(session, user_id)

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
    fb = chat_repo.save_feedback(
        db,
        turn_id=payload.message_id,
        rating=payload.rating,
        comment=payload.comment,
    )

    return {
        "feedback_id": fb.id,
        "status": "saved",
    }

from pydantic import BaseModel as PydanticBase

class RenameRequest(PydanticBase):
    preview: str

@router.patch("/sessions/{session_id}/rename")
def rename_session(
    session_id: str,
    payload: RenameRequest = Body(...),
    db: DBSession = Depends(get_db),
    token: dict = Depends(verify_token),
):
    user_id = get_user_id_from_token(db, token)
    session = chat_repo.get_session(db, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    require_session_owner(session, user_id)

    session.preview = payload.preview[:120]
    db.commit()

    return {"session_id": session_id, "preview": session.preview}