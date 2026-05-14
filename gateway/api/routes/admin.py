from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func
import httpx

from shared.config.settings import settings
from gateway.storage.db import get_db
from gateway.storage.models import Turn, Feedback
from gateway.storage.user_repo import get_user_by_email
from gateway.api.auth.jwt_handler import verify_token

router = APIRouter(prefix="/admin")


def require_admin(token: dict, db: DBSession):
    email = token.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = get_user_by_email(db, email)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return user

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    token: dict = Depends(verify_token),
    db: DBSession = Depends(get_db),
):
    require_admin(token, db)

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_bytes = await file.read()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.NLP_SERVER_URL}/ingest",
                files={"file": (file.filename, file_bytes, "application/pdf")},
            )
            response.raise_for_status()
            return {"message": f"{file.filename} uploaded and indexed successfully"}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"NLP engine error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error connecting to NLP engine: {e}")


@router.get("/documents")
async def list_documents(
    token: dict = Depends(verify_token),
    db: DBSession = Depends(get_db),
):
    require_admin(token, db)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.NLP_SERVER_URL}/documents")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"NLP engine error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error connecting to NLP engine: {e}")


@router.get("/stats")
def get_stats(
    db: DBSession = Depends(get_db),
    token: dict = Depends(verify_token),
):
    require_admin(token, db)

    total_queries = db.query(Turn).filter(Turn.role == "user").count()

    intent_breakdown = (
        db.query(Turn.intent, func.count(Turn.intent))
        .filter(Turn.role == "assistant", Turn.intent != None)
        .group_by(Turn.intent)
        .all()
    )

    total_feedback = db.query(Feedback).count()
    positive_feedback = db.query(Feedback).filter(Feedback.rating == "up").count()
    negative_feedback = db.query(Feedback).filter(Feedback.rating == "down").count()

    return {
        "total_queries": total_queries,
        "intent_breakdown": {intent: count for intent, count in intent_breakdown},
        "feedback": {
            "total": total_feedback,
            "positive": positive_feedback,
            "negative": negative_feedback,
            "ratio": round(positive_feedback / total_feedback, 2) if total_feedback > 0 else 0,
        },
    }
