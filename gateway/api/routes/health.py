from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession
from gateway.storage.db import get_db
import httpx
from shared.config.settings import settings

router = APIRouter()

@router.get("/health")
async def health_check(db: DBSession = Depends(get_db)):
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.NLP_SERVER_URL}/health")
            nlp_status = "ok" if response.status_code == 200 else "error"
    except Exception:
        nlp_status = "unreachable"

    return {
        "status": "ok",
        "service": "gateway",
        "database": db_status,
        "nlp_engine": nlp_status,
    }