from fastapi import Request, HTTPException
from gateway.storage.db import SessionLocal
from gateway.storage import chat_repo


def get_valid_session(session_id: str, request: Request):
    """
    Dependency — verifies the session exists in the DB.
    Attach to any endpoint that requires a valid session.
    """
    db = SessionLocal()
    try:
        session = chat_repo.get_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    finally:
        db.close()