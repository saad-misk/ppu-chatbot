import uuid
from sqlalchemy.orm import Session as DBSession
from gateway.storage.models import ChatSession, Turn, Feedback


def create_session(db: DBSession, channel: str = "web") -> ChatSession:
    session = ChatSession(id=str(uuid.uuid4()), channel=channel)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: DBSession, session_id: str) -> ChatSession | None:
    return db.query(ChatSession).filter(ChatSession.id == session_id).first()


def delete_session(db: DBSession, session_id: str):
    db.query(ChatSession).filter(ChatSession.id == session_id).delete()
    db.commit()

def save_turn(
    db: DBSession,
    session_id: str,
    role: str,
    content: str,
    intent: str = None,
    confidence: float = None,
) -> Turn:
    turn = Turn(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        content=content,
        intent=intent,
        confidence=confidence,
    )
    db.add(turn)
    db.commit()
    db.refresh(turn)
    return turn


def get_history(db: DBSession, session_id: str) -> list[Turn]:
    return (
        db.query(Turn)
        .filter(Turn.session_id == session_id)
        .order_by(Turn.created_at)
        .all()
    )

def save_feedback(
    db: DBSession,
    turn_id: str,
    rating: str,
    comment: str = None,
) -> Feedback:
    fb = Feedback(
        id=str(uuid.uuid4()),
        turn_id=turn_id,
        rating=rating,
        comment=comment,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb