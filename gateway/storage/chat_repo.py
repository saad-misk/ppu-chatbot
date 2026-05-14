import uuid

from sqlalchemy.orm import Session as DBSession

from gateway.storage.models import ChatSession, Feedback, Turn


def create_session(
    db: DBSession,
    channel: str = "web",
    user_id: str | None = None,
) -> ChatSession:
    session = ChatSession(id=str(uuid.uuid4()), channel=channel, user_id=user_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: DBSession, session_id: str) -> ChatSession | None:
    return db.query(ChatSession).filter(ChatSession.id == session_id).first()


def get_sessions_for_user(db: DBSession, user_id: str) -> list[ChatSession]:
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )


def delete_session(db: DBSession, session_id: str):
    from gateway.storage.models import Turn, Feedback
    turn_ids = [t.id for t in db.query(Turn).filter(Turn.session_id == session_id).all()]
    if turn_ids:
        db.query(Feedback).filter(Feedback.turn_id.in_(turn_ids)).delete(synchronize_session=False)
        db.query(Turn).filter(Turn.session_id == session_id).delete(synchronize_session=False)
    db.query(ChatSession).filter(ChatSession.id == session_id).delete(synchronize_session=False)
    db.commit()


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
    comment: str | None = None,
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
