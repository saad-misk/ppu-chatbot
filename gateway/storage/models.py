from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from gateway.storage.db import Base


class ChatSession(Base):
    __tablename__ = "sessions"

    id         = Column(String, primary_key=True)
    channel    = Column(String, default="web")
    created_at = Column(DateTime, default=datetime.utcnow)

    turns = relationship("Turn", back_populates="session", cascade="all, delete")


class Turn(Base):
    __tablename__ = "turns"

    id         = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role       = Column(String)           # "user" or "assistant"
    content    = Column(Text)
    intent     = Column(String, nullable=True)
    confidence = Column(Float,  nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session  = relationship("ChatSession", back_populates="turns")
    feedback = relationship("Feedback", back_populates="turn", uselist=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id         = Column(String, primary_key=True)
    turn_id    = Column(String, ForeignKey("turns.id"))
    rating     = Column(String)           # "up" or "down"
    comment    = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    turn = relationship("Turn", back_populates="feedback")