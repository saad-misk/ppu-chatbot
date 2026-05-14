from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base


class ChatSession(Base):
    __tablename__ = "sessions"

    id         = Column(String, primary_key=True)
    user_id    = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    channel    = Column(String, default="web")
    created_at = Column(DateTime, default=datetime.utcnow)

    turns = relationship("Turn", back_populates="session", cascade="all, delete")
    user  = relationship("User", back_populates="sessions")


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

class User(Base):
    __tablename__ = "users"

    id              = Column(String, primary_key=True)
    email           = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name       = Column(String, nullable=True)
    is_verified     = Column(String, default="false")  # "true" / "false"
    role            = Column(String, default="student")  # "student" / "academic"
    created_at      = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("ChatSession", back_populates="user")


class VerificationCode(Base):
    __tablename__ = "verification_codes"

    id         = Column(String, primary_key=True)
    email      = Column(String, nullable=False, index=True)
    code       = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used       = Column(String, default="false")  # "true" / "false"
    created_at = Column(DateTime, default=datetime.utcnow)



    
