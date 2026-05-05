"""
Shared request/response schemas.
"""
from pydantic import BaseModel
from typing import Literal, Optional


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ProcessRequest(BaseModel):
    session_id: str
    message: str
    history: list[Turn] = []
    channel: Literal["web", "telegram", "gmail"] = "web"


class Source(BaseModel):
    text: str
    doc_name: str
    page: Optional[int] = None


class ProcessResponse(BaseModel):
    reply: str
    intent: str
    confidence: float        # 0.0 – 1.0
    sources: list[Source] = []


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None