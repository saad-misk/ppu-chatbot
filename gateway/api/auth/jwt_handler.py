"""
JWT utilities.

  create_token(data)       → issue a signed JWT
  verify_token(...)        → strict — raises 401 if missing/invalid (admin/auth routes)
  optional_token(...)      → lenient — returns payload dict OR None (chat routes)
"""
from datetime import datetime, timedelta

from fastapi import HTTPException, Security, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError

from shared.config.settings import settings

_security_required = HTTPBearer(auto_error=True)   # raises 403 automatically
_security_optional = HTTPBearer(auto_error=False)  # returns None if header absent


def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def _decode(token_str: str) -> dict:
    try:
        return jwt.decode(token_str, settings.JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(_security_required),
) -> dict:
    """Strict dependency — 401 if no valid token."""
    return _decode(credentials.credentials)


def optional_token(
    credentials: HTTPAuthorizationCredentials = Security(_security_optional),
) -> dict | None:
    """
    Lenient dependency — returns the decoded payload if a valid token is
    provided, otherwise returns None (guest / unauthenticated).
    """
    if credentials is None:
        return None
    try:
        return _decode(credentials.credentials)
    except HTTPException:
        return None  # treat invalid token as guest