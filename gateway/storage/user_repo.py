"""
User repository — password hashing, user CRUD, and verification codes.
All writes include try/except + rollback for safety.
"""
import logging
import uuid
from datetime import datetime, timedelta

from passlib.context import CryptContext
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DBSession

from .models import User, VerificationCode
from shared.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Password hashing (single source of truth)
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plain-text password. Call ONLY on raw, unhashed input."""
    if not password:
        raise ValueError("Password cannot be empty")
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if `plain` matches the bcrypt `hashed` value."""
    if not plain or not hashed:
        return False
    try:
        return pwd_context.verify(plain, hashed)
    except Exception as exc:
        logger.warning("Password verification error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def get_user_by_email(db: DBSession, email: str) -> User | None:
    try:
        return db.query(User).filter(User.email == email).first()
    except SQLAlchemyError as exc:
        logger.error("get_user_by_email failed: %s", exc)
        raise


def create_user(
    db: DBSession,
    email: str,
    password: str,
    full_name: str | None = None,
) -> User:
    """Create a new user, hashing the password inside this function."""
    domain = email.split("@")[-1].lower()
    role = "teacher" if domain == "ppu.edu" else "student"

    hashed_pw = hash_password(password)

    user = User(
        id=str(uuid.uuid4()),
        email=email,
        hashed_password=hashed_pw,
        full_name=full_name,
        is_verified="false",
        role=role,
    )
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("create_user failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Verification codes
# ---------------------------------------------------------------------------

def save_verification_code(db: DBSession, email: str, code: str) -> VerificationCode:
    try:
        # Remove any existing codes for this email first
        db.query(VerificationCode).filter(VerificationCode.email == email).delete(
            synchronize_session=False
        )
        db.flush()  # flush deletion before insert

        vc = VerificationCode(
            id=str(uuid.uuid4()),
            email=email,
            code=code,
            expires_at=datetime.utcnow()
            + timedelta(minutes=settings.VERIFICATION_CODE_EXPIRE_MINUTES),
            used="false",
        )
        db.add(vc)
        db.commit()
        db.refresh(vc)
        return vc
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("save_verification_code failed: %s", exc)
        raise


def get_verification_code(
    db: DBSession, email: str, code: str
) -> VerificationCode | None:
    try:
        return (
            db.query(VerificationCode)
            .filter(
                VerificationCode.email == email,
                VerificationCode.code == code,
                VerificationCode.used == "false",
                VerificationCode.expires_at > datetime.utcnow(),
            )
            .first()
        )
    except SQLAlchemyError as exc:
        logger.error("get_verification_code failed: %s", exc)
        raise


def mark_user_verified(db: DBSession, email: str) -> None:
    try:
        user = get_user_by_email(db, email)
        if user:
            user.is_verified = "true"
            db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("mark_user_verified failed: %s", exc)
        raise


def mark_code_used(db: DBSession, code_obj: VerificationCode) -> None:
    try:
        code_obj.used = "true"
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("mark_code_used failed: %s", exc)
        raise