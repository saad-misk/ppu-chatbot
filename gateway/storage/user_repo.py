import uuid
from datetime import datetime, timedelta
from sqlalchemy.orm import Session as DBSession
from passlib.context import CryptContext

from .models import User, VerificationCode
from shared.config.settings import settings

# ====================== PASSWORD HASHING (Only one place) ======================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash plain password - Call this ONLY on raw password"""
    if not password:
        raise ValueError("Password cannot be empty")
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False


# ====================== USER FUNCTIONS ======================
def get_user_by_email(db: DBSession, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def create_user(db: DBSession, email: str, password: str, full_name: str = None) -> User:
    """Create new user - hashes password inside this function"""
    domain = email.split("@")[-1]
    role = "teacher" if domain == "ppu.edu" else "student"

    print(f"DEBUG: Raw password length = {len(password)}")   # ← Debug
    print(f"DEBUG: Raw password = {password[:20]}...")       # ← Debug

    hashed_pw = hash_password(password)

    print(f"DEBUG: Hashed password length = {len(hashed_pw)}")  # ← Should be around 60

    user = User(
        id=str(uuid.uuid4()),
        email=email,
        hashed_password=hashed_pw,
        full_name=full_name,
        is_verified="false",
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def save_verification_code(db: DBSession, email: str, code: str) -> VerificationCode:
    db.query(VerificationCode).filter(VerificationCode.email == email).delete()
    db.commit()

    vc = VerificationCode(
        id=str(uuid.uuid4()),
        email=email,
        code=code,
        expires_at=datetime.utcnow() + timedelta(minutes=settings.VERIFICATION_CODE_EXPIRE_MINUTES),
        used="false",
    )
    db.add(vc)
    db.commit()
    db.refresh(vc)
    return vc


def get_verification_code(db: DBSession, email: str, code: str) -> VerificationCode | None:
    return db.query(VerificationCode).filter(
        VerificationCode.email == email,
        VerificationCode.code == code,
        VerificationCode.used == "false",
        VerificationCode.expires_at > datetime.utcnow(),
    ).first()


def mark_user_verified(db: DBSession, email: str):
    user = get_user_by_email(db, email)
    if user:
        user.is_verified = "true"
        db.commit()


def mark_code_used(db: DBSession, code_obj: VerificationCode):
    code_obj.used = "true"
    db.commit()