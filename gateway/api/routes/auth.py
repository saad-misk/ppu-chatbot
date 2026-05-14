import random
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from pydantic import BaseModel
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from fastapi import Header, HTTPException
from jose import jwt, JWTError

from gateway.storage.db import get_db
from gateway.storage.user_repo import get_user_by_email
from shared.config.settings import settings
from gateway.storage.user_repo import (
    create_user,
    get_user_by_email,
    save_verification_code,
    get_verification_code,
    mark_user_verified,
    mark_code_used,
    verify_password,
)

from shared.config.settings import settings
from gateway.storage.db import get_db

from gateway.api.auth.jwt_handler import create_token
from gateway.api.auth.email_sender import send_verification_email, generate_code


router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class VerifyRequest(BaseModel):
    email: str
    code: str


@router.post("/register")
async def register(request: RegisterRequest, db: DBSession = Depends(get_db)):
    
    domain = request.email.split("@")[-1].lower()
    if domain not in [d.lower() for d in settings.PPU_EMAIL_DOMAINS]:
        raise HTTPException(status_code=400, detail="Only PPU emails allowed")

    if get_user_by_email(db, request.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    try:
        user = create_user(
            db=db,
            email=request.email,
            password=request.password,
            full_name=request.full_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create user")

    # Generate verification code
    code = str(random.randint(100000, 999999))
    save_verification_code(db, request.email, code)

    # === SEND EMAIL TO THE NEW USER ===
    try:
        send_verification_email(request.email, code)
        print(f"✅ Verification code sent to {request.email}")
    except Exception as e:
        print(f"❌ Failed to send email to {request.email}: {e}")
        # Don't fail registration if email fails (for development)

    return {
        "message": "Account created successfully! Please check your email for verification code.",
        "email": request.email
    }


@router.post("/verify")
async def verify_email(payload: VerifyRequest, db: DBSession = Depends(get_db)):
    code_obj = get_verification_code(db, payload.email, payload.code)
    if not code_obj:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code")

    mark_user_verified(db, payload.email)
    mark_code_used(db, code_obj)

    return {"message": "Email verified successfully. You can now log in."}


@router.post("/login")
async def login(payload: LoginRequest, db: DBSession = Depends(get_db)):
    user = get_user_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if user.is_verified == "false":
        raise HTTPException(status_code=403, detail="Email not verified")

    token = create_token({"sub": user.email, "role": user.role})
    return {
        "token": token,
        "role": user.role,
        "full_name": user.full_name
    }

@router.get("/me")
async def get_current_user(
    Authorization: str = Header(None), 
    db: DBSession = Depends(get_db)
):
    """Simple /me endpoint - works with your existing JWT token"""
    if not Authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    try:
        # Remove "Bearer " if present
        token = Authorization.replace("Bearer ", "").strip()

        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"],
        )
        
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = get_user_by_email(db, email=email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "is_admin": user.role == "admin"
        }

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
@router.post("/resend-code")
async def resend_code(email: str, db: DBSession = Depends(get_db)):
    domain = email.split("@")[-1].lower()
    if domain not in [d.lower() for d in settings.PPU_EMAIL_DOMAINS]:
        raise HTTPException(status_code=400, detail="Invalid PPU email")

    user = get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")
    if user.is_verified == "true":
        raise HTTPException(status_code=400, detail="Email already verified")

    code = generate_code()
    save_verification_code(db, email, code)

    try:
        send_verification_email(email, code)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to send email")

    return {"message": "Verification code resent successfully."}
