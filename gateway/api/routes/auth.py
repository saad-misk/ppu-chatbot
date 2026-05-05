from fastapi import APIRouter
from gateway.api.auth.jwt_handler import create_token

router = APIRouter()

@router.post("/auth/token")
def get_token(username: str = "admin"):
    return {"token": create_token({"sub": username})}