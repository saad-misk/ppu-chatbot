"""
Central config. Both services read from here via environment variables.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # ====================== Core ======================
    NLP_SERVER_PORT: int = 8001
    GATEWAY_PORT: int = 8000
    NLP_SERVER_URL: str = "http://localhost:8001"

    # ====================== Database ======================
    DATABASE_URL: str = "sqlite:///./ppu_chat.db"

    # ====================== Chroma ======================
    CHROMA_PERSIST_DIR: str = "./data/chroma"


    # openrouter
    OPENROUTER_API_KEY: str = ""
    LLM_PROVIDER: str = "openrouter"
    LLM_MODEL: str | None = None
    OPENROUTER_API_KEY: str | None = None

    # NVIDIA
    NVIDIA_API_KEY: str | None = None

    # HuggingFace
    HF_MODEL_NAME: str = "bert-base-uncased"
    HF_INFERENCE_API_KEY: str = ""

    # Gemini    
    GEMINI_API_KEY: str = ""

    # JWT
    JWT_SECRET: str = "change-me-in-production"
    JWT_EXPIRE_MINUTES: int = 1440

    # ====================== Application ======================
    CONFIDENCE_THRESHOLD: float = 0.55

    # Hybrid retrieval (BM25 + embeddings)
    BM25_TOP_K: int = 25
    EMBED_TOP_K: int = 25
    BM25_WEIGHT: float = 0.6


    # ====================== Email Verification ======================
    GMAIL_SENDER: str = ""
    GMAIL_APP_PASSWORD: str = ""
    VERIFICATION_CODE_EXPIRE_MINUTES: int = 15
    PPU_EMAIL_DOMAINS: List[str] = Field(
        default=["ppu.edu.ps", "ppu.edu"],
        description="Allowed PPU email domains"
    )

    # ====================== Telegram ======================
    TELEGRAM_BOT_TOKEN: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


settings = Settings()