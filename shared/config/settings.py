"""
Central config. Both services read from here via environment variables.
Copy .env.example to .env and fill in values.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ports
    NLP_SERVER_PORT: int = 8001
    GATEWAY_PORT: int = 8000

    # Internal NLP server URL (used by gateway to call nlp_engine)
    NLP_SERVER_URL: str = "http://localhost:8001"

    # Database
    DATABASE_URL: str = "sqlite:///./ppu_chat.db"   # swap for postgres in prod

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # HuggingFace
    HF_MODEL_NAME: str = "bert-base-uncased"
    HF_INFERENCE_API_KEY: str = ""

    # JWT
    JWT_SECRET: str = "change-me-in-production"
    JWT_EXPIRE_MINUTES: int = 60 * 24

    # Confidence threshold — below this, bot admits uncertainty
    CONFIDENCE_THRESHOLD: float = 0.55

    class Config:
        env_file = ".env"


settings = Settings()