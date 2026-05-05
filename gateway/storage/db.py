from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Temporarily hardcode for testing
DATABASE_URL = "sqlite:///./ppu_chatbot.db"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()