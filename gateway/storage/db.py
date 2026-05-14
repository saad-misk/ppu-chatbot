from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from shared.config.settings import settings

# Create engine (Neon PostgreSQL)
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,           # Change to True if you want to see SQL queries
    pool_pre_ping=True,   # Good for Neon
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
from sqlalchemy.orm import declarative_base
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()