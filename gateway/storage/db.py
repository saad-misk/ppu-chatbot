"""
Database engine, session factory, and FastAPI dependency.
"""
from contextlib import contextmanager
import logging

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from shared.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------
_db_url = settings.DATABASE_URL
_is_sqlite = _db_url.startswith("sqlite")

_engine_kwargs: dict = {
    "echo": False,
    "pool_pre_ping": True,  # validates connections before use
}

if _is_sqlite:
    # SQLite needs special handling for multi-threaded FastAPI workers
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    _engine_kwargs["poolclass"] = StaticPool
else:
    # PostgreSQL / Neon
    _engine_kwargs["pool_size"] = 5
    _engine_kwargs["max_overflow"] = 10
    _engine_kwargs["pool_timeout"] = 30
    _engine_kwargs["pool_recycle"] = 1800  # recycle after 30 min (good for Neon)

engine = create_engine(_db_url, **_engine_kwargs)

# For SQLite: enable WAL mode and foreign keys on every new connection
if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,   # avoid lazy-load errors after commit
)

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
Base = declarative_base()


# ---------------------------------------------------------------------------
# FastAPI dependency — yields a session and always closes it
# ---------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as exc:
        logger.error("Database error, rolling back: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Context-manager version for non-FastAPI callers (e.g., chat_service.py)
# ---------------------------------------------------------------------------
@contextmanager
def get_db_ctx():
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as exc:
        logger.error("Database error, rolling back: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()