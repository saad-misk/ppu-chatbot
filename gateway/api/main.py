"""
Gateway application entry-point.
"""
import uuid
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from gateway.storage.db import Base, engine, SessionLocal
from gateway.storage.models import User

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Create / migrate tables
# ---------------------------------------------------------------------------
Base.metadata.create_all(bind=engine)


def ensure_schema_updates():
    """
    Apply lightweight schema migrations that SQLAlchemy's create_all misses
    (e.g. adding columns to existing tables).
    """
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    # --- sessions.preview column ---
    if "sessions" in existing_tables:
        session_cols = {col["name"] for col in inspector.get_columns("sessions")}

        if "preview" not in session_cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN preview VARCHAR"))
            logger.info("✅ sessions.preview column added")

        if "user_id" not in session_cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN user_id VARCHAR"))
            logger.info("✅ sessions.user_id column added")


try:
    ensure_schema_updates()
except Exception as exc:
    logger.warning("Schema migration warning (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Seed default admin
# ---------------------------------------------------------------------------
def seed_default_admin():
    db = SessionLocal()
    try:
        admin_email = "admin@ppu.edu.ps"
        existing = db.query(User).filter(User.email == admin_email).first()
        if existing:
            changed = False
            if existing.role != "admin":
                existing.role = "admin"
                changed = True
            if existing.is_verified != "true":
                existing.is_verified = "true"
                changed = True
            if changed:
                db.commit()
                logger.info("✅ Default admin role/verification corrected")
            return

        from gateway.storage.user_repo import hash_password
        admin = User(
            id=str(uuid.uuid4()),
            email=admin_email,
            hashed_password=hash_password("admin123"),
            full_name="System Administrator",
            is_verified="true",
            role="admin",
        )
        db.add(admin)
        db.commit()
        logger.info("✅ Default admin created: %s / admin123", admin_email)

    except Exception as exc:
        logger.error("❌ Admin seed error: %s", exc)
        db.rollback()
    finally:
        db.close()


seed_default_admin()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="PPU Chatbot Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from gateway.api.routes.auth import router as auth_router
from gateway.api.routes.chat import router as chat_router
from gateway.api.routes.admin import router as admin_router
from gateway.api.routes.health import router as health_router

app.include_router(auth_router,   prefix="/api")
app.include_router(chat_router,   prefix="/api")
app.include_router(admin_router,  prefix="/api")
app.include_router(health_router, prefix="/api")

# Static files
app.mount("/static", StaticFiles(directory="gateway/frontend/static", check_dir=False), name="static")

# ---------------------------------------------------------------------------
# Frontend routes
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_index():
    """Chat page — open to everyone, no auth redirect."""
    return FileResponse("gateway/frontend/index.html")

@app.get("/chat")
async def serve_chat():
    return FileResponse("gateway/frontend/index.html")

@app.get("/login")
async def serve_login():
    return FileResponse("gateway/frontend/login.html")

@app.get("/style.css")
async def serve_css():
    return FileResponse("gateway/frontend/style.css", media_type="text/css")

@app.get("/chat.js")
async def serve_chat_js():
    return FileResponse("gateway/frontend/chat.js", media_type="application/javascript")

@app.get("/embed.js")
async def serve_embed_js():
    return FileResponse("gateway/frontend/embed.js", media_type="application/javascript")

@app.get("/admin.html")
async def serve_admin():
    return FileResponse("gateway/frontend/admin.html")

@app.get("/admin.js")
async def serve_admin_js():
    return FileResponse("gateway/frontend/admin.js", media_type="application/javascript")
