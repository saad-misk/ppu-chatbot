from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from sqlalchemy import inspect, text
from gateway.storage.db import Base, engine, SessionLocal
from gateway.storage.models import User

from gateway.storage.db import Base, engine, SessionLocal
from gateway.storage.models import User 
from passlib.hash import bcrypt

# Create tables
Base.metadata.create_all(bind=engine)


def ensure_schema_updates():
    inspector = inspect(engine)
    if "sessions" not in inspector.get_table_names():
        return

    session_columns = {col["name"] for col in inspector.get_columns("sessions")}
    if "user_id" in session_columns:
        return

    dialect = engine.dialect.name
    column_type = "VARCHAR" if dialect == "postgresql" else "VARCHAR"
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE sessions ADD COLUMN user_id {column_type}"))
        if dialect == "postgresql":
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_sessions_user_id ON sessions (user_id)"))
        else:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_sessions_user_id ON sessions (user_id)"))


ensure_schema_updates()

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
                print("✅ Default admin role/verification corrected")
            print("✅ Default admin already exists")
            return

        from gateway.storage.user_repo import hash_password

        hashed_pw = hash_password("admin123")

        admin = User(
            id=str(uuid.uuid4()),         
            email=admin_email,
            hashed_password=hashed_pw,
            full_name="System Administrator",
            is_verified="true",          
            role="admin"
        )
        db.add(admin)
        db.commit()
        print("✅ Default admin created successfully!")
        print("   Email    : admin@ppu.edu.ps")
        print("   Password : admin123")
        
    except Exception as e:
        print(f"❌ Seeding error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

seed_default_admin()

app = FastAPI(title="PPU Chatbot Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from gateway.api.routes.auth import router as auth_router
from gateway.api.routes.chat import router as chat_router
from gateway.api.routes.admin import router as admin_router
from gateway.api.routes.health import router as health_router

app.include_router(auth_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(health_router, prefix="/api")

app.mount("/static", StaticFiles(directory="gateway/frontend/static", check_dir=False), name="static")

# Routes...
@app.get("/")
async def serve_login():
    return FileResponse("gateway/frontend/login.html")

@app.get("/chat")
async def serve_chat():
    return FileResponse("gateway/frontend/index.html")

@app.get("/index.html")
async def serve_index():
    return FileResponse("gateway/frontend/index.html")

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
