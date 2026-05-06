from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from gateway.storage.db import Base, engine

from gateway.api.routes.chat import router as chat_router
from gateway.api.routes.admin import router as admin_router
from gateway.api.routes.health import router as health_router
from gateway.api.routes.auth import router as auth_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="PPU Chatbot Gateway", 
    version="1.0.0",
    description="Backend for PPU Chatbot"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(health_router, prefix="/api")
app.include_router(auth_router, prefix="/api")

app.mount("/", StaticFiles(directory="gateway/frontend", html=True), name="frontend")