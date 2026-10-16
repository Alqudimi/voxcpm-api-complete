from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.model_manager import model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.load_all()
    yield
    await model_manager.unload_all()
