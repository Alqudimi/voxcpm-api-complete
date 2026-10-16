from fastapi import APIRouter
from app.core.model_manager import model_manager

router = APIRouter(tags=["System"])

@router.get("/health")
async def health():
    return {
        "success": True,
        "status": "healthy" if model_manager.ready else "starting",
        "models": {
            "voxcpm2": model_manager.voxcpm is not None,
            "whisper": model_manager.whisper is not None,
            "ecapa": model_manager.speaker_verification is not None,
        },
    }

@router.get("/")
async def root():
    return {
        "service": "VoxCPM2 Professional Voice API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
