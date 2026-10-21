from fastapi import APIRouter
import torch
from app.core.model_manager import model_manager

router = APIRouter(tags=["System"])

@router.get("/health")
async def health():
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch.cuda.memory_reserved(0)),
            "total_bytes": int(props.total_memory),
        }
    dtype = None
    if model_manager.voxcpm is not None:
        try:
            dtype = str(next(model_manager.voxcpm.tts_model.parameters()).dtype)
        except (AttributeError, StopIteration, TypeError):
            dtype = "unknown"
    return {
        "success": True,
        "status": "healthy" if model_manager.ready else "starting",
        "models": {
            "voxcpm2": model_manager.voxcpm is not None,
            "whisper": model_manager.whisper is not None,
            "ecapa": model_manager.speaker_verification is not None,
        },
        "runtime": {
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu,
            "voxcpm_dtype": dtype,
            "whisper_backend": model_manager.whisper_backend,
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
