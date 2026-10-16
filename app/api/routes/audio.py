from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.security import require_api_key
from app.services.audio.merge_service import AudioMergeService

router = APIRouter(prefix="/audio", tags=["Audio"])

@router.post("/merge", dependencies=[Depends(require_api_key)])
async def merge(request: Request, files: list[UploadFile] = File(...)):  # noqa: B008
    output = await AudioMergeService().merge(files)
    base = str(request.base_url).rstrip("/")
    return {
        "success": True,
        "request_id": Path(output).stem,
        "format": "ogg",
        "mime_type": "audio/ogg",
        "audio_url": f"{base}/api/v1/audio/output/{Path(output).name}",
        "input_count": len(files),
    }

@router.get("/output/{filename}", dependencies=[Depends(require_api_key)])
async def get_output(filename: str):
    safe_filename = Path(filename).name
    if safe_filename != filename or safe_filename in {".", ".."}:
        raise HTTPException(status_code=404, detail="Audio not found")
    path = Path(settings.OUTPUT_DIR) / safe_filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/ogg", filename=path.name)
