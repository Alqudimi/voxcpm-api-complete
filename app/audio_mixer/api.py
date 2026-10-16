import json
import math
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .admin import ServiceAdmin
from .catalog_import import CatalogImportManager
from .config import Settings
from .db import Database
from .errors import AudioMixerError, ForbiddenError, UnauthorizedError
from .ffmpeg import FFmpeg
from .service import AudioService

settings = Settings.from_env()
settings.prepare()
fallback_database_path = os.getenv(
    "AUDIO_MIXER_FALLBACK_SQLITE_PATH",
    str(settings.temp_path / "database-fallback" / "audio_mixer_fallback.sqlite3"),
)
db = Database(settings.database_url, fallback_path=fallback_database_path)
ffmpeg = FFmpeg(settings.ffmpeg_path, settings.ffprobe_path)
service = AudioService(db, ffmpeg, settings)


@asynccontextmanager
async def lifespan(_app):
    service.start_cleanup_worker()
    db.start_recovery_monitor()
    catalog_import.start_if_enabled()
    try:
        yield
    finally:
        db.stop_recovery_monitor()
        service.stop_cleanup_worker()


app = FastAPI(title="Audio Mixer Backend", version=settings.app_version, lifespan=lifespan)
router = APIRouter()
catalog_import = CatalogImportManager(settings)
catalog_import.database_url_provider = lambda: db.active_url
service_admin = ServiceAdmin(db, service, ffmpeg, catalog_import)


class CategoryIn(BaseModel):
    slug: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    description: str = ""


class KeyIn(BaseModel):
    name: str
    description: str = ""
    scopes: list[str] = ["audio:read", "audio:write", "audio:process"]
    expires_at: str | None = None


class RecommendationIn(BaseModel):
    recommended: bool


class SettingsPatch(BaseModel):
    settings: dict[str, int | float | str]


class MaintenanceRequest(BaseModel):
    action: str
    dry_run: bool = True


def resolve_noise_gain(noise_gain_db, noise_mix_percent):
    if noise_mix_percent is None:
        return noise_gain_db
    if not 0 <= noise_mix_percent <= 100:
        raise HTTPException(422, "noise_mix_percent must be between 0 and 100")
    return -60 if noise_mix_percent == 0 else 20 * math.log10(noise_mix_percent / 100)


def auth(required="audio:read", x_api_key: str | None = Header(default=None)):
    bootstrap = os.getenv("AUDIO_MIXER_API_KEY")
    row = db.verify_key(x_api_key) if x_api_key else None
    if not row and (
        not bootstrap
        or not x_api_key
        or not secrets.compare_digest(bootstrap, x_api_key)
    ):
        raise UnauthorizedError()
    if (
        row
        and required not in json.loads(row["scopes_json"])
        and "audio:admin" not in json.loads(row["scopes_json"])
    ):
        raise ForbiddenError()
    return row


def require_read(x_api_key: str | None = Header(default=None)):
    return auth("audio:read", x_api_key)


def require_admin(x_api_key: str | None = Header(default=None)):
    return auth("audio:admin", x_api_key)


def require_write(x_api_key: str | None = Header(default=None)):
    return auth("audio:write", x_api_key)


def require_process(x_api_key: str | None = Header(default=None)):
    return auth("audio:process", x_api_key)


def require_jobs(x_api_key: str | None = Header(default=None)):
    return auth("jobs:read", x_api_key)


@app.exception_handler(AudioMixerError)
async def err(_, e):
    return JSONResponse(
        status_code=e.status, content={"error": {"code": e.code, "message": e.message}}
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "audio-mixer",
        "version": settings.app_version,
    }


@app.get("/ready")
def ready():
    try:
        db.init()
        cap = ffmpeg.capabilities()
        database_status = db.status()
        return {
            "status": "ready",
            "ffmpeg": cap["available"],
            "database": "ready",
            "database_mode": database_status["mode"],
            "database_fallback_active": database_status["fallback_active"],
            "audio_storage": "postgresql_bytea" if db.postgres else "sqlite_blob",
        }
    except Exception:
        raise HTTPException(503, "dependencies_unavailable")


@router.get("/api/v1/audio/capabilities")
def capabilities(_: object = Depends(require_read)):
    return ffmpeg.capabilities()


@router.get("/api/v1/audio/catalog/import-status")
def catalog_import_status(_: object = Depends(require_read)):
    return catalog_import.status()


@router.get("/api/v1/audio/stats")
def stats(_: object = Depends(require_admin)):
    return service.stats()


@router.post("/api/v1/audio/maintenance/cleanup")
def cleanup_expired_data(
    dry_run: bool = Query(False), _: object = Depends(require_admin)
):
    return service.cleanup_expired_data(dry_run=dry_run)


def _require_known_service(service_name):
    if not service_admin.has_service(service_name):
        raise HTTPException(404, "service_not_found")


@router.get("/api/v1/audio/admin/services")
def admin_services(_: object = Depends(require_admin)):
    return {"services": service_admin.list_services()}


@router.get("/api/v1/audio/admin/services/{service_name}/health")
def admin_service_health(service_name: str, _: object = Depends(require_admin)):
    _require_known_service(service_name)
    return service_admin.health(service_name)


@router.get("/api/v1/audio/admin/services/{service_name}/errors")
def admin_service_errors(service_name: str, _: object = Depends(require_admin)):
    _require_known_service(service_name)
    return {"service": service_name, "errors": service_admin.errors(service_name)}


@router.get("/api/v1/audio/admin/services/{service_name}/settings")
def admin_service_settings(service_name: str, _: object = Depends(require_admin)):
    _require_known_service(service_name)
    return service_admin.settings(service_name)


@router.patch("/api/v1/audio/admin/services/{service_name}/settings")
def update_admin_service_settings(
    service_name: str,
    payload: SettingsPatch,
    dry_run: bool = Query(True),
    _: object = Depends(require_admin),
):
    _require_known_service(service_name)
    try:
        return service_admin.update_settings(service_name, payload.settings, dry_run)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/api/v1/audio/admin/services/{service_name}/maintenance")
def admin_service_maintenance(
    service_name: str,
    payload: MaintenanceRequest,
    _: object = Depends(require_admin),
):
    _require_known_service(service_name)
    try:
        return service_admin.maintenance(service_name, payload.action, payload.dry_run)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/api/v1/audio/admin/services/{service_name}/audit")
def admin_service_audit(
    service_name: str,
    limit: int = Query(50, ge=1, le=200),
    _: object = Depends(require_admin),
):
    _require_known_service(service_name)
    return {"service": service_name, "events": service_admin.audit(service_name, limit)}


@router.get("/api/v1/audio/categories")
def categories(_: object = Depends(require_read)):
    return [
        {"id": row["id"], "name": row["title"], "description": row["description"]}
        for row in service.categories()
    ]


@router.post("/api/v1/audio/categories")
def create_category(x: CategoryIn, _: object = Depends(require_admin)):
    return {"id": db.seed_category(x.slug, x.title, x.description), **x.model_dump()}


@router.get("/api/v1/audio/assets")
def assets(
    category: str | None = None,
    search: str | None = None,
    recommended: bool | None = None,
    limit: int | None = Query(None, le=200),
    offset: int = 0,
    _: object = Depends(require_read),
):
    return service.assets(category, search, limit, offset, recommended)


@router.get("/api/v1/audio/assets/recommended")
def recommended_assets(
    limit: int | None = Query(None, le=200),
    offset: int = 0,
    _: object = Depends(require_read),
):
    return [
        {"id": row["id"], "name": row["title"], "description": row["description"]}
        for row in service.recommended_assets(limit, offset)
    ]


@router.get("/api/v1/audio/assets/category/{category_id}")
def assets_by_category(
    category_id: str,
    limit: int | None = Query(None, le=200),
    offset: int = 0,
    _: object = Depends(require_read),
):
    return [
        {"id": row["id"], "name": row["title"], "description": row["description"]}
        for row in service.assets(category=category_id, limit=limit, offset=offset)
    ]


@router.get("/api/v1/audio/assets/{aid}/download")
def download_asset(aid: str, _: object = Depends(require_read)):
    data, filename, content_type = service.asset_bytes(aid)
    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/api/v1/audio/assets/{aid}")
def asset(aid: str, _: object = Depends(require_read)):
    return service.asset(aid)


@router.post("/api/v1/audio/assets")
def upload_asset(
    file: UploadFile = File(...),
    title: str = Form(""),
    description: str = Form(""),
    category_id: str | None = Form(None),
    _: object = Depends(require_write),
):
    return service.ingest(file, title, description, category_id)


@router.delete("/api/v1/audio/assets/{aid}")
def delete_asset(aid: str, _: object = Depends(require_admin)):
    return service.delete_asset(aid)


@router.patch("/api/v1/audio/assets/{aid}/recommendation")
def set_recommendation(
    aid: str, payload: RecommendationIn, _: object = Depends(require_admin)
):
    if not db.set_recommended(aid, payload.recommended):
        raise HTTPException(404, "asset_not_found")
    return {"id": aid, "recommended": payload.recommended}


@router.post("/api/v1/audio/mixes/from-library", status_code=202)
def mix_library(
    source_audio: UploadFile = File(...),
    noise_asset_id: str = Form(...),
    source_gain_db: float = Form(0),
    noise_gain_db: float = Form(-18),
    variation_seed: int | None = Form(None),
    output_format: str | None = Form(None),
    _: object = Depends(require_process),
):
    source = service._read_upload(source_audio)
    noise, _, _, noise_metadata = service.asset_audio_bundle(noise_asset_id)
    return {
        "job_id": service.submit_mix_bytes(
            source,
            source_audio.filename or "source.audio",
            noise,
            noise_asset_id,
            source_gain_db,
            noise_gain_db,
            variation_seed,
            output_format,
            noise_metadata=noise_metadata,
        ),
        "status": "queued",
    }


@router.post("/api/v1/audio/mixes/from-upload", status_code=202)
def mix_upload(
    source_audio: UploadFile = File(...),
    noise_audio: UploadFile = File(...),
    source_gain_db: float = Form(0),
    noise_gain_db: float = Form(-18),
    variation_seed: int | None = Form(None),
    output_format: str | None = Form(None),
    _: object = Depends(require_process),
):
    source = service._read_upload(source_audio)
    noise = service._read_upload(noise_audio)
    return {
        "job_id": service.submit_mix_bytes(
            source,
            source_audio.filename or "source.audio",
            noise,
            noise_audio.filename or "noise.audio",
            source_gain_db,
            noise_gain_db,
            variation_seed,
            output_format,
        ),
        "status": "queued",
    }


@router.post("/api/v1/audio/process/from-library")
def process_from_library(
    source_audio: UploadFile = File(...),
    noise_asset_id: str = Form(...),
    source_gain_db: float = Form(0),
    noise_gain_db: float = Form(-18),
    noise_mix_percent: float | None = Form(None),
    variation_seed: int | None = Form(None),
    output_format: str | None = Form(None),
    _: object = Depends(require_process),
):
    noise, _, _, noise_metadata = service.asset_audio_bundle(noise_asset_id)
    noise_gain_db = resolve_noise_gain(noise_gain_db, noise_mix_percent)
    data = service.process_sync_uploads(
        source_audio,
        noise_bytes=noise,
        noise_reference=noise_asset_id,
        source_gain=source_gain_db,
        noise_gain=noise_gain_db,
        variation_seed=variation_seed,
        output_format=output_format,
        noise_metadata=noise_metadata,
    )
    output_format = service.resolve_output_format(output_format)
    spec = ffmpeg.output_spec(output_format)
    return Response(
        content=data,
        media_type=spec["media_type"],
        headers={"Content-Disposition": f'attachment; filename="processed{spec["suffix"]}"'},
    )


@router.post("/api/v1/audio/process/from-upload")
def process_from_upload(
    source_audio: UploadFile = File(...),
    noise_audio: UploadFile = File(...),
    source_gain_db: float = Form(0),
    noise_gain_db: float = Form(-18),
    noise_mix_percent: float | None = Form(None),
    variation_seed: int | None = Form(None),
    output_format: str | None = Form(None),
    _: object = Depends(require_process),
):
    noise_gain_db = resolve_noise_gain(noise_gain_db, noise_mix_percent)
    data = service.process_sync_uploads(
        source_audio,
        noise_upload=noise_audio,
        source_gain=source_gain_db,
        noise_gain=noise_gain_db,
        variation_seed=variation_seed,
        output_format=output_format,
    )
    output_format = service.resolve_output_format(output_format)
    spec = ffmpeg.output_spec(output_format)
    return Response(
        content=data,
        media_type=spec["media_type"],
        headers={"Content-Disposition": f'attachment; filename="processed{spec["suffix"]}"'},
    )


@router.get("/api/v1/audio/jobs")
def jobs(
    status: str | None = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    _: object = Depends(require_jobs),
):
    return service.jobs(status, limit, offset)


@router.get("/api/v1/audio/jobs/{jid}")
def job(jid: str, _: object = Depends(require_jobs)):
    return service.job(jid)


@router.get("/api/v1/audio/jobs/{jid}/status")
def job_status(jid: str, _: object = Depends(require_jobs)):
    return service.job_status(jid)


@router.get("/api/v1/audio/jobs/{jid}/result")
def result(jid: str, _: object = Depends(require_jobs)):
    result = service.job_result_info(jid)
    return Response(
        content=result["data"],
        media_type=result["media_type"],
        headers={"Content-Disposition": f'attachment; filename="{result["filename"]}"'},
    )


@router.post("/api/v1/audio/jobs/{jid}/cancel")
def cancel_job(jid: str, _: object = Depends(require_process)):
    return service.cancel_job(jid)


@router.post("/api/v1/audio/jobs/{jid}/retry", status_code=202)
def retry_job(jid: str, _: object = Depends(require_process)):
    return {"job_id": service.retry_job(jid), "status": "queued"}


@router.post("/api/v1/audio/api-keys")
def create_key(x: KeyIn, _: object = Depends(require_admin)):
    kid, raw = db.create_key(x.name, x.description, x.scopes, x.expires_at)
    return {
        "id": kid,
        "secret": raw,
        "warning": "Store this secret now; it will not be shown again.",
    }


@router.get("/api/v1/audio/api-keys")
def list_keys(_: object = Depends(require_admin)):
    return db.list_keys()


@router.post("/api/v1/audio/api-keys/{key_id}/revoke")
def revoke_key(key_id: str, _: object = Depends(require_admin)):
    if not db.revoke_key(key_id):
        raise HTTPException(404, "api_key_not_found")
    return {"id": key_id, "status": "revoked"}


# Keep a standalone ASGI app for independent deployment while exposing the
# same route set for composition inside another FastAPI application.
app.include_router(router)


def include_audio_mixer(host_app: FastAPI, *, prefix: str = ""):
    """Install audio routes, error mapping, and lifecycle into a host FastAPI app."""
    if getattr(host_app.state, "audio_mixer_installed", False):
        return host_app

    host_app.include_router(router, prefix=prefix)
    if AudioMixerError not in host_app.exception_handlers:
        host_app.add_exception_handler(AudioMixerError, err)

    host_lifespan = host_app.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(app_instance):
        async with host_lifespan(app_instance), lifespan(app_instance):
            yield

    host_app.router.lifespan_context = combined_lifespan
    host_app.state.audio_mixer_installed = True
    return host_app
