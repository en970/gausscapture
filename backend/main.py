from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes_capturepack import router as capturepack_router
from backend.api.routes_export import router as export_router
from backend.api.routes_jobs import router as jobs_router
from backend.api.routes_preprocess import router as preprocess_router
from backend.api.routes_preview import files_router, router as preview_router
from backend.api.routes_projects import router as projects_router, settings_router
from backend.api.routes_quality import router as quality_router
from backend.api.routes_training import router as training_router
from backend.config import ROOT_DIR


app = FastAPI(title="GaussCapture MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:7860", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects_router)
app.include_router(settings_router)
app.include_router(capturepack_router)
app.include_router(quality_router)
app.include_router(preprocess_router)
app.include_router(training_router)
app.include_router(preview_router)
app.include_router(files_router)
app.include_router(export_router)
app.include_router(jobs_router)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": "GaussCapture MVP"}


frontend_dist = ROOT_DIR / "frontend" / "dist"
mobile_dir = ROOT_DIR / "mobile"
if mobile_dir.exists():
    app.mount("/mobile", StaticFiles(directory=mobile_dir, html=True), name="mobile")

if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    public_placeholder = ROOT_DIR / "backend" / "static_placeholder"
    public_placeholder.mkdir(exist_ok=True)
    (public_placeholder / "index.html").write_text(
        "<h1>GaussCapture backend is running</h1><p>Start the Vite frontend or build frontend/dist.</p>",
        encoding="utf-8",
    )
    app.mount("/", StaticFiles(directory=public_placeholder, html=True), name="frontend-placeholder")
