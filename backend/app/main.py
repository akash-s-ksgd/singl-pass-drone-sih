"""
FastAPI application entry point.
Sets up CORS, static file serving, and mounts all API routes.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.app.api.routes import router as api_router
from backend.app.config import settings

# ─── App Setup ───────────────────────────────────────────────────────

app = FastAPI(
    title="Drone 3D Reconstruction API",
    description=(
        "Single-Pass Drone Video to Accurate 3D Model Generation System. "
        "A confidence-aware, sensor-fused hybrid reconstruction pipeline."
    ),
    version="0.1.0",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api")

# Serve frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


# ─── Startup ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    settings.ensure_dirs()
    print("=" * 60)
    print("  Drone 3D Reconstruction System - API Ready")
    print(f"  Upload dir:  {settings.upload_dir}")
    print(f"  Output dir:  {settings.output_dir}")
    print(f"  GPU:         {settings.device}")
    print("=" * 60)


# ─── CLI Entry ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
