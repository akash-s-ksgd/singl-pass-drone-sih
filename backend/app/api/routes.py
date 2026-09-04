"""
FastAPI routes for the drone 3D reconstruction API.
Handles video upload, job management, model retrieval, and progress tracking.
"""
from __future__ import annotations
import uuid
import time
import json
import asyncio
import shutil
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from backend.app.config import settings
from backend.app.models.schemas import (
    ProjectCreate, ProjectResponse, ProcessingConfig,
    UploadResponse, JobProgress, JobStatus,
    ReconstructionReport, ModelDownloadInfo, FlightMetadata,
)

router = APIRouter()

# ─── In-memory stores (replaced by DB in production) ────────────────
projects: Dict[str, dict] = {}
jobs: Dict[str, JobProgress] = {}
reports: Dict[str, ReconstructionReport] = {}


def _get_media_info(file_path: Path) -> dict:
    """Extract basic media information using OpenCV."""
    import cv2
    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        img = cv2.imread(str(file_path))
        return {
            "duration": 0.0,
            "fps": 1.0,
            "width": int(img.shape[1]) if img is not None else 0,
            "height": int(img.shape[0]) if img is not None else 0,
            "frame_count": 1,
            "is_image": True
        }
    else:
        cap = cv2.VideoCapture(str(file_path))
        info = {
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "is_image": False
        }
        cap.release()
        return info


# ─── Project & Upload ───────────────────────────────────────────────

@router.post("/projects", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """Create a new reconstruction project."""
    project_id = str(uuid.uuid4())[:8]
    proj = {
        "id": project_id,
        "name": project.name,
        "description": project.description or "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": JobStatus.PENDING,
        "video_filename": None,
    }
    projects[project_id] = proj
    # Create project directories
    (settings.upload_dir / project_id).mkdir(parents=True, exist_ok=True)
    (settings.frames_dir / project_id).mkdir(parents=True, exist_ok=True)
    (settings.output_dir / project_id).mkdir(parents=True, exist_ok=True)
    return ProjectResponse(**proj)


@router.get("/projects", response_model=list[ProjectResponse])
async def list_projects():
    """List all projects."""
    return [ProjectResponse(**p) for p in projects.values()]


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    """Get project details."""
    if project_id not in projects:
        raise HTTPException(404, "Project not found")
    return ProjectResponse(**projects[project_id])


@router.post("/projects/{project_id}/upload", response_model=UploadResponse)
async def upload_video(project_id: str, video: UploadFile = File(...)):
    """Upload a drone video to a project."""
    if project_id not in projects:
        raise HTTPException(404, "Project not found")

    if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.jpg', '.jpeg', '.png')):
        raise HTTPException(400, "Unsupported media format. Use MP4, MOV, AVI, MKV, JPG, or PNG.")

    # Save the uploaded video
    upload_path = settings.upload_dir / project_id / video.filename
    with open(upload_path, "wb") as f:
        content = await video.read()
        f.write(content)

    file_size_mb = upload_path.stat().st_size / (1024 * 1024)
    projects[project_id]["video_filename"] = video.filename

    # Get media info
    info = _get_media_info(upload_path)

    return UploadResponse(
        project_id=project_id,
        filename=video.filename,
        file_size_mb=round(file_size_mb, 2),
        video_duration_s=round(info["duration"], 1),
        resolution=f"{info['width']}x{info['height']}",
    )


@router.post("/projects/{project_id}/metadata")
async def upload_metadata(project_id: str, metadata: FlightMetadata):
    """Upload flight metadata (GPS, IMU, camera intrinsics)."""
    if project_id not in projects:
        raise HTTPException(404, "Project not found")
    projects[project_id]["metadata"] = metadata.model_dump()
    return {"message": "Metadata uploaded", "gps_points": len(metadata.gps_points)}


# ─── Processing ──────────────────────────────────────────────────────

@router.post("/projects/{project_id}/process")
async def start_processing(
    project_id: str,
    config: ProcessingConfig,
    background_tasks: BackgroundTasks,
):
    """Start the reconstruction pipeline as a background job."""
    if project_id not in projects:
        raise HTTPException(404, "Project not found")
    if not projects[project_id].get("video_filename"):
        raise HTTPException(400, "No video uploaded yet")

    job_id = f"job-{project_id}-{str(uuid.uuid4())[:6]}"
    job = JobProgress(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        current_step="Initializing pipeline...",
    )
    jobs[job_id] = job
    projects[project_id]["status"] = JobStatus.PENDING
    projects[project_id]["current_job"] = job_id

    # Run pipeline in background
    background_tasks.add_task(
        _run_pipeline, project_id, job_id, config
    )

    return {"job_id": job_id, "message": "Processing started"}


async def _run_pipeline(project_id: str, job_id: str, config: ProcessingConfig):
    """Execute the full reconstruction pipeline."""
    from reconstruction.pipeline import ReconstructionPipeline

    job = jobs[job_id]
    start_time = time.time()

    try:
        video_path = (
            settings.upload_dir / project_id / projects[project_id]["video_filename"]
        )
        metadata = projects[project_id].get("metadata")
        flight_meta = FlightMetadata(**metadata) if metadata else None

        pipeline = ReconstructionPipeline(
            video_path=video_path,
            project_id=project_id,
            config=config,
            flight_metadata=flight_meta,
            progress_callback=lambda status, progress, msg: _update_job(
                job_id, status, progress, msg
            ),
        )

        report = await asyncio.to_thread(pipeline.run)

        report.job_id = job_id
        report.project_name = projects[project_id]["name"]
        report.processing_time_seconds = round(time.time() - start_time, 1)
        reports[job_id] = report

        _update_job(job_id, JobStatus.COMPLETED, 1.0, "Reconstruction complete!")
        projects[project_id]["status"] = JobStatus.COMPLETED

    except Exception as e:
        _update_job(job_id, JobStatus.FAILED, job.progress, str(e))
        jobs[job_id].error = str(e)
        projects[project_id]["status"] = JobStatus.FAILED


def _update_job(job_id: str, status: JobStatus, progress: float, message: str):
    """Update job progress."""
    if job_id in jobs:
        jobs[job_id].status = status
        jobs[job_id].progress = min(progress, 1.0)
        jobs[job_id].message = message
        jobs[job_id].current_step = message


# ─── Job Status & Results ───────────────────────────────────────────

@router.get("/jobs/{job_id}", response_model=JobProgress)
async def get_job_status(job_id: str):
    """Get processing job status and progress."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@router.get("/jobs/{job_id}/report", response_model=ReconstructionReport)
async def get_report(job_id: str):
    """Get the reconstruction quality report."""
    if job_id not in reports:
        raise HTTPException(404, "Report not available yet")
    return reports[job_id]


# ─── Model Download ─────────────────────────────────────────────────

@router.get("/projects/{project_id}/models")
async def list_models(project_id: str):
    """List available 3D model files for a project."""
    if project_id not in projects:
        raise HTTPException(404, "Project not found")

    output_dir = settings.output_dir / project_id
    models = []
    for ext in ["*.ply", "*.obj", "*.glb", "*.las"]:
        for f in output_dir.glob(ext):
            models.append(ModelDownloadInfo(
                format=f.suffix[1:].upper(),
                filename=f.name,
                size_mb=round(f.stat().st_size / (1024 * 1024), 2),
                download_url=f"/api/projects/{project_id}/models/{f.name}",
            ))
    return models


@router.get("/projects/{project_id}/models/{filename}")
async def download_model(project_id: str, filename: str):
    """Download a 3D model file."""
    file_path = settings.output_dir / project_id / filename
    if not file_path.exists():
        raise HTTPException(404, "Model file not found")
    return FileResponse(file_path, filename=filename)


@router.get("/projects/{project_id}/pointcloud")
async def get_pointcloud_data(project_id: str):
    """Get point cloud data as JSON for web visualization."""
    output_dir = settings.output_dir / project_id
    json_path = output_dir / "pointcloud_preview.json"
    if json_path.exists():
        return FileResponse(json_path, media_type="application/json")

    # Try to generate from PLY
    ply_path = output_dir / "dense_confidence.ply"
    if not ply_path.exists():
        ply_path = output_dir / "dense.ply"
    if not ply_path.exists():
        raise HTTPException(404, "No point cloud available")

    # Convert to JSON preview (downsampled)
    import numpy as np
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        # Downsample for web
        if len(pcd.points) > 50000:
            pcd = pcd.uniform_down_sample(max(1, len(pcd.points) // 50000))
        points = np.asarray(pcd.points).tolist()
        colors = np.asarray(pcd.colors).tolist() if pcd.has_colors() else []
        data = {"points": points, "colors": colors}
        with open(json_path, "w") as f:
            json.dump(data, f)
        return JSONResponse(data)
    except ImportError:
        raise HTTPException(500, "Open3D not installed")


@router.get("/projects/{project_id}/trajectory")
async def get_trajectory(project_id: str):
    """Get estimated camera trajectory as JSON."""
    traj_path = settings.output_dir / project_id / "camera_trajectory.json"
    if not traj_path.exists():
        raise HTTPException(404, "Trajectory not available")
    return FileResponse(traj_path, media_type="application/json")


# ─── Health ──────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """API health check with system info."""
    gpu_available = False
    gpus = []

    if _check_import("torch"):
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            for i in range(torch.cuda.device_count()):
                gpus.append({"id": f"cuda:{i}", "name": torch.cuda.get_device_name(i)})

    colmap_available = shutil.which(settings.colmap_path) is not None

    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpus": gpus,
        "colmap_available": colmap_available,
        "active_jobs": sum(
            1 for j in jobs.values()
            if j.status not in (JobStatus.COMPLETED, JobStatus.FAILED)
        ),
    }


def _check_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False
