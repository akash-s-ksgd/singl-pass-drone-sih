import os
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# 1. Initialize the API Matrix
app = FastAPI(
    title="Single-Pass Reconnaissance API",
    description="God-Tier Backend for AI-Driven 3D Reconstruction",
    version="1.0.0"
)

# Allow our future Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. System Paths
BASE_DIR = Path(os.getcwd()).resolve()
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OUTPUT_DIR = BASE_DIR / "data" / "outputs" / "dense"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. The Background Engine
def run_heavy_pipeline(video_path: str, filename: str):
    """
    Executes the heavy C++/AI math in the background so the API never blocks.
    """
    logger.info(f"🚀 BACKGROUND WORKER: Initiating God-Tier Pipeline for {filename}...")
    
    try:
        # Step 1: Extract, Filter, and COLMAP Camera Tracking
        logger.info("Executing Phase 1: COLMAP Sparse Reconstruction...")
        subprocess.run(["python", "-m", "reconstruction.pipeline"], check=True)
        
        # Step 2: AI Depth Hallucination
        logger.info("Executing Phase 2: Neural Depth Hallucination...")
        subprocess.run(["python", "-m", "reconstruction.depth_estimation.ai_depth"], check=True)
        
        # Step 3: 3D Point Cloud Forge
        logger.info("Executing Phase 3: Matrix Projection...")
        subprocess.run(["python", "-m", "reconstruction.depth_estimation.project_3d"], check=True)
        
        logger.info(f"✅ BACKGROUND WORKER: Pipeline Complete for {filename}!")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ BACKGROUND WORKER FAILED: The pipeline crashed at step execution.")
        logger.error(str(e))

# 4. The API Endpoints
@app.get("/")
def health_check():
    return {"status": "VIBE CODER API ONLINE", "version": "1.0.0"}

@app.post("/api/v1/process")
async def upload_and_process(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Ingests the video, saves it to disk, and immediately kicks off the background pipeline.
    """
    if not file.filename.endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Invalid file type. Send a video.")

    # Save the payload securely
    file_location = UPLOAD_DIR / file.filename
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    logger.info(f"Payload secured: {file.filename}")

    # Fire the background task
    background_tasks.add_task(run_heavy_pipeline, str(file_location), file.filename)

    return JSONResponse(content={
        "status": "PROCESSING_INITIATED",
        "message": f"{file.filename} is being processed in the background.",
        "filename": file.filename
    })

@app.get("/api/v1/models")
def list_available_models():
    """
    Returns a list of all successfully generated .ply models ready for the UI.
    """
    models = [f.name for f in OUTPUT_DIR.glob("*.ply")]
    return {"available_models": models}