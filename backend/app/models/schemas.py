"""
Pydantic schemas for API request/response models and internal data structures.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# ─── Enums ───────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING_FRAMES = "extracting_frames"
    SELECTING_FRAMES = "selecting_frames"
    ESTIMATING_POSES = "estimating_poses"
    ESTIMATING_DEPTH = "estimating_depth"
    RECONSTRUCTING = "reconstructing"
    GENERATING_MESH = "generating_mesh"
    GEOREFERENCING = "georeferencing"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ─── GPS & Sensor Data ──────────────────────────────────────────────

class GPSPoint(BaseModel):
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    timestamp: Optional[float] = None
    accuracy: Optional[float] = None


class IMUData(BaseModel):
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


class FlightMetadata(BaseModel):
    gps_points: List[GPSPoint] = []
    imu_data: List[IMUData] = []
    camera_intrinsics: Optional[CameraIntrinsics] = None
    drone_model: Optional[str] = None
    flight_altitude: Optional[float] = None
    capture_date: Optional[str] = None


# ─── Frame Quality ───────────────────────────────────────────────────

class FrameQuality(BaseModel):
    frame_index: int
    timestamp: float
    sharpness_score: float = 0.0
    feature_count: int = 0
    brightness_score: float = 0.0
    contrast_score: float = 0.0
    blur_score: float = 0.0
    exposure_quality: float = 0.0
    composite_score: float = 0.0
    selected: bool = False
    gps: Optional[GPSPoint] = None


# ─── Reconstruction ─────────────────────────────────────────────────

class CameraPose(BaseModel):
    frame_index: int
    rotation: List[List[float]]       # 3x3 rotation matrix
    translation: List[float]          # 3D translation vector
    confidence: float = 1.0
    source: str = "visual"            # visual, sensor, fused


class DepthEstimate(BaseModel):
    frame_index: int
    method: str                       # mvs, monocular, fused
    min_depth: float
    max_depth: float
    confidence_mean: float


class ConfidenceRegion(BaseModel):
    level: ConfidenceLevel
    point_count: int
    percentage: float
    mean_confidence: float


# ─── Job & Project ───────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    status: JobStatus
    video_filename: Optional[str] = None


class ProcessingConfig(BaseModel):
    fps: float = 2.0
    min_sharpness: float = 50.0
    min_feature_count: int = 100
    max_frames: int = 500
    use_gpu: bool = True
    gpu_device: Optional[str] = None
    enable_depth_fusion: bool = True
    enable_neural_refinement: bool = False
    enable_dynamic_removal: bool = True
    enable_georeferencing: bool = True


class JobProgress(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0            # 0.0 to 1.0
    current_step: str = ""
    message: str = ""
    frames_extracted: int = 0
    frames_selected: int = 0
    points_reconstructed: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


class ReconstructionReport(BaseModel):
    job_id: str
    project_name: str
    processing_time_seconds: float
    frames_extracted: int
    frames_selected: int
    camera_poses_recovered: int
    sparse_points: int
    dense_points: int
    mesh_vertices: int
    mesh_faces: int
    confidence_distribution: List[ConfidenceRegion]
    georeferencing_error_m: Optional[float] = None
    output_files: Dict[str, str] = {}
    metrics: Dict[str, float] = {}


# ─── API Responses ───────────────────────────────────────────────────

class UploadResponse(BaseModel):
    project_id: str
    filename: str
    file_size_mb: float
    video_duration_s: Optional[float] = None
    resolution: Optional[str] = None
    message: str = "Video uploaded successfully"


class ModelDownloadInfo(BaseModel):
    format: str
    filename: str
    size_mb: float
    download_url: str
