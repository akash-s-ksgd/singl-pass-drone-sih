"""
Application configuration loaded from environment variables.
"""
import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseModel):
    """Application settings."""
    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"

    # Paths
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
    frames_dir: Path = Path(os.getenv("FRAMES_DIR", str(BASE_DIR / "data" / "frames")))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "data" / "outputs")))
    models_dir: Path = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "data" / "models")))

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'drone3d.db'}"
    )

    # COLMAP
    colmap_path: str = os.getenv("COLMAP_PATH", "colmap")

    # GPU
    use_gpu: bool = os.getenv("USE_GPU", "true").lower() == "true"
    device: str = os.getenv("DEVICE", "cuda")

    # Reconstruction defaults
    default_fps: float = float(os.getenv("DEFAULT_FPS", "2"))
    min_sharpness: float = float(os.getenv("MIN_SHARPNESS", "50.0"))
    min_feature_count: int = int(os.getenv("MIN_FEATURE_COUNT", "100"))
    max_frames: int = int(os.getenv("MAX_FRAMES", "500"))

    def ensure_dirs(self):
        """Create all required directories."""
        for d in [self.upload_dir, self.frames_dir, self.output_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
