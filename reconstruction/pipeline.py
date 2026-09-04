import subprocess
import shutil
from pathlib import Path
from typing import List
from loguru import logger

from reconstruction.frame_selection.selector import VideoPreprocessor

class ReconstructionPipeline:
    """
    Manager class for orchestrating Laptop-Mode 3D model generation.
    Strictly linear single-pass constraints to prevent OOM crashes.
    """

    def __init__(
        self,
        upload_dir: str | Path = "data/uploads",
        frames_dir: str | Path = "data/frames",
        output_dir: str | Path = "data/outputs",
        colmap_path: str = "colmap",
    ) -> None:
        """
        Initializes the ReconstructionPipeline manager.
        """
        self.upload_dir = Path(upload_dir)
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.colmap_path = colmap_path

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.output_dir / "database.db"

        # Initialize the video preprocessor with aggressive memory optimizations
        self.preprocessor = VideoPreprocessor(
            upload_dir=self.upload_dir,
            frames_dir=self.frames_dir,
            target_fps=4.0,        # Target 3-5 FPS
            blur_threshold=100.0,  # Quality gate for Laplacian variance
            max_dim=1024           # Aggressive downscaling dimension
        )

    def run(self, video_filename: str) -> bool:
        """
        Executes the reconstruction pipeline: preprocessing followed by COLMAP sparse reconstruction.
        
        Args:
            video_filename: The name of the video file in the upload directory.
            
        Returns:
            True if the pipeline completed successfully, False otherwise.
        """
        logger.info(f"{'='*60}")
        logger.info(f"Starting Reconstruction Pipeline for {video_filename}")
        logger.info(f"Running in STRICT Laptop-Mode (Sequential Matching Only)")
        logger.info(f"{'='*60}")

        try:
            # 1. Video Preprocessing (Frame Extraction, Downscaling & Quality Gate)
            saved_frames = self.preprocessor.process(video_filename)
            if not saved_frames:
                logger.error("No valid frames were extracted. Pipeline aborted.")
                return False

            # 2. Run COLMAP Sparse Reconstruction
            self._run_colmap()

            logger.info(f"{'='*60}")
            logger.info("Pipeline completed successfully.")
            logger.info(f"Sparse reconstruction outputs available in {self.output_dir / 'sparse'}")
            logger.info(f"{'='*60}")
            return True

        except Exception as e:
            logger.exception(f"Reconstruction Pipeline failed: {str(e)}")
            return False

    def _run_subprocess(self, cmd: List[str], step_name: str) -> None:
        """
        Helper to run a subprocess command with live telemetry.
        """
        # Convert list to string for Windows shell execution
        cmd_str = " ".join(cmd)
        logger.info(f"[{step_name}] Executing command: {cmd_str}")
        try:
            # VIBE CODER FIX 1: We drop the PIPE and use shell=True. 
            # This forces Windows to execute the .bat properly and streams 
            # COLMAP's native C++ output directly to your terminal in real-time.
            subprocess.run(
                cmd_str,
                shell=True,
                check=True
            )
            logger.info(f"[{step_name}] Completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"[{step_name}] Command failed with exit code {e.returncode}")
            raise RuntimeError(f"COLMAP subprocess '{step_name}' failed.") from e

    def _run_colmap(self) -> None:
        """
        Orchestrates COLMAP commands. 
        CRITICAL: Uses Absolute Paths to prevent SQLite database corruption.
        """
        if self.database_path.exists():
            logger.info(f"Cleaning up existing database at {self.database_path}")
            self.database_path.unlink()

        sparse_dir = self.output_dir / "sparse"
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # VIBE CODER FIX 2: Absolute paths are bulletproof.
        abs_db = str(self.database_path.resolve())
        abs_frames = str(self.frames_dir.resolve())
        abs_sparse = str(sparse_dir.resolve())

        # 1. Feature Extraction
        extract_cmd = [
            self.colmap_path, "feature_extractor",
            "--database_path", f'"{abs_db}"',
            "--image_path", f'"{abs_frames}"',
            "--ImageReader.single_camera", "1"
        ]
        self._run_subprocess(extract_cmd, "Feature Extraction")

        # 2. Feature Matching (Sequential Matching)
        match_cmd = [
            self.colmap_path, "sequential_matcher",
            "--database_path", f'"{abs_db}"',
            "--SequentialMatching.overlap", "10"
        ]
        self._run_subprocess(match_cmd, "Sequential Matching")

        # 3. Sparse Reconstruction (Mapper)
        mapper_cmd = [
            self.colmap_path, "mapper",
            "--database_path", f'"{abs_db}"',
            "--image_path", f'"{abs_frames}"',
            "--output_path", f'"{abs_sparse}"'
        ]
        self._run_subprocess(mapper_cmd, "Sparse Reconstruction")

# --- ADD THIS TO THE BOTTOM OF pipeline.py ---
if __name__ == "__main__":
    import os
    
    print("🚀 VIBE CODER: INITIATING SINGLE-PASS RECONSTRUCTION PIPELINE...")
    
    # The pipeline already knows to look in data/uploads/
    # We only pass the filename to the pipeline manager.
    target_filename = "test_flight.mp4"
    full_check_path = os.path.join("data", "uploads", target_filename)
    
    if not os.path.exists(full_check_path):
        print(f"❌ CRITICAL ERROR: Test video not found at {full_check_path}")
        print("Please run the yt-dlp download command first.")
        exit(1)
        
    # Instantiate and fire the pipeline using ONLY the filename
    pipeline = ReconstructionPipeline(colmap_path=r"P:\Downloads\COLMAP\colmap.bat") 
    pipeline.run(video_filename=target_filename)