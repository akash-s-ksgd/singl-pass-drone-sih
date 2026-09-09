import os
import shutil
import gc
from pathlib import Path
from loguru import logger

# Import our God-Tier Modules
from reconstruction.pipeline import ReconstructionPipeline
from reconstruction.depth_estimation.ai_depth import VibeDepthEngine
from reconstruction.depth_estimation.project_3d import VibeProjector

class MasterOrchestrator:
    """
    The central nervous system of the 3D Reconstruction Pipeline.
    Engineered with Just-In-Time (JIT) memory management to survive local hardware limits.
    """
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or os.getcwd()).resolve()
        
        # Define standard directories
        self.upload_dir = self.base_dir / "data" / "uploads"
        self.frames_dir = self.base_dir / "data" / "frames"
        self.outputs_dir = self.base_dir / "data" / "outputs"
        self.depth_dir = self.outputs_dir / "depth_maps"
        self.dense_dir = self.outputs_dir / "dense"

        # Initialize lightweight engines only (DO NOT load AI here)
        self.colmap_pipeline = ReconstructionPipeline(colmap_path=r"P:\Downloads\COLMAP\colmap.bat")
        self.projector = VibeProjector(base_dir=self.base_dir)

    def _wipe_previous_run(self):
        """Clears old artifacts so we don't mix up 3D models from different videos."""
        logger.info("🧹 Wiping previous pipeline artifacts...")
        for d in [self.frames_dir, self.depth_dir, self.dense_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            
        # Clean colmap database
        db_path = self.outputs_dir / "database.db"
        if db_path.exists():
            db_path.unlink()

    def process_video(self, video_filename: str) -> bool:
        video_path = self.upload_dir / video_filename
        if not video_path.exists():
            logger.error(f"❌ Target video not found: {video_path}")
            return False

        logger.info(f"{'='*60}")
        logger.info(f"🚀 INITIATING GOD-TIER PIPELINE FOR: {video_filename}")
        logger.info(f"{'='*60}")

        try:
            self._wipe_previous_run()
            
            # Flush RAM before Phase 1
            gc.collect()

            # ---------------------------------------------------------
            # PHASE 1: Preprocessing & Sparse Geometry
            # ---------------------------------------------------------
            logger.info("▶️ PHASE 1: Video Extraction & Spatial Tracking")
            success = self.colmap_pipeline.run(video_filename=video_filename)
            if not success:
                raise RuntimeError("COLMAP Pipeline failed.")

            # ---------------------------------------------------------
            # PHASE 2: Neural Depth Hallucination (JIT Loading)
            # ---------------------------------------------------------
            logger.info("▶️ PHASE 2: Neural Depth Hallucination")
            # We ONLY instantiate the AI now, when we actually need it.
            depth_engine = VibeDepthEngine(output_dir=self.depth_dir)
            depth_engine.process_directory(self.frames_dir)
            
            # The AI is done. Assassinate the process and free the RAM.
            logger.info("🧹 Phase 2 complete. Purging Neural Core from memory...")
            del depth_engine
            gc.collect()

            # ---------------------------------------------------------
            # PHASE 3: Dense Matrix Projection
            # ---------------------------------------------------------
            logger.info("▶️ PHASE 3: 3D Point Cloud Forging")
            success = self.projector.process_all_frames()
            if not success:
                raise RuntimeError("3D Projection failed.")

            logger.info(f"{'='*60}")
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY.")
            logger.info(f"💾 Final dense geometry available in: {self.dense_dir}")
            logger.info(f"{'='*60}")
            return True

        except Exception as e:
            logger.exception(f"❌ PIPELINE CRITICAL FAILURE: {str(e)}")
            return False

if __name__ == "__main__":
    # Test the Master Orchestrator directly
    engine = MasterOrchestrator()
    engine.process_video("test_flight.mp4")