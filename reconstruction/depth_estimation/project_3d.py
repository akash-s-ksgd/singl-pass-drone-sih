import cv2
import numpy as np
import open3d as o3d
import os
import glob
from pathlib import Path
from loguru import logger

class VibeProjector:
    """
    Enterprise-grade 3D Projection Engine. 
    Converts directories of 2D RGB frames and AI Depth Maps into Dense 3D Point Clouds.
    """
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self.frames_dir = self.base_dir / "data" / "frames"
        self.depth_dir = self.base_dir / "data" / "outputs" / "depth_maps"
        self.dense_dir = self.base_dir / "data" / "outputs" / "dense"
        self.dense_dir.mkdir(parents=True, exist_ok=True)

    def process_all_frames(self):
        logger.info("🌌 IGNITING MATRIX PROJECTION FOR ALL FRAMES...")
        frames = sorted(glob.glob(str(self.frames_dir / "*.jpg")) + glob.glob(str(self.frames_dir / "*.png")))
        
        if not frames:
            logger.error(f"❌ No frames found in {self.frames_dir}")
            return False

        successful_projections = 0

        for frame_path in frames:
            target_frame = Path(frame_path)
            target_depth = self.depth_dir / f"depth_{target_frame.name}"

            if not target_depth.exists():
                logger.warning(f"⚠️ Skipping {target_frame.name} - No matching depth map found.")
                continue

            try:
                self._project_single_frame(target_frame, target_depth)
                successful_projections += 1
            except Exception as e:
                logger.error(f"❌ Failed to project {target_frame.name}: {str(e)}")

        logger.info(f"✅ BATCH COMPLETE: Successfully forged {successful_projections}/{len(frames)} 3D models.")
        return successful_projections > 0

    def _project_single_frame(self, target_frame: Path, target_depth: Path):
        img = cv2.imread(str(target_frame))
        depth = cv2.imread(str(target_depth), cv2.IMREAD_GRAYSCALE)

        if img is None or depth is None:
            raise ValueError("OpenCV failed to load images.")

        if img.shape[:2] != depth.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = depth.shape

        fx, fy = w * 0.8, w * 0.8
        cx, cy = w // 2, h // 2
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Absolute Brute-Force Matrix Extraction
        z = np.clip((depth.astype(np.float64) / 255.0) * 10.0, 0.1, 10.0)
        u, v, z = u.flatten(), v.flatten(), z.flatten()
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.stack((x, y, z), axis=-1).astype(np.float64)
        colors = (img.reshape(-1, 3) / 255.0).astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        out_file = self.dense_dir / f"dense_{target_frame.stem}.ply"
        if not o3d.io.write_point_cloud(str(out_file), pcd):
            raise RuntimeError("Open3D write operation failed.")
        
        logger.debug(f"Saved: {out_file.name} ({len(pcd.points):,} points)")

if __name__ == "__main__":
    projector = VibeProjector(os.getcwd())
    projector.process_all_frames()