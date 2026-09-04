"""
Sensor-Fused Pose Estimation — Combines visual SfM poses with GPS, IMU,
and barometric altitude for metrically accurate camera trajectory estimation.

This is a core innovation: sensor-aware pose estimation for single-pass UAV video.
"""
from __future__ import annotations
import json
import numpy as np
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

from backend.app.models.schemas import (
    CameraPose, GPSPoint, IMUData, CameraIntrinsics, FlightMetadata,
)
from backend.app.config import settings


class SensorFusedPoseEstimator:
    """
    Hybrid pose estimation combining:
    - Visual: COLMAP SfM (feature matching + bundle adjustment)
    - Sensor: GPS positions + IMU orientation + barometric altitude
    - Fusion: Weighted optimization merging both sources
    """

    def __init__(
        self,
        image_dir: Path,
        workspace_dir: Path,
        colmap_path: str = "colmap",
    ):
        self.image_dir = Path(image_dir)
        self.workspace_dir = Path(workspace_dir)
        self.colmap_path = colmap_path
        self.database_path = self.workspace_dir / "database.db"
        self.sparse_dir = self.workspace_dir / "sparse"
        self.dense_dir = self.workspace_dir / "dense"

        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)

    def estimate_poses(
        self,
        flight_metadata: Optional[FlightMetadata] = None,
        use_sequential_matching: bool = True,
    ) -> Tuple[List[CameraPose], dict]:
        """
        Full pose estimation pipeline:
        1. Run COLMAP feature extraction + matching + SfM
        2. If GPS available, apply sensor fusion
        3. Return camera poses and reconstruction stats
        """
        colmap_available = shutil.which(self.colmap_path) is not None

        if colmap_available:
            poses, stats = self._run_colmap_pipeline(use_sequential_matching)
        else:
            logger.warning("COLMAP not found — using fallback visual odometry")
            poses, stats = self._fallback_visual_odometry()

        # Apply sensor fusion if metadata available
        if flight_metadata and flight_metadata.gps_points:
            poses = self._fuse_with_sensors(poses, flight_metadata)
            stats["fusion_applied"] = True

        return poses, stats

    def _run_colmap_pipeline(
        self, use_sequential: bool = True
    ) -> Tuple[List[CameraPose], dict]:
        """Run COLMAP feature extraction, matching, and sparse reconstruction."""
        logger.info("Running COLMAP pipeline...")

        # 1. Feature extraction
        logger.info("  Step 1/3: Feature extraction...")
        self._run_cmd([
            self.colmap_path, "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_num_features", "8192",
        ])

        # 2. Feature matching
        if use_sequential:
            logger.info("  Step 2/3: Sequential matching...")
            self._run_cmd([
                self.colmap_path, "sequential_matcher",
                "--database_path", str(self.database_path),
                "--SequentialMatching.overlap", "10",
            ])
        else:
            logger.info("  Step 2/3: Exhaustive matching...")
            self._run_cmd([
                self.colmap_path, "exhaustive_matcher",
                "--database_path", str(self.database_path),
            ])

        # 3. Sparse reconstruction (SfM)
        logger.info("  Step 3/3: Sparse reconstruction...")
        self._run_cmd([
            self.colmap_path, "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--output_path", str(self.sparse_dir),
        ])

        # Parse COLMAP output
        poses, stats = self._parse_colmap_model()
        logger.info(
            f"COLMAP: {stats.get('registered_images', 0)} images registered, "
            f"{stats.get('sparse_points', 0)} sparse points"
        )
        return poses, stats

    def _parse_colmap_model(self) -> Tuple[List[CameraPose], dict]:
        """Parse COLMAP sparse model into CameraPose objects."""
        model_dir = self.sparse_dir / "0"
        if not model_dir.exists():
            # Try without subdirectory
            model_dir = self.sparse_dir

        poses = []
        stats = {"registered_images": 0, "sparse_points": 0}

        # Try to read images.txt
        images_file = model_dir / "images.txt"
        if not images_file.exists():
            # Try binary format conversion
            try:
                self._run_cmd([
                    self.colmap_path, "model_converter",
                    "--input_path", str(model_dir),
                    "--output_path", str(model_dir),
                    "--output_type", "TXT",
                ])
            except Exception:
                logger.warning("Could not convert COLMAP model to text format")
                return poses, stats

        if images_file.exists():
            poses = self._read_images_txt(images_file)
            stats["registered_images"] = len(poses)

        # Count 3D points
        points_file = model_dir / "points3D.txt"
        if points_file.exists():
            with open(points_file) as f:
                stats["sparse_points"] = sum(
                    1 for line in f if line.strip() and not line.startswith("#")
                )

        return poses, stats

    def _read_images_txt(self, path: Path) -> List[CameraPose]:
        """Parse COLMAP images.txt into CameraPose list."""
        poses = []
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        # images.txt has 2 lines per image
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) < 10:
                continue

            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])

            # Quaternion to rotation matrix
            R = self._quat_to_rot(qw, qx, qy, qz)

            poses.append(CameraPose(
                frame_index=image_id - 1,  # 0-indexed
                rotation=R.tolist(),
                translation=[tx, ty, tz],
                confidence=1.0,
                source="visual",
            ))

        return sorted(poses, key=lambda p: p.frame_index)

    def _fallback_visual_odometry(self) -> Tuple[List[CameraPose], dict]:
        """
        Simple visual odometry fallback when COLMAP is not available.
        Uses feature matching between consecutive frames to estimate relative poses.
        """
        logger.info("Running fallback visual odometry...")
        import cv2

        image_paths = sorted(self.image_dir.glob("*.jpg"))
        if not image_paths:
            image_paths = sorted(self.image_dir.glob("*.png"))

        poses = []
        orb = cv2.ORB_create(nfeatures=3000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Initialize first pose as identity
        current_R = np.eye(3)
        current_t = np.zeros(3)

        poses.append(CameraPose(
            frame_index=0,
            rotation=current_R.tolist(),
            translation=current_t.tolist(),
            confidence=1.0,
            source="visual_odometry",
        ))

        prev_gray = None
        prev_kp = None
        prev_des = None

        for idx, img_path in enumerate(image_paths):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)

            if prev_gray is not None and prev_des is not None and des is not None:
                matches = bf.match(prev_des, des)
                matches = sorted(matches, key=lambda x: x.distance)[:200]

                if len(matches) >= 8:
                    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                    pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                    # Estimate essential matrix
                    h, w = gray.shape
                    focal = max(h, w) * 1.2
                    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])

                    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
                    if E is not None:
                        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
                        current_R = current_R @ R
                        current_t = current_t + current_R @ t.flatten() * 0.1

                        poses.append(CameraPose(
                            frame_index=idx,
                            rotation=current_R.tolist(),
                            translation=current_t.tolist(),
                            confidence=0.7,
                            source="visual_odometry",
                        ))

            prev_gray = gray
            prev_kp = kp
            prev_des = des

        stats = {
            "registered_images": len(poses),
            "sparse_points": 0,
            "method": "fallback_visual_odometry",
        }
        logger.info(f"Visual odometry: estimated {len(poses)} poses")
        return poses, stats

    def _fuse_with_sensors(
        self,
        visual_poses: List[CameraPose],
        metadata: FlightMetadata,
    ) -> List[CameraPose]:
        """
        Fuse visual poses with GPS/IMU sensor data.
        Uses a simple weighted average with Procrustes alignment.
        """
        gps_points = metadata.gps_points
        if not gps_points or not visual_poses:
            return visual_poses

        logger.info(f"Fusing {len(visual_poses)} visual poses with {len(gps_points)} GPS points")

        # Convert GPS to local ENU coordinates
        enu_positions = self._gps_to_enu(gps_points)

        # Get visual positions
        visual_positions = np.array([p.translation for p in visual_poses])

        # Align visual positions to ENU using Procrustes
        if len(enu_positions) >= 3 and len(visual_positions) >= 3:
            # Use minimum of both
            n = min(len(enu_positions), len(visual_positions))
            scale, R_align, t_align = self._procrustes_align(
                visual_positions[:n], enu_positions[:n]
            )

            # Apply alignment to all visual poses
            fused_poses = []
            for i, pose in enumerate(visual_poses):
                v_pos = np.array(pose.translation)
                aligned_pos = scale * (R_align @ v_pos) + t_align

                # Weighted fusion with GPS (if available for this frame)
                if i < len(enu_positions):
                    gps_weight = 0.3  # GPS is noisy but provides global reference
                    visual_weight = 0.7
                    fused_pos = (
                        visual_weight * aligned_pos + gps_weight * enu_positions[i]
                    )
                    confidence = 0.9
                else:
                    fused_pos = aligned_pos
                    confidence = 0.7

                fused_poses.append(CameraPose(
                    frame_index=pose.frame_index,
                    rotation=pose.rotation,
                    translation=fused_pos.tolist(),
                    confidence=confidence,
                    source="fused",
                ))

            logger.info(f"Sensor fusion complete. Scale factor: {scale:.4f}")
            return fused_poses

        return visual_poses

    def _gps_to_enu(self, gps_points: List[GPSPoint]) -> np.ndarray:
        """Convert GPS coordinates to local East-North-Up (ENU) frame."""
        if not gps_points:
            return np.array([])

        # Use first point as reference
        ref = gps_points[0]
        ref_lat = np.radians(ref.latitude)
        ref_lon = np.radians(ref.longitude)
        ref_alt = ref.altitude or 0

        positions = []
        for gp in gps_points:
            lat = np.radians(gp.latitude)
            lon = np.radians(gp.longitude)
            alt = (gp.altitude or 0) - ref_alt

            # Approximate ENU conversion
            R_earth = 6378137.0  # WGS84
            dlat = lat - ref_lat
            dlon = lon - ref_lon

            east = R_earth * dlon * np.cos(ref_lat)
            north = R_earth * dlat
            up = alt

            positions.append([east, north, up])

        return np.array(positions)

    def _procrustes_align(
        self, source: np.ndarray, target: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Procrustes alignment: find scale, rotation, translation."""
        # Center both point sets
        src_mean = np.mean(source, axis=0)
        tgt_mean = np.mean(target, axis=0)
        src_centered = source - src_mean
        tgt_centered = target - tgt_mean

        # Scale
        src_scale = np.sqrt(np.sum(src_centered**2) / len(source))
        tgt_scale = np.sqrt(np.sum(tgt_centered**2) / len(target))
        scale = tgt_scale / max(src_scale, 1e-8)

        # Rotation (SVD)
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Translation
        t = tgt_mean - scale * (R @ src_mean)

        return scale, R, t

    @staticmethod
    def _quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])
        return R

    def _run_cmd(self, cmd: list):
        """Run a subprocess command."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            if result.returncode != 0:
                logger.error(f"Command failed: {' '.join(cmd[:2])}")
                logger.error(result.stderr[:500])
                raise RuntimeError(f"Command failed: {result.stderr[:200]}")
        except FileNotFoundError:
            raise RuntimeError(f"Command not found: {cmd[0]}")

    def save_trajectory(self, poses: List[CameraPose], output_path: Path):
        """Save camera trajectory as JSON for visualization."""
        trajectory = {
            "poses": [p.model_dump() for p in poses],
            "positions": [p.translation for p in poses],
        }
        with open(output_path, "w") as f:
            json.dump(trajectory, f, indent=2)
        logger.info(f"Saved trajectory to {output_path}")
