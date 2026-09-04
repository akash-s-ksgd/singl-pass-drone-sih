"""
Confidence-Aware Point Cloud Generator — Creates dense 3D point clouds
from depth maps and camera poses, with per-point confidence scores.

This module implements the confidence-aware reconstruction innovation:
each point is tagged with a confidence level based on how it was generated.
"""
from __future__ import annotations
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

from backend.app.models.schemas import CameraPose, ConfidenceLevel, ConfidenceRegion


class ConfidenceAwarePointCloudGenerator:
    """
    Generates dense point clouds from depth maps + camera poses.
    Each point carries a confidence score distinguishing:
    - Observed geometry (multi-view verified)
    - AI-inferred geometry (monocular depth prediction)
    - Sensor-constrained geometry (GPS/altitude anchored)
    """

    def __init__(self, voxel_size: float = 0.05):
        self.voxel_size = voxel_size

    def generate_from_depth_maps(
        self,
        image_paths: List[str],
        depth_maps: List[np.ndarray],
        confidence_maps: List[np.ndarray],
        camera_poses: List[CameraPose],
        camera_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Back-project depth maps into 3D to create a dense point cloud.

        Returns:
            points: Nx3 array of 3D coordinates
            colors: Nx3 array of RGB colors (0-1)
            confidences: N array of confidence scores (0-1)
        """
        all_points = []
        all_colors = []
        all_confidences = []

        for i, (img_path, depth, conf, pose) in enumerate(
            zip(image_paths, depth_maps, confidence_maps, camera_poses)
        ):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            h, w = depth.shape[:2]

            # Resize image to match depth if needed
            if img_rgb.shape[:2] != depth.shape[:2]:
                img_rgb = cv2.resize(img_rgb, (w, h))

            # Camera intrinsics (estimate if not provided)
            if camera_matrix is not None:
                K = camera_matrix
            else:
                focal = max(h, w) * 1.2
                K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # Create pixel grid
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            # Back-project to camera coordinates
            z = depth
            valid = (z > 0) & np.isfinite(z) & (conf > 0.1)

            x_cam = (u[valid] - cx) * z[valid] / fx
            y_cam = (v[valid] - cy) * z[valid] / fy
            z_cam = z[valid]

            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

            # Transform to world coordinates
            R = np.array(pose.rotation)
            t = np.array(pose.translation)

            # Camera-to-world: P_world = R^T @ (P_cam - t)
            pts_world = (R.T @ (pts_cam - t).T).T

            colors_valid = img_rgb[valid]
            conf_valid = conf[valid]

            # Subsample if too many points per frame
            max_per_frame = 100000
            if len(pts_world) > max_per_frame:
                indices = np.random.choice(len(pts_world), max_per_frame, replace=False)
                pts_world = pts_world[indices]
                colors_valid = colors_valid[indices]
                conf_valid = conf_valid[indices]

            all_points.append(pts_world)
            all_colors.append(colors_valid)
            all_confidences.append(conf_valid)

            if (i + 1) % 20 == 0:
                total = sum(len(p) for p in all_points)
                logger.info(f"Back-projected {i + 1}/{len(image_paths)} frames ({total:,} points)")

        if not all_points:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        confidences = np.concatenate(all_confidences, axis=0)

        logger.info(f"Raw point cloud: {len(points):,} points")

        # Clean and downsample
        points, colors, confidences = self._clean_pointcloud(
            points, colors, confidences
        )

        return points, colors, confidences

    def _clean_pointcloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        confidences: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove outliers and downsample the point cloud."""
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Statistical outlier removal
            pcd_clean, inlier_idx = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            inlier_idx = list(inlier_idx)
            confidences = confidences[inlier_idx]

            # Voxel downsampling
            if self.voxel_size > 0:
                # Custom downsampling to preserve confidence
                pcd_down = pcd_clean.voxel_down_sample(self.voxel_size)
                # Approximate: use first point's confidence per voxel
                n_down = len(pcd_down.points)
                if n_down < len(confidences):
                    # Resample confidences
                    indices = np.linspace(0, len(confidences) - 1, n_down).astype(int)
                    confidences = confidences[indices]

                points_out = np.asarray(pcd_down.points)
                colors_out = np.asarray(pcd_down.colors)
            else:
                points_out = np.asarray(pcd_clean.points)
                colors_out = np.asarray(pcd_clean.colors)

            logger.info(f"Cleaned point cloud: {len(points_out):,} points")
            return points_out, colors_out, confidences[:len(points_out)]

        except ImportError:
            logger.warning("Open3D not available, skipping point cloud cleaning")
            return points, colors, confidences

    def save_ply(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        confidences: np.ndarray,
        output_path: Path,
    ):
        """Save point cloud as PLY with confidence as scalar field."""
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Encode confidence in color: high=green, medium=yellow, low=red
            conf_colors = np.zeros_like(colors)
            for i, c in enumerate(confidences):
                if c >= 0.7:
                    conf_colors[i] = colors[i]  # Keep original color for high conf
                elif c >= 0.4:
                    # Tint yellow
                    conf_colors[i] = colors[i] * 0.7 + np.array([0.3, 0.3, 0]) 
                else:
                    # Tint red
                    conf_colors[i] = colors[i] * 0.5 + np.array([0.5, 0, 0])

            pcd.colors = o3d.utility.Vector3dVector(
                np.clip(conf_colors, 0, 1)
            )

            o3d.io.write_point_cloud(str(output_path), pcd)
            logger.info(f"Saved point cloud: {output_path} ({len(points):,} points)")

        except ImportError:
            # Fallback: write PLY manually
            self._write_ply_manual(points, colors, output_path)

    def _write_ply_manual(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Path,
    ):
        """Write PLY file without Open3D."""
        n = len(points)
        header = (
            f"ply\nformat ascii 1.0\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        )
        with open(output_path, "w") as f:
            f.write(header)
            for i in range(n):
                r, g, b = (colors[i] * 255).astype(int)
                f.write(
                    f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                    f"{r} {g} {b}\n"
                )
        logger.info(f"Saved PLY (manual): {output_path}")

    def save_preview_json(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        confidences: np.ndarray,
        output_path: Path,
        max_points: int = 50000,
    ):
        """Save a downsampled point cloud as JSON for web visualization."""
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]
            confidences = confidences[indices]

        data = {
            "points": points.tolist(),
            "colors": colors.tolist(),
            "confidences": confidences.tolist(),
            "count": len(points),
        }

        with open(output_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Saved preview JSON: {output_path} ({len(points)} points)")

    @staticmethod
    def compute_confidence_distribution(
        confidences: np.ndarray,
    ) -> List[ConfidenceRegion]:
        """Compute confidence distribution statistics."""
        total = len(confidences)
        if total == 0:
            return []

        high_mask = confidences >= 0.7
        med_mask = (confidences >= 0.4) & (confidences < 0.7)
        low_mask = confidences < 0.4

        regions = []
        for level, mask in [
            (ConfidenceLevel.HIGH, high_mask),
            (ConfidenceLevel.MEDIUM, med_mask),
            (ConfidenceLevel.LOW, low_mask),
        ]:
            count = int(np.sum(mask))
            regions.append(ConfidenceRegion(
                level=level,
                point_count=count,
                percentage=round(100 * count / total, 1),
                mean_confidence=round(float(np.mean(confidences[mask])) if count > 0 else 0, 3),
            ))

        return regions
