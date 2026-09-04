"""
Evaluation Metrics — Measures reconstruction quality across
geometry, pose accuracy, depth accuracy, and visual quality.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, List
from loguru import logger


class ReconstructionEvaluator:
    """Evaluate 3D reconstruction quality."""

    def evaluate_all(
        self,
        points: np.ndarray,
        confidences: np.ndarray,
        camera_poses_count: int,
        total_frames: int,
        selected_frames: int,
        mesh_vertices: int = 0,
        mesh_faces: int = 0,
        gt_points: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all available metrics."""
        metrics = {}

        # Registration rate
        metrics["registration_rate"] = round(camera_poses_count / max(selected_frames, 1), 3)

        # Point cloud statistics
        if len(points) > 0:
            metrics["total_points"] = len(points)
            metrics["point_density"] = self._estimate_density(points)

            # Bounding box volume
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            bbox_size = bbox_max - bbox_min
            metrics["bbox_x_m"] = round(float(bbox_size[0]), 2)
            metrics["bbox_y_m"] = round(float(bbox_size[1]), 2)
            metrics["bbox_z_m"] = round(float(bbox_size[2]), 2)
            metrics["bbox_volume_m3"] = round(float(np.prod(bbox_size)), 2)

        # Confidence statistics
        if len(confidences) > 0:
            metrics["mean_confidence"] = round(float(np.mean(confidences)), 3)
            metrics["median_confidence"] = round(float(np.median(confidences)), 3)
            metrics["high_confidence_pct"] = round(
                100 * float(np.mean(confidences >= 0.7)), 1
            )
            metrics["low_confidence_pct"] = round(
                100 * float(np.mean(confidences < 0.4)), 1
            )

        # Frame selection efficiency
        metrics["frame_reduction_ratio"] = round(
            1 - selected_frames / max(total_frames, 1), 3
        )

        # Mesh quality
        if mesh_vertices > 0:
            metrics["mesh_vertices"] = mesh_vertices
            metrics["mesh_faces"] = mesh_faces

        # Ground truth comparison (if available)
        if gt_points is not None and len(gt_points) > 0 and len(points) > 0:
            chamfer = self._chamfer_distance(points, gt_points)
            metrics["chamfer_distance"] = round(float(chamfer), 4)

        return metrics

    def _estimate_density(self, points: np.ndarray) -> float:
        """Estimate point density (points per cubic meter)."""
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        volume = max(np.prod(bbox_max - bbox_min), 1e-6)
        return round(len(points) / volume, 1)

    def _chamfer_distance(
        self, pred: np.ndarray, gt: np.ndarray, sample_size: int = 10000
    ) -> float:
        """Compute Chamfer distance between two point clouds."""
        from scipy.spatial import cKDTree

        # Subsample for speed
        if len(pred) > sample_size:
            pred = pred[np.random.choice(len(pred), sample_size, replace=False)]
        if len(gt) > sample_size:
            gt = gt[np.random.choice(len(gt), sample_size, replace=False)]

        tree_pred = cKDTree(pred)
        tree_gt = cKDTree(gt)

        dist_pred_to_gt, _ = tree_gt.query(pred)
        dist_gt_to_pred, _ = tree_pred.query(gt)

        chamfer = float(np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred))
        return chamfer

    def generate_report_text(self, metrics: Dict[str, float]) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 60,
            "  RECONSTRUCTION QUALITY REPORT",
            "=" * 60,
            "",
        ]

        sections = {
            "Coverage": ["registration_rate", "frame_reduction_ratio"],
            "Point Cloud": ["total_points", "point_density", "bbox_volume_m3"],
            "Confidence": ["mean_confidence", "high_confidence_pct", "low_confidence_pct"],
            "Mesh": ["mesh_vertices", "mesh_faces"],
            "Accuracy": ["chamfer_distance"],
        }

        for section, keys in sections.items():
            relevant = {k: v for k, v in metrics.items() if k in keys}
            if relevant:
                lines.append(f"  {section}")
                lines.append("  " + "-" * 40)
                for k, v in relevant.items():
                    label = k.replace("_", " ").title()
                    lines.append(f"    {label}: {v}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
