"""
Main Reconstruction Pipeline — Orchestrates all modules from video ingestion
to final 3D model output. This is the central integration point.

Pipeline stages:
1. Frame Extraction → 2. Quality Scoring → 3. Frame Selection →
4. Pose Estimation (+ Sensor Fusion) → 5. Depth Estimation (+ Fusion) →
6. Point Cloud Generation (+ Confidence) → 7. Mesh Generation →
8. Evaluation & Report
"""
from __future__ import annotations
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from loguru import logger

from backend.app.config import settings
from backend.app.models.schemas import (
    ProcessingConfig, FlightMetadata, JobStatus,
    ReconstructionReport, ConfidenceRegion,
)


class ReconstructionPipeline:
    """
    End-to-end reconstruction pipeline orchestrator.
    Integrates all modules into a cohesive processing flow.
    """

    def __init__(
        self,
        video_path: Path,
        project_id: str,
        config: ProcessingConfig,
        flight_metadata: Optional[FlightMetadata] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.video_path = Path(video_path)
        self.project_id = project_id
        self.config = config
        self.flight_metadata = flight_metadata
        self.progress_callback = progress_callback or (lambda *a: None)

        # Directories
        self.frames_dir = settings.frames_dir / project_id
        self.output_dir = settings.output_dir / project_id
        self.workspace_dir = self.output_dir / "workspace"

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> ReconstructionReport:
        """Execute the full reconstruction pipeline."""
        start_time = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"  Starting Reconstruction Pipeline")
        logger.info(f"  Project: {self.project_id}")
        logger.info(f"  Video: {self.video_path.name}")
        logger.info(f"{'='*60}")

        # ── Stage 1: Frame Extraction ────────────────────────────
        self.progress_callback(JobStatus.EXTRACTING_FRAMES, 0.05, "Extracting frames...")
        frames = self._extract_frames()

        # ── Stage 2: Quality Scoring ─────────────────────────────
        self.progress_callback(JobStatus.SELECTING_FRAMES, 0.15, "Scoring frame quality...")
        frame_qualities = self._score_frames(frames)

        # ── Stage 3: Intelligent Frame Selection ─────────────────
        self.progress_callback(JobStatus.SELECTING_FRAMES, 0.20, "Selecting optimal frames...")
        selected, selection_summary = self._select_frames(frame_qualities, frames)

        # ── Stage 4: Pose Estimation + Sensor Fusion ─────────────
        self.progress_callback(JobStatus.ESTIMATING_POSES, 0.30, "Estimating camera poses...")
        poses, pose_stats = self._estimate_poses(selected)

        # ── Stage 5: Depth Estimation + Fusion ───────────────────
        self.progress_callback(JobStatus.ESTIMATING_DEPTH, 0.45, "Estimating depth...")
        depth_maps, confidence_maps = self._estimate_depth(selected, poses)

        # ── Stage 6: Point Cloud Generation ──────────────────────
        self.progress_callback(JobStatus.RECONSTRUCTING, 0.60, "Generating point cloud...")
        points, colors, confidences = self._generate_pointcloud(
            selected, depth_maps, confidence_maps, poses
        )

        # ── Stage 7: Mesh Generation ────────────────────────────
        self.progress_callback(JobStatus.GENERATING_MESH, 0.75, "Generating mesh...")
        mesh_data, mesh_files = self._generate_mesh(points, colors)

        # ── Stage 8: Evaluation ──────────────────────────────────
        self.progress_callback(JobStatus.COMPLETED, 0.90, "Computing metrics...")
        metrics, confidence_dist = self._evaluate(
            points, confidences, len(poses), len(frames), len(selected),
            mesh_data.get("vertex_count", 0), mesh_data.get("face_count", 0)
        )

        # ── Build Report ─────────────────────────────────────────
        elapsed = round(time.time() - start_time, 1)
        report = ReconstructionReport(
            job_id="",  # Set by caller
            project_name="",  # Set by caller
            processing_time_seconds=elapsed,
            frames_extracted=len(frames),
            frames_selected=len(selected),
            camera_poses_recovered=len(poses),
            sparse_points=pose_stats.get("sparse_points", 0),
            dense_points=len(points),
            mesh_vertices=mesh_data.get("vertex_count", 0),
            mesh_faces=mesh_data.get("face_count", 0),
            confidence_distribution=confidence_dist,
            output_files=mesh_files,
            metrics=metrics,
        )

        # Save report
        report_path = self.output_dir / "reconstruction_report.json"
        with open(report_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)

        logger.info(f"\n{'='*60}")
        logger.info(f"  Pipeline Complete in {elapsed}s")
        logger.info(f"  Frames: {len(frames)} extracted → {len(selected)} selected")
        logger.info(f"  Poses: {len(poses)} recovered")
        logger.info(f"  Points: {len(points):,}")
        logger.info(f"  Mesh: {mesh_data.get('vertex_count', 0):,} vertices")
        logger.info(f"{'='*60}")

        return report

    # ─── Stage Implementations ───────────────────────────────────────

    def _extract_frames(self) -> list:
        """Stage 1: Extract frames from video."""
        from reconstruction.frame_selection.extractor import FrameExtractor

        extractor = FrameExtractor(
            video_path=self.video_path,
            output_dir=self.frames_dir,
            target_fps=self.config.fps,
            max_frames=self.config.max_frames,
        )
        return extractor.extract(
            progress_callback=lambda p, m: self.progress_callback(
                JobStatus.EXTRACTING_FRAMES, 0.05 + p * 0.10, m
            )
        )

    def _score_frames(self, frames: list) -> list:
        """Stage 2: Score each frame's quality."""
        from reconstruction.frame_selection.quality_scorer import FrameQualityScorer

        scorer = FrameQualityScorer(feature_detector="orb")
        gps_points = None
        if self.flight_metadata:
            gps_points = self.flight_metadata.gps_points

        return scorer.score_batch(frames, gps_points)

    def _select_frames(self, qualities: list, frames: list) -> tuple:
        """Stage 3: Select optimal frame subset."""
        from reconstruction.frame_selection.selector import IntelligentFrameSelector

        selector = IntelligentFrameSelector(
            min_sharpness=self.config.min_sharpness,
            min_feature_count=self.config.min_feature_count,
            max_frames=self.config.max_frames,
        )

        frame_paths = [f["path"] for f in frames]
        selected = selector.select(qualities, frame_paths)
        summary = selector.get_selection_summary(qualities, selected)

        # Save selection summary
        with open(self.output_dir / "selection_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Selection: {summary}")
        return selected, summary

    def _estimate_poses(self, selected_frames: list) -> tuple:
        """Stage 4: Estimate camera poses with optional sensor fusion."""
        from reconstruction.pose_estimation.sensor_fusion import SensorFusedPoseEstimator

        # Create symlinks/copies of selected frames for COLMAP
        selected_image_dir = self.workspace_dir / "images"
        selected_image_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        for fq in selected_frames:
            src = Path(self.frames_dir / f"frame_{fq.frame_index:05d}.jpg")
            if src.exists():
                dst = selected_image_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

        estimator = SensorFusedPoseEstimator(
            image_dir=selected_image_dir,
            workspace_dir=self.workspace_dir,
            colmap_path=settings.colmap_path,
        )

        poses, stats = estimator.estimate_poses(
            flight_metadata=self.flight_metadata,
            use_sequential_matching=True,
        )

        # Save trajectory
        estimator.save_trajectory(
            poses, self.output_dir / "camera_trajectory.json"
        )

        return poses, stats

    def _estimate_depth(self, selected_frames: list, poses: list) -> tuple:
        """Stage 5: Hybrid depth estimation."""
        from reconstruction.depth_estimation.depth_fusion import (
            MonocularDepthEstimator, DepthFusionEngine,
        )

        if self.config.use_gpu:
            device = self.config.gpu_device if self.config.gpu_device else "cuda"
        else:
            device = "cpu"
        depth_estimator = MonocularDepthEstimator(device=device)
        fusion_engine = DepthFusionEngine(
            sensor_altitude=(
                self.flight_metadata.flight_altitude
                if self.flight_metadata else None
            )
        )

        depth_maps = []
        confidence_maps = []

        for i, fq in enumerate(selected_frames):
            frame_path = str(self.frames_dir / f"frame_{fq.frame_index:05d}.jpg")

            # AI monocular depth
            ai_depth, ai_conf = depth_estimator.estimate_depth(frame_path)

            # Fuse (MVS depth would come from COLMAP dense; None for prototype)
            fused_depth, fused_conf = fusion_engine.fuse(
                mvs_depth=None,  # TODO: integrate COLMAP dense output
                ai_depth=ai_depth,
                ai_confidence=ai_conf,
            )

            depth_maps.append(fused_depth)
            confidence_maps.append(fused_conf)

            if (i + 1) % 20 == 0:
                self.progress_callback(
                    JobStatus.ESTIMATING_DEPTH,
                    0.45 + (i / len(selected_frames)) * 0.15,
                    f"Depth: {i + 1}/{len(selected_frames)} frames"
                )

        return depth_maps, confidence_maps

    def _generate_pointcloud(
        self, selected_frames, depth_maps, confidence_maps, poses
    ) -> tuple:
        """Stage 6: Generate confidence-aware point cloud."""
        from reconstruction.pointcloud.generator import ConfidenceAwarePointCloudGenerator

        generator = ConfidenceAwarePointCloudGenerator(voxel_size=0.05)

        image_paths = [
            str(self.frames_dir / f"frame_{fq.frame_index:05d}.jpg")
            for fq in selected_frames
        ]

        # Match poses to selected frames
        pose_map = {p.frame_index: p for p in poses}
        matched_poses = []
        matched_images = []
        matched_depths = []
        matched_confs = []

        for i, fq in enumerate(selected_frames):
            if fq.frame_index in pose_map:
                matched_poses.append(pose_map[fq.frame_index])
                matched_images.append(image_paths[i])
                matched_depths.append(depth_maps[i])
                matched_confs.append(confidence_maps[i])

        if not matched_poses:
            logger.warning("No matched poses! Using all frames with identity poses.")
            from backend.app.models.schemas import CameraPose
            for i, fq in enumerate(selected_frames):
                matched_poses.append(CameraPose(
                    frame_index=fq.frame_index,
                    rotation=np.eye(3).tolist(),
                    translation=[0, 0, i * 0.5],
                    confidence=0.3,
                    source="fallback",
                ))
                matched_images.append(image_paths[i])
                matched_depths.append(depth_maps[i])
                matched_confs.append(confidence_maps[i])

        # Camera intrinsics from metadata
        camera_matrix = None
        if self.flight_metadata and self.flight_metadata.camera_intrinsics:
            ci = self.flight_metadata.camera_intrinsics
            camera_matrix = np.array([
                [ci.fx, 0, ci.cx],
                [0, ci.fy, ci.cy],
                [0, 0, 1],
            ])

        points, colors, confidences = generator.generate_from_depth_maps(
            image_paths=matched_images,
            depth_maps=matched_depths,
            confidence_maps=matched_confs,
            camera_poses=matched_poses,
            camera_matrix=camera_matrix,
        )

        # Save outputs
        if len(points) > 0:
            generator.save_ply(
                points, colors, confidences,
                self.output_dir / "dense_confidence.ply"
            )
            generator.save_preview_json(
                points, colors, confidences,
                self.output_dir / "pointcloud_preview.json"
            )

        return points, colors, confidences

    def _generate_mesh(self, points, colors) -> tuple:
        """Stage 7: Generate 3D mesh."""
        from reconstruction.mesh.generator import MeshGenerator

        if len(points) < 100:
            logger.warning("Too few points for mesh generation")
            return {"vertex_count": 0, "face_count": 0}, {}

        generator = MeshGenerator(method="poisson")
        mesh_data = generator.generate(points, colors)
        saved_files = generator.save_mesh(mesh_data, self.output_dir, "model")

        return mesh_data, saved_files

    def _evaluate(
        self, points, confidences, pose_count, total_frames, selected_frames,
        mesh_verts, mesh_faces
    ) -> tuple:
        """Stage 8: Evaluate reconstruction quality."""
        from reconstruction.evaluation.metrics import ReconstructionEvaluator
        from reconstruction.pointcloud.generator import ConfidenceAwarePointCloudGenerator

        evaluator = ReconstructionEvaluator()
        metrics = evaluator.evaluate_all(
            points=points,
            confidences=confidences,
            camera_poses_count=pose_count,
            total_frames=total_frames,
            selected_frames=selected_frames,
            mesh_vertices=mesh_verts,
            mesh_faces=mesh_faces,
        )

        # Confidence distribution
        conf_dist = ConfidenceAwarePointCloudGenerator.compute_confidence_distribution(
            confidences
        )

        # Save evaluation report
        report_text = evaluator.generate_report_text(metrics)
        with open(self.output_dir / "evaluation_report.txt", "w") as f:
            f.write(report_text)
        logger.info(f"\n{report_text}")

        return metrics, conf_dist
