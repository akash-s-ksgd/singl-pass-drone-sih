"""
Intelligent Frame Selector — Selects the optimal subset of frames
for reconstruction based on quality scores, viewpoint diversity,
and GPS displacement. This is a key innovation module.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional
from loguru import logger

from backend.app.models.schemas import FrameQuality


class IntelligentFrameSelector:
    """
    Select the optimal frame subset for reconstruction.

    Innovation: Information-aware selection that maximizes reconstruction
    quality while minimizing redundancy and computation.
    """

    def __init__(
        self,
        min_sharpness: float = 50.0,
        min_feature_count: int = 100,
        min_composite_score: float = 30.0,
        min_parallax_pixels: float = 20.0,
        max_frames: int = 300,
    ):
        self.min_sharpness = min_sharpness
        self.min_feature_count = min_feature_count
        self.min_composite_score = min_composite_score
        self.min_parallax_pixels = min_parallax_pixels
        self.max_frames = max_frames

    def select(
        self,
        frame_qualities: List[FrameQuality],
        frame_paths: List[str],
    ) -> List[FrameQuality]:
        """
        Multi-stage frame selection:
        1. Quality gate — reject blurry/dark/featureless frames
        2. Diversity filter — ensure sufficient viewpoint change
        3. Budget cap — respect max_frames limit
        """
        logger.info(f"Selecting from {len(frame_qualities)} candidate frames...")

        # Stage 1: Quality gate
        quality_passed = self._quality_gate(frame_qualities)
        logger.info(f"  Quality gate: {len(quality_passed)}/{len(frame_qualities)} passed")

        # Stage 2: Diversity filter via visual parallax
        diverse = self._diversity_filter(quality_passed, frame_paths)
        logger.info(f"  Diversity filter: {len(diverse)} frames selected")

        # Stage 3: Budget cap
        if len(diverse) > self.max_frames:
            diverse.sort(key=lambda f: f.composite_score, reverse=True)
            diverse = diverse[:self.max_frames]
            diverse.sort(key=lambda f: f.frame_index)
            logger.info(f"  Budget cap: trimmed to {self.max_frames} frames")

        # Mark selected frames
        for fq in diverse:
            fq.selected = True

        logger.info(
            f"Final selection: {len(diverse)} frames "
            f"(mean score: {np.mean([f.composite_score for f in diverse]):.1f})"
        )
        return diverse

    def _quality_gate(self, frames: List[FrameQuality]) -> List[FrameQuality]:
        """Reject frames below quality thresholds."""
        passed = []
        for fq in frames:
            if (
                fq.sharpness_score >= self.min_sharpness
                and fq.feature_count >= self.min_feature_count
                and fq.composite_score >= self.min_composite_score
            ):
                passed.append(fq)

        # Always keep first and last frame
        if frames and frames[0] not in passed:
            passed.insert(0, frames[0])
        if frames and frames[-1] not in passed:
            passed.append(frames[-1])

        return passed

    def _diversity_filter(
        self,
        frames: List[FrameQuality],
        frame_paths: List[str],
    ) -> List[FrameQuality]:
        """
        Ensure sufficient visual diversity between selected frames.
        Uses optical flow to estimate parallax between consecutive frames.
        """
        if len(frames) <= 2:
            return frames

        selected = [frames[0]]  # Always keep first
        prev_gray = None

        # Build path lookup
        path_map = {}
        for fq in frames:
            if fq.frame_index < len(frame_paths):
                path_map[fq.frame_index] = frame_paths[fq.frame_index]

        for fq in frames:
            path = path_map.get(fq.frame_index)
            if path is None:
                continue

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize for speed
            small = cv2.resize(img, (640, 360))

            if prev_gray is None:
                prev_gray = small
                continue

            # Compute optical flow magnitude as parallax proxy
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, small, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mean_parallax = float(np.mean(magnitude))

            if mean_parallax >= self.min_parallax_pixels:
                selected.append(fq)
                prev_gray = small
            # else: skip this frame (too similar to previous)

        # Always include last frame
        if frames[-1] not in selected:
            selected.append(frames[-1])

        return selected

    def get_selection_summary(
        self,
        all_frames: List[FrameQuality],
        selected: List[FrameQuality],
    ) -> dict:
        """Generate a summary of the selection process."""
        all_scores = [f.composite_score for f in all_frames]
        sel_scores = [f.composite_score for f in selected]

        return {
            "total_extracted": len(all_frames),
            "total_selected": len(selected),
            "reduction_ratio": round(1 - len(selected) / max(len(all_frames), 1), 3),
            "mean_score_all": round(float(np.mean(all_scores)), 2),
            "mean_score_selected": round(float(np.mean(sel_scores)), 2),
            "min_score_selected": round(float(np.min(sel_scores)), 2),
            "max_score_selected": round(float(np.max(sel_scores)), 2),
            "quality_improvement": round(
                float(np.mean(sel_scores) - np.mean(all_scores)), 2
            ),
        }
