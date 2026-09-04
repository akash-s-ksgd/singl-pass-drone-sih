"""
Frame Quality Scorer — Computes quality metrics for each extracted frame.
Scores: sharpness, feature count, brightness, contrast, blur, exposure.
Produces a composite score used for intelligent frame selection.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import List
from loguru import logger

from backend.app.models.schemas import FrameQuality, GPSPoint


class FrameQualityScorer:
    """Analyze and score frame quality for reconstruction suitability."""

    def __init__(self, feature_detector: str = "orb"):
        self.feature_detector = feature_detector
        if feature_detector == "orb":
            self.detector = cv2.ORB_create(nfeatures=2000)
        elif feature_detector == "sift":
            self.detector = cv2.SIFT_create(nfeatures=2000)
        else:
            self.detector = cv2.ORB_create(nfeatures=2000)

    def score_frame(
        self,
        image_path: str,
        frame_index: int,
        timestamp: float,
        gps: GPSPoint | None = None,
    ) -> FrameQuality:
        """Compute all quality metrics for a single frame."""
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Cannot read frame: {image_path}")
            return FrameQuality(frame_index=frame_index, timestamp=timestamp)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # 2. Feature count
        keypoints = self.detector.detect(gray, None)
        feature_count = len(keypoints)

        # 3. Brightness (mean intensity)
        brightness = float(np.mean(gray))

        # 4. Contrast (std deviation of intensity)
        contrast = float(np.std(gray))

        # 5. Blur score (inverse of sharpness, normalized)
        # High blur = low score
        blur_score = min(100.0, sharpness / 10.0)

        # 6. Exposure quality (penalize over/under-exposed)
        # Ideal brightness ~120-140 for outdoor scenes
        exposure_quality = max(0, 100 - abs(brightness - 130) * 1.5)

        # 7. Composite score (weighted combination)
        composite = (
            0.30 * min(100, sharpness / 5.0) +      # Sharpness (30%)
            0.25 * min(100, feature_count / 10.0) +  # Features (25%)
            0.15 * exposure_quality +                  # Exposure (15%)
            0.15 * min(100, contrast) +                # Contrast (15%)
            0.15 * blur_score                          # Blur (15%)
        )

        return FrameQuality(
            frame_index=frame_index,
            timestamp=timestamp,
            sharpness_score=round(sharpness, 2),
            feature_count=feature_count,
            brightness_score=round(brightness, 2),
            contrast_score=round(contrast, 2),
            blur_score=round(blur_score, 2),
            exposure_quality=round(exposure_quality, 2),
            composite_score=round(composite, 2),
            gps=gps,
        )

    def score_batch(
        self,
        frames: List[dict],
        gps_points: List[GPSPoint] | None = None,
    ) -> List[FrameQuality]:
        """Score all extracted frames."""
        scores = []
        for i, frame_info in enumerate(frames):
            gps = None
            if gps_points and i < len(gps_points):
                gps = gps_points[i]

            quality = self.score_frame(
                image_path=frame_info["path"],
                frame_index=frame_info["index"],
                timestamp=frame_info["timestamp"],
                gps=gps,
            )
            scores.append(quality)

            if (i + 1) % 50 == 0:
                logger.info(f"Scored {i + 1}/{len(frames)} frames")

        logger.info(
            f"Quality scoring complete. "
            f"Mean composite: {np.mean([s.composite_score for s in scores]):.1f}"
        )
        return scores
