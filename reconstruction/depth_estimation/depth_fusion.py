"""
Hybrid Depth Engine — Fuses multi-view stereo depth, AI monocular depth,
and sensor-derived scale constraints into a unified depth representation.

This module implements the core depth fusion innovation:
- MVS depth: from COLMAP or custom stereo matching (high confidence where available)
- AI depth: from Depth Anything V2 / Metric3D (fills gaps, lower confidence)
- Sensor scale: GPS displacement + altitude anchor the metric scale

The output is a per-pixel depth map with confidence scores.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger


class MonocularDepthEstimator:
    """
    AI-based monocular depth estimation using Depth Anything V2
    or MiDaS as fallback. Produces relative depth maps.
    """

    def __init__(self, model_type: str = "depth_anything_v2", device: str = "cuda"):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.transform = None

    def load_model(self):
        """Lazy-load the depth model."""
        if self.model is not None:
            return

        try:
            import torch
            if self.model_type == "depth_anything_v2":
                self._load_depth_anything_v2()
            else:
                self._load_midas()
        except ImportError as e:
            logger.warning(f"Could not load AI depth model: {e}")
            logger.info("Using OpenCV stereo as fallback")
            self.model_type = "fallback"

    def _load_depth_anything_v2(self):
        """Load Depth Anything V2 via transformers."""
        try:
            import torch
            from transformers import pipeline

            self.model = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
            )
            logger.info("Loaded Depth Anything V2 (Small)")
        except Exception as e:
            logger.warning(f"Depth Anything V2 failed: {e}, falling back to MiDaS")
            self._load_midas()

    def _load_midas(self):
        """Load MiDaS depth model."""
        try:
            import torch
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.eval()
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            self.model_type = "midas"
            logger.info("Loaded MiDaS Small")
        except Exception as e:
            logger.warning(f"MiDaS failed: {e}")
            self.model_type = "fallback"

    def estimate_depth(self, image_path: str) -> Tuple[np.ndarray, float]:
        """
        Estimate depth for a single image.

        Returns:
            depth_map: HxW numpy array (relative depth, higher = farther)
            confidence: overall confidence score (0-1)
        """
        self.load_model()

        if self.model_type == "fallback":
            return self._fallback_depth(image_path)

        if self.model_type == "depth_anything_v2":
            return self._depth_anything_inference(image_path)
        elif self.model_type == "midas":
            return self._midas_inference(image_path)
        else:
            return self._fallback_depth(image_path)

    def _depth_anything_inference(self, image_path: str) -> Tuple[np.ndarray, float]:
        """Run Depth Anything V2 inference."""
        from PIL import Image
        img = Image.open(image_path)
        result = self.model(img)
        depth = np.array(result["depth"])

        # Normalize to 0-1 range
        depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-8)
        confidence = 0.6  # AI depth gets moderate confidence
        return depth.astype(np.float32), confidence

    def _midas_inference(self, image_path: str) -> Tuple[np.ndarray, float]:
        """Run MiDaS inference."""
        import torch
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb)

        if self.device == "cuda" and torch.cuda.is_available():
            input_batch = input_batch.cuda()

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-8)
        return depth.astype(np.float32), 0.5

    def _fallback_depth(self, image_path: str) -> Tuple[np.ndarray, float]:
        """Simple gradient-based pseudo-depth (for testing without GPU)."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((480, 640), dtype=np.float32), 0.0

        h, w = img.shape
        # Simple assumption: lower in image = closer (for aerial views this is rough)
        y_gradient = np.linspace(0.3, 1.0, h).reshape(-1, 1)
        y_gradient = np.tile(y_gradient, (1, w))

        # Add texture-based depth cue (edges suggest closer objects)
        edges = cv2.Canny(img, 50, 150).astype(np.float32) / 255.0
        edges_blur = cv2.GaussianBlur(edges, (21, 21), 0)

        depth = y_gradient * 0.7 + (1 - edges_blur) * 0.3
        depth = depth.astype(np.float32)
        return depth, 0.2  # Very low confidence for fallback

    def estimate_batch(
        self, image_paths: List[str]
    ) -> List[Tuple[np.ndarray, float]]:
        """Estimate depth for multiple images."""
        results = []
        for i, path in enumerate(image_paths):
            depth, conf = self.estimate_depth(path)
            results.append((depth, conf))
            if (i + 1) % 10 == 0:
                logger.info(f"Depth estimated for {i + 1}/{len(image_paths)} frames")
        return results


class DepthFusionEngine:
    """
    Fuses depth from multiple sources with confidence-aware weighting.

    Sources:
    - MVS depth (from COLMAP or stereo matching): highest confidence
    - AI monocular depth: moderate confidence, fills gaps
    - Sensor constraints: provides metric scale anchor

    Innovation: Each pixel gets a fused depth AND a confidence score
    indicating whether it came from geometric observation or AI prediction.
    """

    def __init__(self, sensor_altitude: Optional[float] = None):
        self.sensor_altitude = sensor_altitude

    def fuse(
        self,
        mvs_depth: Optional[np.ndarray],
        ai_depth: np.ndarray,
        ai_confidence: float,
        mvs_confidence_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse MVS and AI depth maps.

        Args:
            mvs_depth: Multi-view stereo depth (may have holes/zeros)
            ai_depth: AI monocular depth (dense but scale-ambiguous)
            ai_confidence: Overall confidence of AI depth (0-1)
            mvs_confidence_map: Per-pixel MVS confidence

        Returns:
            fused_depth: Combined depth map
            confidence_map: Per-pixel confidence (0-1)
        """
        h, w = ai_depth.shape[:2]

        if mvs_depth is None:
            # No MVS depth — use AI depth with low-medium confidence
            confidence = np.full((h, w), ai_confidence * 0.7, dtype=np.float32)
            fused = ai_depth.copy()

            # Apply sensor altitude scaling if available
            if self.sensor_altitude and self.sensor_altitude > 0:
                scale = self.sensor_altitude / max(np.median(ai_depth[ai_depth > 0]), 1e-6)
                fused = fused * scale
                confidence *= 1.2  # Slightly boost confidence with sensor scale
                confidence = np.clip(confidence, 0, 1)

            return fused, confidence

        # Ensure same resolution
        if mvs_depth.shape != ai_depth.shape:
            ai_depth = cv2.resize(ai_depth, (mvs_depth.shape[1], mvs_depth.shape[0]))
            h, w = mvs_depth.shape[:2]

        # Create validity masks
        mvs_valid = (mvs_depth > 0) & np.isfinite(mvs_depth)

        # Align AI depth to MVS scale where MVS is valid
        if np.sum(mvs_valid) > 100:
            ai_depth_aligned = self._align_depth_scale(
                ai_depth, mvs_depth, mvs_valid
            )
        else:
            ai_depth_aligned = ai_depth

        # Create confidence maps
        if mvs_confidence_map is None:
            mvs_conf = np.where(mvs_valid, 0.9, 0.0).astype(np.float32)
        else:
            mvs_conf = mvs_confidence_map.astype(np.float32)

        ai_conf = np.full((h, w), ai_confidence * 0.5, dtype=np.float32)

        # Weighted fusion
        total_conf = mvs_conf + ai_conf + 1e-8
        fused = (
            mvs_conf * np.where(mvs_valid, mvs_depth, 0)
            + ai_conf * ai_depth_aligned
        ) / total_conf

        # Confidence map: maximum of the contributing sources
        confidence = np.maximum(mvs_conf, ai_conf)

        # Where MVS is valid, boost confidence
        confidence[mvs_valid] = np.clip(
            mvs_conf[mvs_valid] + 0.1, 0, 1
        )

        # Where only AI depth exists, mark lower confidence
        ai_only = ~mvs_valid
        confidence[ai_only] = np.clip(
            ai_conf[ai_only] * 0.8, 0, 1
        )

        return fused.astype(np.float32), confidence.astype(np.float32)

    def _align_depth_scale(
        self,
        ai_depth: np.ndarray,
        mvs_depth: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Align AI depth to MVS depth scale using least squares."""
        ai_vals = ai_depth[valid_mask].flatten()
        mvs_vals = mvs_depth[valid_mask].flatten()

        # Robust scale+shift alignment: mvs = scale * ai + shift
        if len(ai_vals) > 10:
            A = np.vstack([ai_vals, np.ones_like(ai_vals)]).T
            result = np.linalg.lstsq(A, mvs_vals, rcond=None)
            scale, shift = result[0]

            if scale > 0:
                aligned = ai_depth * scale + shift
                return np.clip(aligned, 0, None)

        return ai_depth

    def classify_confidence(
        self, confidence_map: np.ndarray
    ) -> dict:
        """Classify regions by confidence level."""
        total = confidence_map.size
        high = np.sum(confidence_map >= 0.7)
        medium = np.sum((confidence_map >= 0.4) & (confidence_map < 0.7))
        low = np.sum(confidence_map < 0.4)

        return {
            "high": {"count": int(high), "percentage": round(100 * high / total, 1)},
            "medium": {"count": int(medium), "percentage": round(100 * medium / total, 1)},
            "low": {"count": int(low), "percentage": round(100 * low / total, 1)},
        }
