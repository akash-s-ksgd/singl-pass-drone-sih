"""
Frame Extractor — Extracts frames from drone video at specified FPS,
with optional adaptive extraction based on motion/GPS displacement.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Callable
from loguru import logger


class FrameExtractor:
    """Extracts frames from a video file."""

    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        target_fps: float = 2.0,
        max_frames: int = 500,
    ):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[dict]:
        """
        Extract frames at the target FPS.

        Returns:
            List of dicts: {index, path, timestamp, original_frame_num}
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / max(video_fps, 1)
        frame_interval = max(1, int(video_fps / self.target_fps))

        logger.info(
            f"Video: {total_frames} frames, {video_fps:.1f} fps, "
            f"{duration:.1f}s — extracting every {frame_interval} frames"
        )

        extracted = []
        frame_num = 0
        extract_idx = 0

        while cap.isOpened() and extract_idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                timestamp = frame_num / max(video_fps, 1)
                filename = f"frame_{extract_idx:05d}.jpg"
                out_path = self.output_dir / filename

                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                extracted.append({
                    "index": extract_idx,
                    "path": str(out_path),
                    "timestamp": timestamp,
                    "original_frame_num": frame_num,
                })
                extract_idx += 1

                if progress_callback and total_frames > 0:
                    progress_callback(
                        frame_num / total_frames,
                        f"Extracted {extract_idx} frames..."
                    )

            frame_num += 1

        cap.release()
        logger.info(f"Extracted {len(extracted)} frames to {self.output_dir}")
        return extracted

    def extract_adaptive(
        self,
        min_displacement: float = 0.05,
        gps_points: Optional[list] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[dict]:
        """
        Adaptive extraction: extract frames when there's sufficient
        visual change (optical flow magnitude) or GPS displacement.
        Falls back to fixed-fps extraction as baseline.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        extracted = []
        prev_gray = None
        frame_num = 0
        extract_idx = 0
        # Ensure minimum extraction rate (at least 1 fps)
        min_interval = max(1, int(video_fps / 4))
        last_extracted = -min_interval

        while cap.isOpened() and extract_idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            should_extract = False

            if prev_gray is None:
                should_extract = True  # Always extract first frame
            elif frame_num - last_extracted >= min_interval:
                # Check optical flow magnitude for scene change
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                mean_motion = np.mean(magnitude)

                if mean_motion > min_displacement * gray.shape[1]:
                    should_extract = True

                # Force extraction if it's been too long
                if frame_num - last_extracted > min_interval * 4:
                    should_extract = True

            if should_extract:
                timestamp = frame_num / max(video_fps, 1)
                filename = f"frame_{extract_idx:05d}.jpg"
                out_path = self.output_dir / filename
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                extracted.append({
                    "index": extract_idx,
                    "path": str(out_path),
                    "timestamp": timestamp,
                    "original_frame_num": frame_num,
                })
                extract_idx += 1
                last_extracted = frame_num
                prev_gray = gray.copy()

                if progress_callback and total_frames > 0:
                    progress_callback(
                        frame_num / total_frames,
                        f"Adaptively extracted {extract_idx} frames..."
                    )
            else:
                prev_gray = gray.copy()

            frame_num += 1

        cap.release()
        logger.info(f"Adaptively extracted {len(extracted)} frames")
        return extracted
