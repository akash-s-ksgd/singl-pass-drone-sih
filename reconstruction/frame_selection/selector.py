import cv2
import os
from pathlib import Path
from typing import List
from loguru import logger

class VideoPreprocessor:
    """
    Ingests a video file, extracts frames at a target FPS, applies a Laplacian variance
    quality gate to discard blurry frames, and aggressively downscales accepted frames.
    """

    def __init__(
        self,
        upload_dir: str | Path = "data/uploads",
        frames_dir: str | Path = "data/frames",
        target_fps: float = 4.0,
        blur_threshold: float = 100.0,
        max_dim: int = 1024,
    ) -> None:
        """
        Initializes the VideoPreprocessor.
        """
        self.upload_dir = Path(upload_dir)
        self.frames_dir = Path(frames_dir)
        self.target_fps = target_fps
        self.blur_threshold = blur_threshold
        self.max_dim = max_dim

        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def process(self, video_filename: str) -> List[Path]:
        """
        Process the video and return a list of paths to the saved curated frames.
        
        Args:
            video_filename: The name of the video file in the upload directory.
            
        Returns:
            A list of Paths pointing to the successfully extracted and downscaled frames.
        """
        video_path = self.upload_dir / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Starting VideoPreprocessor for {video_filename}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            logger.warning("Could not detect video FPS, falling back to 30.0 FPS")
            video_fps = 30.0
            
        frame_interval = max(1, int(round(video_fps / self.target_fps)))
        logger.info(f"Detected FPS: {video_fps:.2f} | Extracting 1 frame every {frame_interval} frames (~{self.target_fps} FPS)")

        saved_frames: List[Path] = []
        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # 1. Quality Gate: Laplacian Variance
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()

                if variance < self.blur_threshold:
                    logger.debug(f"Frame {frame_idx} rejected: Blurry (Variance {variance:.2f} < {self.blur_threshold})")
                else:
                    # 2. Memory Optimization: Aggressive Downscaling (Maintain Aspect Ratio)
                    h, w = frame.shape[:2]
                    if max(h, w) > self.max_dim:
                        scale = float(self.max_dim) / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Save the curated frame
                    frame_name = f"frame_{saved_idx:05d}.jpg"
                    out_path = self.frames_dir / frame_name
                    
                    cv2.imwrite(str(out_path), frame)
                    saved_frames.append(out_path)
                    saved_idx += 1

            frame_idx += 1

        cap.release()
        logger.info(f"VideoPreprocessor completed. Extracted {len(saved_frames)} high-quality frames to {self.frames_dir}")
        return saved_frames
