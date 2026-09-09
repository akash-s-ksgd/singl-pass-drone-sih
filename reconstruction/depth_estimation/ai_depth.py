import torch
import cv2
import numpy as np
import gc
from pathlib import Path
from PIL import Image
from transformers import pipeline
from loguru import logger

class VibeDepthEngine:
    """
    Neural Depth generation using Depth Anything V2.
    Engineered with aggressive memory management for local laptop inference.
    """
    def __init__(self, output_dir: str | Path = "data/outputs/depth_maps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing Neural Core: Depth Anything V2 (Small)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Compute Hardware Locked: {self.device.upper()}")
        
        # 1. FORCE RAM FLUSH: Clear any lingering C++ memory before loading the AI
        logger.info("🧹 Flushing system memory before neural network injection...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. LOAD AI: Bypass the Windows 'mmap' bug by disabling low_cpu_mem_usage
        logger.info("🧠 Loading weights into memory...")
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if self.device == "cuda" else -1,
            model_kwargs={"low_cpu_mem_usage": False} # CRITICAL FIX FOR OS ERROR 1455
        )
        logger.info("⚡ Neural Core Online.")

    def process_frame(self, image_path: str | Path) -> bool:
        image_path = Path(image_path)
        if not image_path.exists():
            return False

        # Load image via PIL (required by transformers pipeline)
        image = Image.open(image_path).convert("RGB")
        
        # Run the neural network
        result = self.pipe(image)
        
        # Extract the depth tensor and normalize for visualization
        depth_map = result["depth"]
        depth_array = np.array(depth_map)
        
        # Normalize to 0-255 for standard image saving
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        depth_normalized = (255 * (depth_array - depth_min) / (depth_max - depth_min)).astype("uint8")
        
        # Apply INFERNO colormap for human visualization, but remember our projector uses raw geometry!
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        out_path = self.output_dir / f"depth_{image_path.name}"
        cv2.imwrite(str(out_path), depth_colored)
        logger.debug(f"Saved neural depth map -> {out_path.name}")
        
        return True

    def process_directory(self, input_dir: str | Path) -> None:
        input_dir = Path(input_dir)
        image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
        
        logger.info(f"Injecting {len(image_files)} frames into the Depth Engine...")
        for img_path in image_files:
            self.process_frame(img_path)
            
        # Flush RAM again after processing is complete
        gc.collect()