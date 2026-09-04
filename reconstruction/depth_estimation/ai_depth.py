import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import pipeline
from loguru import logger

class VibeDepthEngine:
    """
    Neural Depth generation using Depth Anything V2.
    Designed for high-aesthetic, metrically anchored 3D hallucination.
    """
    def __init__(self, output_dir: str | Path = "data/outputs/depth_maps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing Neural Core: Depth Anything V2 (Small)...")
        # Automatically select GPU if available, else fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Compute Hardware Locked: {self.device.upper()}")
        
        # Load the state-of-the-art monocular depth model
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if self.device == "cuda" else -1
        )
        logger.info("Neural Core Online.")

    def process_frame(self, image_path: str | Path) -> bool:
        """
        Ingests a single 2D frame and hallucinates a relative 3D depth map.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return False

        logger.debug(f"Processing depth for: {image_path.name}")
        
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
        
        # Apply a god-tier colormap (INFERNO) for high-end visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        # Save the output
        out_path = self.output_dir / f"depth_{image_path.name}"
        cv2.imwrite(str(out_path), depth_colored)
        logger.info(f"Saved neural depth map -> {out_path.name}")
        
        return True

    def process_directory(self, input_dir: str | Path) -> None:
        """
        Batch processes all frames in a directory.
        """
        input_dir = Path(input_dir)
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        
        logger.info(f"Injecting {len(image_files)} frames into the Depth Engine...")
        for img_path in image_files:
            self.process_frame(img_path)
        
if __name__ == "__main__":
    print("🚀 VIBE CODER: IGNITING NEURAL DEPTH ENGINE...")
    
    # Initialize the engine
    depth_engine = VibeDepthEngine()
    
    # Point it to the beautiful frames we curated in the previous step
    frames_directory = "data/frames"
    
    # Run the batch process
    depth_engine.process_directory(frames_directory)
    print("✨ Depth hallucination complete. Check data/outputs/depth_maps/")