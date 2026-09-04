from loguru import logger
import sys
from reconstruction.pipeline import ReconstructionPipeline

if __name__ == "__main__":
    logger.info("Setting up pipeline test...")
    pipeline = ReconstructionPipeline(
        upload_dir="data/uploads/82e220b8",
        frames_dir="data/frames/82e220b8",
        output_dir="data/outputs/82e220b8"
    )
    
    # Run the pipeline
    success = pipeline.run("mixkit-forest-treetops-532-hd-ready.mp4")
    
    if success:
        logger.info("Test finished successfully!")
        sys.exit(0)
    else:
        logger.error("Test failed!")
        sys.exit(1)
