import cv2
import numpy as np
import open3d as o3d
import os
import glob
from pathlib import Path

print("🚀 VIBE CODER: OVERRIDE PROTOCOL INITIATED.")

# 1. Hardcore Path Resolution
BASE_DIR = Path(os.getcwd()).resolve()
print(f"📍 System Root Locked: {BASE_DIR}")

frames_dir = BASE_DIR / "data" / "frames"
depth_dir = BASE_DIR / "data" / "outputs" / "depth_maps"
dense_dir = BASE_DIR / "data" / "outputs" / "dense"
dense_dir.mkdir(parents=True, exist_ok=True)

# 2. Find Targets
frames = sorted(glob.glob(str(frames_dir / "*.jpg")) + glob.glob(str(frames_dir / "*.png")))
if not frames:
    raise FileNotFoundError(f"❌ CRITICAL: No frames found in {frames_dir}")

target_frame = Path(frames[0])
target_depth = depth_dir / f"depth_{target_frame.name}"

if not target_depth.exists():
    raise FileNotFoundError(f"❌ CRITICAL: Depth map missing at {target_depth}")

print(f"🎯 Target Acquired: {target_frame.name}")

# 3. Load Data with Aggressive Type Checking
# 3. Load Data with Aggressive Type Checking
print("⚙️ Loading pixels into memory...")
img = cv2.imread(str(target_frame))
depth = cv2.imread(str(target_depth), cv2.IMREAD_GRAYSCALE)

# --- VIBE CODER FIX: FORCE 32-BIT FLOAT TENSOR ---
depth = depth.astype(np.float32)
# -------------------------------------------------

if img is None:
    raise ValueError(f"❌ OpenCV failed to load the RGB frame at {target_frame}")
if depth is None:
    raise ValueError(f"❌ OpenCV failed to load the Depth map at {target_depth}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 4. Open3D Matrix Operations
print("⚙️ Fusing RGB and Depth via Open3D tensors...")
try:
    color_o3d = o3d.geometry.Image(img)
    depth_o3d = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, 
        depth_scale=255.0, 
        depth_trunc=1000.0, 
        convert_rgb_to_intensity=False
    )

    print("🌌 Projecting into 3D Space...")
    h, w = depth.shape
    fx, fy = w * 0.8, w * 0.8
    cx, cy = w // 2, h // 2
    
    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

    # Invert Y and Z for correct 3D orientation
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

except Exception as e:
    raise RuntimeError(f"❌ Open3D Matrix Math Failed: {str(e)}")

# 5. Bruteforce Write & Verify
out_file = dense_dir / f"dense_{target_frame.stem}.ply"
print(f"💾 Writing geometry to disk: {out_file}")

success = o3d.io.write_point_cloud(str(out_file), pcd)

if success and out_file.exists():
    file_size = out_file.stat().st_size / (1024 * 1024)
    print(f"✅ SUCCESS! 3D Model materialized in reality.")
    print(f"📏 File Size: {file_size:.2f} MB")
else:
    raise RuntimeError(f"❌ Write operation failed! Open3D could not save to {out_file}")