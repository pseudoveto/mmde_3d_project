#!/usr/bin/env python3
import os
import glob
import torch
import numpy as np
from PIL import Image

# UniDepth official imports
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PKG_ROOT, "results", "raw_data")
PRED_DIR = os.path.join(PKG_ROOT, "results", "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

def project_bounding_mask(shape, x_dist, y_off, width=640, height=480, fx=320.0, fy=320.0, cx=320.0, cy=240.0):
    """
    Computes a pixel mask covering the obstacle using its 3D bounding envelope,
    strictly discarding floor pixels.
    """
    mask = np.zeros((height, width), dtype=bool)
    
    # Obstacle dimensions
    if shape == "box":
        sx, sy, sz = 0.5, 0.5, 1.0
    else:
        # Cylinder radius 0.4 -> bounding width 0.8
        sx, sy, sz = 0.8, 0.8, 1.0

    # Camera is at (0, 0, 1.0) looking down +X.
    # In camera frame:
    # Z_cam = X_world
    # X_cam = -Y_world (right-handed image plane)
    # Y_cam = -(Z_world - 1.0) (down is +Y_cam)
    z_min = max(0.1, x_dist - sx / 2.0)
    z_max = x_dist + sx / 2.0
    
    # Shrink bounding box by 20% to prevent edge-bleeding and background contamination
    margin = 0.8
    y_world_min = y_off - (sy / 2.0) * margin
    y_world_max = y_off + (sy / 2.0) * margin
    
    # Ground starts at z=0; obstacle is [0, 1.0]m high.
    # Exclude lowest 15cm to prevent floor grazing contact pixels.
    z_world_min = 0.15
    z_world_max = sz * 0.95

    # Project corners
    u_corners = []
    v_corners = []
    for y_w in [y_world_min, y_world_max]:
        for z_w in [z_world_min, z_world_max]:
            # Use front face depth for conservative 2D bbox projection
            xc = -y_w
            yc = -(z_w - 1.0)
            zc = z_min
            u = int(fx * (xc / zc) + cx)
            v = int(fy * (yc / zc) + cy)
            u_corners.append(u)
            v_corners.append(v)

    u0 = max(0, min(u_corners))
    u1 = min(width - 1, max(u_corners))
    v0 = max(0, min(v_corners))
    v1 = min(height - 1, max(v_corners))

    if u0 < u1 and v0 < v1:
        mask[v0:v1, u0:u1] = True

    return mask

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading UniDepthV2 on device: {device}")

    # Use ViT-S for optimal memory throughput on 8GB host
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14").to(device)
    model.eval()

    # Define camera intrinsics
    K = torch.tensor([
        [320.0,   0.0, 320.0],
        [  0.0, 320.0, 240.0],
        [  0.0,   0.0,   1.0]
    ], dtype=torch.float32, device=device)
    camera = Pinhole(K=K)

    # Warmup
    dummy_input = torch.zeros((1, 3, 480, 640), dtype=torch.float32, device=device)
    with torch.inference_mode():
        _ = model.infer(dummy_input, camera)
    print("Model warmup complete.")

    rgb_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_rgb.npy")))
    print(f"Processing {len(rgb_files)} frames...")

    for rgb_path in rgb_files:
        base = os.path.basename(rgb_path).replace("_rgb.npy", "")
        gt_path = os.path.join(DATA_DIR, f"{base}_gt.npy")
        
        # Parse metadata from filename
        tokens = base.split("_")
        shape = tokens[0]
        dist = float(tokens[1].replace("d", ""))
        y_off = float(tokens[2].replace("y", ""))

        rgb_np = np.load(rgb_path)
        gt_depth = np.load(gt_path)

        # Preprocess for UniDepth
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                predictions = model.infer(rgb_tensor, camera)

        pred_depth = predictions["depth"].squeeze().cpu().numpy().astype(np.float32)

        # Extract obstacle foreground mask
        mask = project_bounding_mask(shape, dist, y_off)

        # Filter out invalid depth / far background pixels within bbox
        valid_obj_mask = mask & (gt_depth < 9.5) & (gt_depth > 0.2)

        # Ground truth surface distance vs predicted surface distance
        if np.sum(valid_obj_mask) > 10:
            median_gt = float(np.median(gt_depth[valid_obj_mask]))
            median_pred = float(np.median(pred_depth[valid_obj_mask]))
        else:
            median_gt = float("nan")
            median_pred = float("nan")

        # Save processed pair and scalar results
        out_file = os.path.join(PRED_DIR, f"{base}_eval.npz")
        np.savez_compressed(
            out_file,
            pred_depth=pred_depth,
            gt_depth=gt_depth,
            mask=valid_obj_mask,
            gt_dist=median_gt,
            pred_dist=median_pred,
            shape=shape,
            nominal_dist=dist,
            y_offset=y_off
        )

    print("UniDepth evaluation complete. Outputs stored in results/predictions.")

if __name__ == "__main__":
    run_evaluation()
