"""Saving images, raw depth and the JSON report."""

import json
import os
from typing import Dict

import cv2
import numpy as np

from .config import PipelineConfig
from .pipeline import PipelineOutput


def build_report(out: PipelineOutput, cfg: PipelineConfig, image_path: str) -> dict:
    H, W = out.image_bgr.shape[:2]
    intr = out.depth.intrinsics
    return {
        "image": image_path,
        "image_size": [W, H],
        "depth_model": cfg.depth.checkpoint,
        "seg_model": cfg.segmentation.weights,
        "intrinsics": intr.tolist() if intr is not None else None,
        "intrinsics_source": out.depth.intrinsics_source,
        "timings": out.timings,
        "num_people": len(out.people),
        "people": [p.to_dict() for p in out.people],
    }


def save_outputs(out: PipelineOutput, cfg: PipelineConfig, image_path: str,
                 out_dir: str, save_depth_npy: bool = False) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    alpha = cfg.visual.alpha

    paths = {
        "people": os.path.join(out_dir, f"{stem}_people.png"),
        "depth": os.path.join(out_dir, f"{stem}_depth.png"),
        "combined": os.path.join(out_dir, f"{stem}_combined.png"),
        "json": os.path.join(out_dir, f"{stem}_results.json"),
    }
    overlay, depth_vis = out.overlay(alpha), out.depth_vis()
    cv2.imwrite(paths["people"], overlay)
    cv2.imwrite(paths["depth"], depth_vis)
    cv2.imwrite(paths["combined"], np.hstack([overlay, depth_vis]))

    with open(paths["json"], "w") as f:
        json.dump(build_report(out, cfg, image_path), f, indent=2)

    if save_depth_npy:
        paths["depth_npy"] = os.path.join(out_dir, f"{stem}_depth.npy")
        np.save(paths["depth_npy"], out.depth.depth)
    return paths


def print_summary(out: PipelineOutput, paths: Dict[str, str]) -> None:
    print(f"[time]  segmentation {out.timings['segmentation_s']}s | depth {out.timings['depth_s']}s")
    print(f"\nFound {len(out.people)} person(s):")
    for p in out.people:
        d = p.to_dict()
        flag = "  [low reliability]" if p.low_reliability else ""
        print(f"  {d['id']}: distance={d['distance_m']} m | Z-depth={d['z_depth_m']} m"
              f" | conf={d['detection_conf']}{flag}")
    print("\nSaved:")
    for path in paths.values():
        print(f"  {path}")
