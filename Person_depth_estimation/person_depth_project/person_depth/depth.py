"""Monocular metric depth with UniDepthV2."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from unidepth.models import UniDepthV2

from .config import DepthConfig


@dataclass
class DepthResult:
    depth: np.ndarray                    # [H, W] metres (Z along optical axis)
    points: Optional[np.ndarray]         # [3, H, W] metres, or None
    intrinsics: Optional[np.ndarray]     # [3, 3]
    intrinsics_source: str               # "provided" | "predicted"


class DepthEstimator:
    def __init__(self, cfg: DepthConfig, device: str):
        self.cfg = cfg
        self.device = device
        print(f"[depth] loading {cfg.checkpoint} on {device} ...")
        self.model = UniDepthV2.from_pretrained(cfg.checkpoint).to(device).eval()

    def __call__(self, img_rgb: np.ndarray, K: Optional[np.ndarray] = None) -> DepthResult:
        H, W = img_rgb.shape[:2]
        rgb_t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().to(self.device)

        with torch.inference_mode():
            if K is not None:
                from unidepth.utils.camera import Pinhole
                camera = Pinhole(K=torch.from_numpy(K).float().unsqueeze(0).to(self.device))
                pred = self.model.infer(rgb_t, camera=camera)
            else:
                pred = self.model.infer(rgb_t)

        depth = pred["depth"].squeeze().float().cpu().numpy()

        points = pred.get("points")
        if points is not None:
            points = points.squeeze(0).float().cpu().numpy()

        intr = pred.get("intrinsics")
        if intr is not None:
            intr = intr.squeeze().float().cpu().numpy()
        elif K is not None:
            intr = K.astype(np.float32)

        depth, points, intr = self._match_resolution(depth, points, intr, H, W)
        return DepthResult(depth, points, intr, "provided" if K is not None else "predicted")

    @staticmethod
    def _match_resolution(depth, points, intr, H, W):
        """Bring model outputs back to the original image size."""
        if depth.shape != (H, W):
            sx, sy = W / depth.shape[1], H / depth.shape[0]
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            if intr is not None:
                intr = intr.copy()
                intr[0, :] *= sx
                intr[1, :] *= sy
        if points is not None and points.shape[1:] != (H, W):
            points = np.stack(
                [cv2.resize(points[c], (W, H), interpolation=cv2.INTER_LINEAR) for c in range(3)]
            )
        return depth, points, intr
