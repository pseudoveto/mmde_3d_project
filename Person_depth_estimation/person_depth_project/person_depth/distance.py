"""Turn a depth map + a person mask into a robust distance."""

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import DistanceConfig


def range_map(depth: np.ndarray, points: Optional[np.ndarray],
              K: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Per-pixel Euclidean distance sqrt(X^2 + Y^2 + Z^2) in metres."""
    if points is not None:
        return np.linalg.norm(points, axis=0)
    if K is not None:
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        return np.sqrt(X ** 2 + Y ** 2 + depth ** 2)
    return None


def trimmed_median(values: np.ndarray, lo_pct: float, hi_pct: float) -> Optional[float]:
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    trimmed = values[(values >= lo) & (values <= hi)]
    return float(np.median(trimmed if trimmed.size else values))


def core_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    """Erode the mask; fall back to the raw mask if erosion wipes out a small person."""
    if erode_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    eroded = cv2.erode(mask.astype(np.uint8), k) > 0
    return eroded if eroded.sum() >= max(50, 0.2 * mask.sum()) else mask


def person_distance(mask: np.ndarray, depth: np.ndarray, rng: Optional[np.ndarray],
                    cfg: DistanceConfig) -> Tuple[Optional[float], Optional[float], int]:
    """Return (z_depth_m, range_m, pixels_used)."""
    core = core_mask(mask, cfg.erode_px)
    z = trimmed_median(depth[core], cfg.trim_low_pct, cfg.trim_high_pct)
    r = trimmed_median(rng[core], cfg.trim_low_pct, cfg.trim_high_pct) if rng is not None else None
    return z, r, int(core.sum())
