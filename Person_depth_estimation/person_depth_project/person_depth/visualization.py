"""Drawing: coloured person masks, distance labels and depth map."""

import colorsys
from typing import List, Tuple

import cv2
import numpy as np


def distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Well-separated BGR colours using the golden-ratio hue trick."""
    cols, h = [], 0.0
    for _ in range(n):
        h = (h + 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
    return cols


def _contours(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_label(img: np.ndarray, text: str, x: int, y: int, color) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y = max(y, th + base + 4)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - base - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def render_overlay(img_bgr: np.ndarray, people, alpha: float = 0.5) -> np.ndarray:
    """people: list of PersonResult (see pipeline.py)."""
    overlay = img_bgr.copy()
    for p in people:
        overlay[p.mask] = p.color
    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

    for p in people:
        cv2.drawContours(out, _contours(p.mask), -1, p.color, 2)
        d = p.distance_m
        text = f"{p.id}: {d:.2f} m" if d is not None else f"{p.id}: n/a"
        if p.low_reliability:
            text += " (?)"
        x1, y1 = int(p.box[0]), int(p.box[1])
        draw_label(out, text, x1, y1 - 4, p.color)
    return out


def render_depth(depth: np.ndarray, people) -> np.ndarray:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap(((1 - norm) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)  # near = bright
    for p in people:
        cv2.drawContours(vis, _contours(p.mask), -1, (255, 255, 255), 1)
    cv2.putText(vis, f"depth {lo:.1f}-{hi:.1f} m", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.hstack([a, b])
