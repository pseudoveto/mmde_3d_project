"""End-to-end pipeline: image -> per-person masks + metric distances."""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .depth import DepthEstimator, DepthResult
from .distance import person_distance, range_map
from .segmentation import PersonSegmenter
from .utils import bgr_to_rgb, resolve_device
from .visualization import distinct_colors, render_depth, render_overlay, side_by_side


@dataclass
class PersonResult:
    id: str
    mask: np.ndarray
    box: np.ndarray
    score: float
    z_depth_m: Optional[float]
    range_m: Optional[float]
    pixels_used: int
    mask_area: int
    low_reliability: bool
    color: Tuple[int, int, int] = (0, 255, 0)

    @property
    def distance_m(self) -> Optional[float]:
        """Straight-line distance if available, else Z-depth."""
        return self.range_m if self.range_m is not None else self.z_depth_m

    def to_dict(self) -> dict:
        r3 = lambda v: None if v is None else round(v, 3)
        return {
            "id": self.id,
            "distance_m": r3(self.range_m),
            "z_depth_m": r3(self.z_depth_m),
            "detection_conf": round(self.score, 3),
            "bbox_xyxy": [int(v) for v in self.box],
            "mask_area_px": self.mask_area,
            "pixels_used": self.pixels_used,
            "color_bgr": list(self.color),
            "low_reliability": bool(self.low_reliability),
        }


@dataclass
class PipelineOutput:
    image_bgr: np.ndarray
    people: List[PersonResult]
    depth: DepthResult
    timings: dict

    def overlay(self, alpha: float = 0.5) -> np.ndarray:
        return render_overlay(self.image_bgr, self.people, alpha)

    def depth_vis(self) -> np.ndarray:
        return render_depth(self.depth.depth, self.people)

    def combined(self, alpha: float = 0.5) -> np.ndarray:
        return side_by_side(self.overlay(alpha), self.depth_vis())


class PersonDepthPipeline:
    """Load models once, then call .run(image_bgr) as many times as you like."""

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        cfg = cfg or PipelineConfig()
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.segmenter = PersonSegmenter(cfg.segmentation, self.device)
        self.depth_estimator = DepthEstimator(cfg.depth, self.device)

    def run(self, img_bgr: np.ndarray) -> PipelineOutput:
        H, W = img_bgr.shape[:2]
        K = self.cfg.intrinsics.to_matrix(W, H) if self.cfg.intrinsics else None

        t0 = time.time()
        detections = self.segmenter(img_bgr)
        t1 = time.time()
        depth_res = self.depth_estimator(bgr_to_rgb(img_bgr), K)
        t2 = time.time()

        rng = range_map(depth_res.depth, depth_res.points, depth_res.intrinsics)
        people = self._measure(detections, depth_res.depth, rng)

        timings = {"segmentation_s": round(t1 - t0, 3), "depth_s": round(t2 - t1, 3)}
        return PipelineOutput(img_bgr, people, depth_res, timings)

    def _measure(self, detections, depth, rng) -> List[PersonResult]:
        dcfg = self.cfg.distance
        people = []
        for det in detections:
            z, r, n_px = person_distance(det.mask, depth, rng, dcfg)
            area = int(det.mask.sum())
            people.append(PersonResult(
                id="", mask=det.mask, box=det.box, score=det.score,
                z_depth_m=z, range_m=r, pixels_used=n_px, mask_area=area,
                low_reliability=area < dcfg.min_area_px or z is None,
            ))

        # Nearest person = P1
        people.sort(key=lambda p: p.distance_m if p.distance_m is not None else float("inf"))
        for i, (p, c) in enumerate(zip(people, distinct_colors(len(people))), start=1):
            p.id, p.color = f"P{i}", c
        return people
