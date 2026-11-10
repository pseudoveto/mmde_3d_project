"""Central configuration for the person-depth pipeline."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics in pixels. cx/cy default to the image centre."""
    fx: float
    fy: float
    cx: Optional[float] = None
    cy: Optional[float] = None

    def to_matrix(self, width: int, height: int) -> np.ndarray:
        cx = self.cx if self.cx is not None else width / 2
        cy = self.cy if self.cy is not None else height / 2
        return np.array([[self.fx, 0, cx], [0, self.fy, cy], [0, 0, 1]], dtype=np.float32)


@dataclass
class SegmentationConfig:
    weights: str = "yolo11m-seg.pt"   # yolo11n-seg.pt (fast) ... yolo11x-seg.pt (best)
    conf: float = 0.4                 # person detection confidence threshold


@dataclass
class DepthConfig:
    backbone: str = "vits"            # vits | vitb | vitl

    @property
    def checkpoint(self) -> str:
        return f"lpiccinelli/unidepth-v2-{self.backbone}14"


@dataclass
class DistanceConfig:
    erode_px: int = 5                 # shrink mask before sampling depth (avoids edge bleeding)
    min_area_px: int = 1500           # smaller masks are flagged low reliability
    trim_low_pct: float = 10.0        # trimmed-median percentiles
    trim_high_pct: float = 90.0


@dataclass
class VisualConfig:
    alpha: float = 0.5                # mask overlay opacity


@dataclass
class PipelineConfig:
    device: str = "auto"              # auto | cuda | cpu | mps
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    intrinsics: Optional[CameraIntrinsics] = None
