"""Person segmentation + metric distance estimation (YOLO-seg + UniDepthV2)."""

from .config import (CameraIntrinsics, DepthConfig, DistanceConfig, PipelineConfig,
                     SegmentationConfig, VisualConfig)
from .pipeline import PersonDepthPipeline, PersonResult, PipelineOutput

__all__ = [
    "CameraIntrinsics", "DepthConfig", "DistanceConfig", "PipelineConfig",
    "SegmentationConfig", "VisualConfig",
    "PersonDepthPipeline", "PersonResult", "PipelineOutput",
]
