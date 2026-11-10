"""
engine.py — UniDepth V2 inference backend for metric 3D reconstruction from a single image.

UniDepth V2 predicts metric depth (in metres) together with its own camera model, so one RGB
photo is enough to recover a true-scale 3D point cloud — no calibration, EXIF, or second view.
Points are expressed in the camera frame (OpenCV convention: x right, y down, z forward).

This module is UI-agnostic; `app.py` wraps it in Gradio.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID: Final[str] = "lpiccinelli/unidepth-v2-vitl14"
MAX_INPUT_SIDE: Final[int] = 2048  # larger inputs are downscaled before inference
MIN_INPUT_SIDE: Final[int] = 32

FloatArray = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]


class EngineError(RuntimeError):
    """Raised for any user-facing failure in loading, preprocessing or inference."""


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PointCloud:
    """Coloured metric point cloud, camera frame, metres."""

    xyz: FloatArray  # (N, 3)
    rgb: UInt8Array  # (N, 3)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def subsample(self, max_points: int, seed: int = 0) -> "PointCloud":
        if len(self) <= max_points:
            return self
        idx = np.random.default_rng(seed).choice(len(self), size=max_points, replace=False)
        return PointCloud(xyz=self.xyz[idx], rgb=self.rgb[idx])

    def extent_m(self) -> tuple[float, float, float]:
        """Robust (1st–99th percentile) scene size as (width, height, depth) in metres."""
        lo, hi = np.percentile(self.xyz, [1, 99], axis=0)
        w, h, d = (hi - lo).tolist()
        return float(w), float(h), float(d)


@dataclass(frozen=True)
class Reconstruction:
    """Dense per-pixel output of one UniDepth pass."""

    rgb: UInt8Array  # (H, W, 3) processed image
    depth: FloatArray  # (H, W) metres
    points: FloatArray  # (H, W, 3) metres, camera frame
    valid: BoolArray  # (H, W) finite, positive, within range
    original_size: tuple[int, int]  # (W, H) before any downscaling
    inference_seconds: float
    device: str

    @property
    def size(self) -> tuple[int, int]:
        return int(self.rgb.shape[1]), int(self.rgb.shape[0])

    def point_cloud(self, remove_edges: bool = True, edge_threshold: float = 0.04) -> PointCloud:
        """Valid pixels as a cloud, optionally dropping 'flying pixels' at depth discontinuities."""
        mask = self.valid & ~depth_edges(self.depth, edge_threshold) if remove_edges else self.valid
        if not mask.any():
            mask = self.valid
        return PointCloud(xyz=self.points[mask], rgb=self.rgb[mask])


# --------------------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------------------


def load_rgb(source: str | Path | Image.Image | np.ndarray, max_side: int = MAX_INPUT_SIDE) -> tuple[UInt8Array, tuple[int, int]]:
    """Load an image from a path / PIL image / array into a uint8 HxWx3 RGB array.

    Handles EXIF orientation, palette/greyscale/RGBA/16-bit inputs, and downscales very
    large images. Returns (rgb, original_size_wh).
    """
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise EngineError(f"Image file not found: {path}")
            with Image.open(path) as im:
                im.load()
                img = ImageOps.exif_transpose(im) or im
        elif isinstance(source, Image.Image):
            img = ImageOps.exif_transpose(source) or source
        elif isinstance(source, np.ndarray):
            img = Image.fromarray(_normalize_array(source))
        else:
            raise EngineError(f"Unsupported image input type: {type(source).__name__}")
    except UnidentifiedImageError as exc:
        raise EngineError("The uploaded file is not a readable image.") from exc
    except OSError as exc:
        raise EngineError(f"Could not decode image: {exc}") from exc

    if img.mode in ("I;16", "I;16B", "I", "F"):
        img = Image.fromarray(_normalize_array(np.asarray(img)))
    if img.mode == "RGBA" or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = background
    img = img.convert("RGB")

    original_size = img.size
    w, h = original_size
    if min(w, h) < MIN_INPUT_SIDE:
        raise EngineError(f"Image too small ({w}x{h}); need at least {MIN_INPUT_SIDE}px per side.")
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        logger.info("Downscaled input from %sx%s to %sx%s", w, h, *img.size)

    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8)), original_size


def _normalize_array(arr: np.ndarray) -> UInt8Array:
    """Convert arbitrary numeric arrays (float, uint16, ...) to uint8 for PIL."""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        raise EngineError("Image contains no finite pixel values.")
    lo, hi = float(arr[finite].min()), float(arr[finite].max())
    if hi <= 1.0 and lo >= 0.0:
        arr = arr * 255.0
    elif hi > 255.0:
        arr = (arr - lo) / max(hi - lo, 1e-12) * 255.0
    return np.clip(np.nan_to_num(arr), 0, 255).astype(np.uint8)


def depth_edges(depth: FloatArray, threshold: float) -> BoolArray:
    """Pixels whose depth jumps by more than `threshold` (relative) to any 4-neighbour."""
    d = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
    p = np.pad(d, 1, mode="edge")
    neighbours = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    with np.errstate(invalid="ignore"):
        jump = np.nanmax(np.abs(neighbours - d) / d, axis=0)
    return np.nan_to_num(jump, nan=0.0) > threshold


# --------------------------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------------------------


class UniDepthEngine:
    """Lazily loads UniDepth V2 and turns RGB images into metric reconstructions."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        resolution_level: int | None = None,
        max_depth_m: float = 80.0,
    ) -> None:
        self.model_id = model_id
        if resolution_level is not None and not 0 <= resolution_level <= 9:
            raise EngineError("resolution_level must be in [0, 9]")
        self.resolution_level = resolution_level  # V2 knob: 0 (fast) .. 9 (detailed); None = model default
        self.max_depth_m = max_depth_m
        self._requested_device = device
        self._model: Any = None
        self._device: str = "cpu"
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Download (first run) and load the model. Safe to call repeatedly."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from unidepth.models import UniDepthV2
            except ImportError as exc:
                raise EngineError(
                    "UniDepth is not installed. Run: "
                    'pip install --no-deps "unidepth @ git+https://github.com/lpiccinelli-eth/UniDepth.git"'
                ) from exc

            device = self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading %s on %s ...", self.model_id, device)
            try:
                model = UniDepthV2.from_pretrained(self.model_id)
            except Exception as exc:  # network / HF hub / checkpoint errors
                raise EngineError(f"Failed to load '{self.model_id}': {exc}") from exc

            # V2 only honours this attribute if it is set explicitly; when absent it uses the full
            # training pixel range, which was the most accurate setting on the UniDepth demo image.
            if self.resolution_level is not None:
                model.resolution_level = self.resolution_level
            model.interpolation_mode = "bilinear"
            self._model = model.to(device).eval()
            self._device = device
            logger.info("Model ready on %s", device)

    def reconstruct(self, source: str | Path | Image.Image | np.ndarray) -> Reconstruction:
        """Run UniDepth on one image and return its dense metric reconstruction."""
        import torch

        self.load()
        rgb, original_size = load_rgb(source)
        h, w = rgb.shape[:2]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(self._device)  # (3, H, W) uint8

        t0 = time.perf_counter()
        try:
            with self._lock, torch.inference_mode():
                predictions: dict[str, Any] = self._model.infer(tensor)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise EngineError("GPU out of memory — try a smaller image.") from exc
        except Exception as exc:
            raise EngineError(f"UniDepth inference failed: {exc}") from exc
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        for key in ("depth", "points"):
            if key not in predictions:
                raise EngineError(f"Model output is missing '{key}' (got {sorted(predictions)})")

        depth = _to_numpy(predictions["depth"]).reshape(h, w)  # (1,1,H,W) -> (H,W)
        points = _to_numpy(predictions["points"]).reshape(3, h, w).transpose(1, 2, 0)  # (H,W,3)

        valid = np.isfinite(depth) & (depth > 1e-3) & (depth < self.max_depth_m) & np.isfinite(points).all(axis=-1)
        if not valid.any():
            raise EngineError("Model returned no valid depth values for this image.")

        return Reconstruction(
            rgb=rgb,
            depth=depth.astype(np.float32),
            points=np.ascontiguousarray(points, dtype=np.float32),
            valid=valid,
            original_size=original_size,
            inference_seconds=elapsed,
            device=self._device,
        )


def _to_numpy(t: Any) -> np.ndarray:
    return t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------


def export_ply(cloud: PointCloud, path: Path) -> Path:
    """Binary little-endian PLY with per-vertex colour (MeshLab, CloudCompare, Open3D, Blender)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cloud)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment UniDepth V2 metric point cloud, camera frame (x right, y down, z forward), metres\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    vertex = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]
    vertex["r"], vertex["g"], vertex["b"] = cloud.rgb[:, 0], cloud.rgb[:, 1], cloud.rgb[:, 2]
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())
    return path
