"""
Web UI: upload an image, get the colour-masked image plus each person's distance.

    python app.py                      # then open http://localhost:8000
    python app.py --backbone vitl --port 8080
"""

import argparse
import base64
import io
import os
import threading

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from person_depth import (DepthConfig, PersonDepthPipeline, PipelineConfig,
                          SegmentationConfig, VisualConfig)
from person_depth.io_utils import build_report

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description="Person depth web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backbone", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--seg-model", default=os.path.join(HERE, "yolo11m-seg.pt"))
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> BGR uint8. Falls back to Pillow (+ pillow-heif if installed) for HEIC etc."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image, ImageOps
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        raise HTTPException(400, "Could not read image (for HEIC: pip install pillow-heif)")


def to_png_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def to_jpg_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def face_crop(img_bgr: np.ndarray, person, size: int = 112) -> np.ndarray:
    """Square head thumbnail for one person, taken from the top of their mask."""
    H, W = img_bgr.shape[:2]
    rows = np.nonzero(person.mask.any(axis=1))[0]
    if rows.size == 0:
        x1, y1, x2, y2 = [int(v) for v in person.box]
        return cv2.resize(img_bgr[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)
    top, bottom = rows[0], rows[-1]

    # Head is roughly the top 1/7 of a standing person; clamp for tiny / cut-off people.
    s = max((bottom - top) * 0.2, 32)
    head = person.mask[top:top + max(int(s * 0.6), 1)]
    cols = np.nonzero(head.any(axis=0))[0]
    cx = cols.mean() if cols.size else person.box[[0, 2]].mean()
    cy = top + s * 0.42

    half = s / 2
    a, b = int(max(cy - half, 0)), int(min(cy + half, H))
    c, d = int(max(cx - half, 0)), int(min(cx + half, W))
    crop = img_bgr[a:b, c:d]
    if crop.size == 0:
        crop = img_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def create_app(cfg: PipelineConfig) -> FastAPI:
    print("Loading models...")
    pipeline = PersonDepthPipeline(cfg)
    lock = threading.Lock()  # one inference at a time on the GPU
    print(f"Models ready on {pipeline.device}")

    app = FastAPI(title="Person Depth")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(HERE, "static", "index.html"))

    @app.post("/predict")
    def predict(file: UploadFile = File(...)):
        img = decode_image(file.file.read())
        with lock:
            out = pipeline.run(img)
        report = build_report(out, cfg, file.filename)
        for d, p in zip(report["people"], out.people):
            d["face_jpg"] = to_jpg_b64(face_crop(img, p))
        report["overlay_png"] = to_png_b64(out.overlay(cfg.visual.alpha))
        report["depth_png"] = to_png_b64(out.depth_vis())
        return report

    return app


def main():
    args = parse_args()
    cfg = PipelineConfig(
        device=args.device,
        segmentation=SegmentationConfig(weights=args.seg_model, conf=args.conf),
        depth=DepthConfig(backbone=args.backbone),
        visual=VisualConfig(alpha=args.alpha),
    )
    uvicorn.run(create_app(cfg), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
