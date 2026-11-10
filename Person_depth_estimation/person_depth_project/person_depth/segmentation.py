"""Person instance segmentation with Ultralytics YOLO-seg."""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from .config import SegmentationConfig

PERSON_CLASS_ID = 0  # COCO


@dataclass
class PersonDetection:
    mask: np.ndarray      # [H, W] bool
    box: np.ndarray       # [x1, y1, x2, y2] int
    score: float


class PersonSegmenter:
    def __init__(self, cfg: SegmentationConfig, device: str):
        self.cfg = cfg
        self.device = 0 if device == "cuda" else device
        print(f"[seg]   loading {cfg.weights} ...")
        self.model = YOLO(cfg.weights)  # auto-downloads official weights on first run

    def __call__(self, img_bgr: np.ndarray) -> List[PersonDetection]:
        H, W = img_bgr.shape[:2]
        res = self.model.predict(
            img_bgr,
            classes=[PERSON_CLASS_ID],
            conf=self.cfg.conf,
            retina_masks=True,        # masks at original image resolution
            device=self.device,
            verbose=False,
        )[0]

        if res.masks is None or len(res.masks) == 0:
            return []

        masks = res.masks.data.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()

        detections = []
        for m, b, s in zip(masks, boxes, scores):
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            detections.append(PersonDetection(mask=m > 0.5, box=b.astype(int), score=float(s)))
        return detections
