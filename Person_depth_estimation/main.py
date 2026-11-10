"""
CLI entry point.

    python main.py --image people.jpg
    python main.py --image people.jpg --backbone vitl --out-dir outputs
    python main.py --image people.jpg --fx 1400 --fy 1400
    python main.py --image img1.jpg img2.jpg img3.jpg        # models load once, reused for all
"""

import argparse

from person_depth import (CameraIntrinsics, DepthConfig, DistanceConfig, PersonDepthPipeline,
                          PipelineConfig, SegmentationConfig, VisualConfig)
from person_depth.io_utils import print_summary, save_outputs
from person_depth.utils import load_image


def parse_args():
    ap = argparse.ArgumentParser(description="Person segmentation + metric distance (UniDepthV2)")
    ap.add_argument("--image", required=True, nargs="+", help="One or more input images")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--backbone", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--seg-model", default="yolo11m-seg.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--erode", type=int, default=5)
    ap.add_argument("--min-area", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fx", type=float)
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--save-depth-npy", action="store_true")
    return ap.parse_args()


def build_config(args) -> PipelineConfig:
    intrinsics = None
    if args.fx is not None and args.fy is not None:
        intrinsics = CameraIntrinsics(args.fx, args.fy, args.cx, args.cy)
    return PipelineConfig(
        device=args.device,
        segmentation=SegmentationConfig(weights=args.seg_model, conf=args.conf),
        depth=DepthConfig(backbone=args.backbone),
        distance=DistanceConfig(erode_px=args.erode, min_area_px=args.min_area),
        visual=VisualConfig(alpha=args.alpha),
        intrinsics=intrinsics,
    )


def main():
    args = parse_args()
    cfg = build_config(args)
    pipeline = PersonDepthPipeline(cfg)

    for image_path in args.image:
        print(f"\n=== {image_path} ===")
        out = pipeline.run(load_image(image_path))
        paths = save_outputs(out, cfg, image_path, args.out_dir, args.save_depth_npy)
        print_summary(out, paths)


if __name__ == "__main__":
    main()
