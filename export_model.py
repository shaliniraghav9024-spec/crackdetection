#!/usr/bin/env python3
"""
export_model.py
---------------
Export a trained YOLOv11/v8 .pt checkpoint to production-ready formats.

Supported formats and typical use-cases:
  • openvino  — Intel CPU / integrated GPU inference (fastest on-device)
  • onnx      — Universal; works in OpenCV DNN, TensorRT, etc.
  • torchscript — PyTorch serve / embedded Python
  • tflite    — Mobile / edge (ARM)
  • engine    — TensorRT FP16 (fastest on NVIDIA GPU)

Usage:
    # Default: OpenVINO FP32
    python export_model.py --weights runs/detect/yolo11_emc_building/weights/best.pt

    # ONNX with dynamic batch
    python export_model.py --weights best.pt --format onnx --dynamic

    # TensorRT FP16 (requires tensorrt installed)
    python export_model.py --weights best.pt --format engine --half

    # Multiple formats at once
    python export_model.py --weights best.pt --format openvino onnx tflite
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics not found. Run: pip install ultralytics")


SUPPORTED_FORMATS = [
    "openvino", "onnx", "torchscript", "tflite",
    "engine", "coreml", "pb", "saved_model", "paddle",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights",  required=True,
                   help="Path to trained .pt weights.")
    p.add_argument("--format",   nargs="+", default=["openvino"],
                   choices=SUPPORTED_FORMATS,
                   help="Export format(s). Space-separated list.")
    p.add_argument("--imgsz",    type=int, default=960,
                   help="Input image size (match training imgsz for best accuracy).")
    p.add_argument("--batch",    type=int, default=1,
                   help="Batch size for the exported model.")
    p.add_argument("--half",     action="store_true",
                   help="FP16 export (supported: onnx, engine, openvino).")
    p.add_argument("--int8",     action="store_true",
                   help="INT8 quantisation (requires --data for calibration).")
    p.add_argument("--data",     default="data.yaml",
                   help="data.yaml — needed for INT8 calibration.")
    p.add_argument("--dynamic",  action="store_true",
                   help="ONNX dynamic axes (batch, H, W).")
    p.add_argument("--simplify", action="store_true", default=True,
                   help="ONNX simplify via onnxsim (default: True).")
    p.add_argument("--device",   default="0",
                   help="Device for calibration / TRT build.")
    return p.parse_args()


def export_one(weights: Path, fmt: str, args: argparse.Namespace) -> None:
    print(f"\n{'=' * 55}")
    print(f"  Exporting → {fmt.upper()}")
    print(f"{'=' * 55}")
    model = YOLO(str(weights))

    kwargs: dict = dict(
        format=fmt,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    if fmt in ("onnx", "engine", "openvino"):
        kwargs["half"]     = args.half
        kwargs["simplify"] = args.simplify
        if fmt == "onnx":
            kwargs["dynamic"] = args.dynamic
    if args.int8:
        kwargs["int8"] = True
        kwargs["data"] = args.data

    out = model.export(**kwargs)
    print(f"✅ Saved: {out}")


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    print("Building Defect Detection — Model Export")
    print(f"  weights : {weights}")
    print(f"  formats : {args.format}")
    print(f"  imgsz   : {args.imgsz}   half: {args.half}   int8: {args.int8}")

    for fmt in args.format:
        export_one(weights, fmt, args)

    print("\n✅ All exports complete.")
    print("  Update defect_analyzer.py to point to the exported model.")


if __name__ == "__main__":
    main()
