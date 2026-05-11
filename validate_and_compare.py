#!/usr/bin/env python3
"""
validate_and_compare.py
-----------------------
Evaluate one or more trained weights files on the val set and print a
side-by-side mAP comparison table.

Useful for comparing:
  • YOLOv8n  vs  YOLOv8s  vs  YOLOv8m
  • Different training runs / checkpoints
  • Last.pt vs best.pt

Usage:
    # Compare two checkpoints
    python validate_and_compare.py \
        --weights runs/detect/bd3_yolov8n/weights/best.pt \
                  runs/detect/bd3_yolov8s/weights/best.pt \
        --labels "YOLOv8n" "YOLOv8s"

    # Quick single-model validation
    python validate_and_compare.py \
        --weights runs/detect/bd3_yolov8n/weights/best.pt

    # Higher-resolution validation
    python validate_and_compare.py \
        --weights runs/detect/bd3_yolov8n/weights/best.pt \
        --imgsz 960
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics not found. Run: pip install ultralytics")


# BD3 class names (must match data.yaml order)
CLASS_NAMES = [
    "algae", "major_crack", "minor_crack",
    "peeling", "spalling", "stain",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", nargs="+", required=True,
                   help="One or more .pt weight files to evaluate.")
    p.add_argument("--labels",  nargs="+", default=None,
                   help="Display names for each weight file (same order). "
                        "Defaults to the filename.")
    p.add_argument("--data",    default="data.yaml")
    p.add_argument("--imgsz",   type=int, default=960)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--device",  default="0")
    p.add_argument("--conf",    type=float, default=0.001,
                   help="Confidence threshold for val (low = full recall curve).")
    p.add_argument("--iou",     type=float, default=0.6,
                   help="NMS IoU threshold for val.")
    p.add_argument("--half",    action="store_true",
                   help="FP16 inference (GPU only).")
    return p.parse_args()


def validate_one(weights: str, data: str, imgsz: int, batch: int,
                 device: str, conf: float, iou: float, half: bool) -> dict:
    """Run validation and return metrics dict."""
    model = YOLO(weights)
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        half=half,
        plots=False,
        verbose=False,
    )
    return metrics


def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "  N/A "
    if pct:
        return f"{v * 100:6.2f}%"
    return f"{v:8.4f}"


def main() -> None:
    args = parse_args()

    weights_list = args.weights
    labels = args.labels or [Path(w).stem for w in weights_list]

    if len(labels) != len(weights_list):
        raise SystemExit("--labels count must match --weights count")

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    print("=" * 70)
    print("Building Defect Detection — Validation & Comparison")
    print("=" * 70)
    print(f"  data   : {data_yaml}")
    print(f"  imgsz  : {args.imgsz}   conf : {args.conf}   iou : {args.iou}")
    print()

    results_table = []

    for weights, label in zip(weights_list, labels):
        w_path = Path(weights).resolve()
        if not w_path.exists():
            print(f"⚠️  Skipping (not found): {w_path}")
            continue

        print(f"Validating: {label}  ({w_path.name})")
        metrics = validate_one(
            str(w_path), str(data_yaml),
            args.imgsz, args.batch, args.device,
            args.conf, args.iou, args.half,
        )

        # Ultralytics metrics object
        mp   = getattr(metrics.box, "mp",   None)   # mean precision
        mr   = getattr(metrics.box, "mr",   None)   # mean recall
        map50= getattr(metrics.box, "map50",None)   # mAP@0.5
        map  = getattr(metrics.box, "map",  None)   # mAP@0.5:0.95
        maps = getattr(metrics.box, "maps", [])     # per-class mAP50:95

        results_table.append({
            "label": label,
            "P": mp, "R": mr,
            "mAP50": map50,
            "mAP50:95": map,
            "per_class_maps": maps,
        })

        print(f"  Prec={fmt(mp)}  Recall={fmt(mr)}  "
              f"mAP50={fmt(map50)}  mAP50:95={fmt(map)}")
        print()

    if not results_table:
        print("No results to display.")
        return

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    col = 18
    header = f"{'Model':<{col}} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50:95':>10}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results_table:
        print(f"{r['label']:<{col}} "
              f"{fmt(r['P']):>8} "
              f"{fmt(r['R']):>8} "
              f"{fmt(r['mAP50']):>8} "
              f"{fmt(r['mAP50:95']):>10}")
    print("=" * len(header))

    # ------------------------------------------------------------------
    # Per-class breakdown (if multiple models, show best per class)
    # ------------------------------------------------------------------
    if results_table[0]["per_class_maps"]:
        print("\nPer-class mAP50:95:")
        print(f"  {'Class':<14}", end="")
        for r in results_table:
            print(f"  {r['label']:>14}", end="")
        print()
        print("  " + "-" * (16 + 16 * len(results_table)))
        for i, cls in enumerate(CLASS_NAMES):
            print(f"  {cls:<14}", end="")
            for r in results_table:
                maps = r["per_class_maps"]
                v = maps[i] if i < len(maps) else None
                print(f"  {fmt(v):>14}", end="")
            print()

    print()


if __name__ == "__main__":
    main()
