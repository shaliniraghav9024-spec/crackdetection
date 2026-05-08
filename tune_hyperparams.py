#!/usr/bin/env python3
"""
tune_hyperparams.py
-------------------
Ray Tune / Ultralytics built-in hyperparameter search for mAP optimisation.

Uses Ultralytics' .tune() API which wraps Ray Tune under the hood and
requires:  pip install ray[tune]

This script runs a short evolutionary search (default 30 iterations) over
the most impactful hyperparameters for building-defect detection, then
prints the best config to stdout and saves it to hyp_best.yaml.

Usage:
    python tune_hyperparams.py
    python tune_hyperparams.py --iterations 50 --epochs 30
    python tune_hyperparams.py --model yolo11s.pt --gpu 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
except ImportError:
    raise SystemExit("ultralytics not found. Run: pip install ultralytics")

import yaml


# Search space: only the hyperparameters with the highest mAP sensitivity
# for fine-grained defect detection tasks.
TUNE_SPACE = {
    "lr0":          (1e-5, 1e-1),
    "lrf":          (0.001, 0.1),
    "momentum":     (0.7, 0.98),
    "weight_decay": (0.0, 0.001),
    "box":          (3.0, 12.0),    # bbox regression loss weight
    "cls":          (0.2, 2.0),     # classification loss weight
    "dfl":          (0.5, 3.0),     # distribution focal loss weight
    "hsv_s":        (0.0, 0.9),
    "hsv_v":        (0.0, 0.9),
    "mixup":        (0.0, 0.3),
    "copy_paste":   (0.0, 0.4),
    "erasing":      (0.0, 0.9),
    "mosaic":       (0.0, 1.0),
    "scale":        (0.3, 0.9),
    "translate":    (0.0, 0.3),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",      default="yolo11n.pt")
    p.add_argument("--data",       default="data.yaml")
    p.add_argument("--iterations", type=int, default=30,
                   help="Number of Ray Tune iterations.")
    p.add_argument("--epochs",     type=int, default=30,
                   help="Epochs per trial. Keep low (20-50) during search.")
    p.add_argument("--imgsz",      type=int, default=640,
                   help="Image size during tuning (use 640 for speed).")
    p.add_argument("--batch",      type=int, default=16)
    p.add_argument("--gpu",        default="0",
                   help="Device: '0' = first GPU, 'cpu' = CPU.")
    p.add_argument("--project",    default="runs/tune")
    p.add_argument("--name",       default="bd3_tune")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--output",     default="hyp_best.yaml",
                   help="Where to save the best hyperparameters.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_seeds(args.seed)

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    print("=" * 60)
    print(f"Hyperparameter tuning — {args.iterations} iterations × {args.epochs} epochs")
    print(f"Model : {args.model}  |  imgsz : {args.imgsz}  |  device : {args.gpu}")
    print("=" * 60)

    model = YOLO(args.model)

    # Ultralytics .tune() runs Ray Tune internally
    result = model.tune(
        data=str(data_yaml),
        epochs=args.epochs,
        iterations=args.iterations,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.gpu,
        project=args.project,
        name=args.name,
        plots=False,
        save=False,
        val=True,
        space=TUNE_SPACE,
    )

    # Save best config
    best_hyp_path = Path(args.output)
    with best_hyp_path.open("w") as f:
        yaml.dump(result, f, default_flow_style=False)

    print("\n" + "=" * 60)
    print(f" Best hyperparameters saved to: {best_hyp_path}")
    print("   Use them in training:")
    print(f"   python train_yolo.py  (hyp_best.yaml is auto-picked up)")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
