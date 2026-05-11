#!/usr/bin/env python3
"""
YOLOv8 training for the BD3 building-defect dataset.

6 classes: algae, major_crack, minor_crack, peeling, spalling, stain

Usage:
    python train_yolo.py                     # default: yolov8n baseline
    python train_yolo.py --model yolov8s.pt  # step up for +3-5% mAP (needs GPU)
    python train_yolo.py --model yolov8m.pt  # higher capacity (more VRAM)
    python train_yolo.py --epochs 150 --batch 8 --imgsz 640 --device cpu  # CPU-friendly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
except ImportError:
    sys.exit("ultralytics not found. Run: pip install ultralytics")


# ---------------------------------------------------------------------------
# Hyperparameter profile tuned for building-defect detection
# (extreme lighting, small cracks, occlusions — BD3 dataset characteristics)
# ---------------------------------------------------------------------------
BUILDING_DEFECT_HYP = {
    # Optimizer
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    # Warmup
    "warmup_epochs": 5,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    # Loss weights — ↑ BOX weight for precise bbox regression on cracks/holes
    "box": 8.0,
    "cls": 0.75,
    "dfl": 1.5,
    # HSV augmentation — extreme lighting + wall-paint variation
    "hsv_h": 0.02,
    "hsv_s": 0.85,
    "hsv_v": 0.55,
    # Geometric augmentation
    "degrees": 15.0,
    "translate": 0.2,
    "scale": 0.7,
    "shear": 3.0,
    "perspective": 0.0002,
    "flipud": 0.0,   # buildings are gravity-correct → no vertical flip
    "fliplr": 0.5,
    # Mosaic/mixup — rare defect classes benefit most
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.25,
    # Erasing — handles occlusions and partial cracks
    "erasing": 0.65,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",    default="yolov8n.pt",
                   help="Starting weights. yolov8n.pt=fastest, yolov8s/m.pt=higher mAP")
    p.add_argument("--cfg",      default=None,
                   help="Custom model architecture YAML. "
                        "Leave empty to use the stock model arch.")
    p.add_argument("--data",     default="data.yaml", help="Path to data.yaml")
    p.add_argument("--epochs",   type=int,   default=250)
    p.add_argument("--imgsz",    type=int,   default=960,
                   help="Input resolution. 960 helps detect small cracks/holes.")
    p.add_argument("--batch",    type=int,   default=16,
                   help="Batch size. Reduce if OOM.")
    p.add_argument("--device",   default="0",
                   help="'0' = first CUDA GPU, 'cpu' = force CPU, '' = auto")
    p.add_argument("--workers",  type=int,   default=8)
    p.add_argument("--project",  default="runs/detect")
    p.add_argument("--name",     default="bd3_yolov8n")
    p.add_argument("--patience", type=int,   default=30,
                   help="Early-stop patience in epochs.")
    p.add_argument("--save-period", type=int, default=10,
                   help="Save checkpoint every N epochs.")
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--export",   action="store_true",
                   help="Export best.pt to OpenVINO after training.")
    p.add_argument("--skip-checks", action="store_true",
                   help="Skip pre-flight dataset sanity checks.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def preflight(data_yaml_path: Path) -> None:
    """Validate dataset structure before wasting training time."""
    if not data_yaml_path.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml_path}")

    with data_yaml_path.open() as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", ".")).expanduser()
    if not base.is_absolute():
        base = (data_yaml_path.parent / base).resolve()

    train_imgs = base / cfg["train"]
    val_imgs   = base / cfg["val"]
    train_lbls = Path(str(train_imgs).replace("/images", "/labels"))
    val_lbls   = Path(str(val_imgs).replace("/images", "/labels"))
    names      = cfg.get("names", {})

    print("=" * 60)
    print("Pre-flight checks")
    print("=" * 60)
    print(f"  data.yaml      : {data_yaml_path}")
    print(f"  dataset root   : {base}")
    print(f"  train images   : {train_imgs}")
    print(f"  train labels   : {train_lbls}")
    print(f"  val images     : {val_imgs}")
    print(f"  val labels     : {val_lbls}")
    print(f"  classes ({len(names)})  : {list(names.values())}")

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for label, p in [("train images", train_imgs), ("val images", val_imgs)]:
        if not p.exists():
            raise SystemExit(f"Missing folder: {p}")
        n = sum(1 for f in p.iterdir() if f.suffix.lower() in EXTS)
        if n == 0:
            raise SystemExit(
                f"No images in {p}.\nRun: python prepare_yolo_dataset.py"
            )
        print(f"  {label:<14}: {n} files")

    non_empty = 0
    if train_lbls.exists():
        for f in train_lbls.iterdir():
            if f.suffix == ".txt" and f.stat().st_size > 0:
                non_empty += 1
    print(f"  train labels   : {non_empty} non-empty .txt files")
    if non_empty == 0:
        raise SystemExit(
            "\nNo annotated training images found.\n"
            "Open labelImg and draw boxes first:\n"
            "    ./annotate.sh train"
        )
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_yolo(args: argparse.Namespace) -> None:
    data_yaml = Path(args.data).resolve()

    if not args.skip_checks:
        preflight(data_yaml)

    # ------------------------------------------------------------------
    # Model: optionally override architecture with custom cfg YAML
    # ------------------------------------------------------------------
    if args.cfg:
        cfg_path = Path(args.cfg).resolve()
        if not cfg_path.exists():
            raise SystemExit(f"Architecture YAML not found: {cfg_path}")
        print(f"Using custom architecture: {cfg_path}")
        model = YOLO(str(cfg_path))
        # Load pretrained backbone weights into the custom arch
        model.load(args.model)
    else:
        print(f"Using stock model: {args.model}")
        model = YOLO(args.model)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("\nStarting YOLOv8 training ...")
    print(f"  epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}  device={args.device}")
    print()

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        seed=args.seed,
        val=True,
        plots=True,
        # Loss / NMS
        iou=0.7,         # NMS IoU threshold
        conf=0.001,      # low confidence threshold during val for full recall curve
        # Hyperparameters
        **BUILDING_DEFECT_HYP,
    )

    # ------------------------------------------------------------------
    # Locate best weights
    # ------------------------------------------------------------------
    save_dir = Path(
        results.save_dir if hasattr(results, "save_dir")
        else f"{args.project}/{args.name}"
    )
    best_pt = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Best weights : {best_pt}")
    print(f"   Results dir  : {save_dir}")
    print("=" * 60)

    print("\nValidate:")
    print(f"  yolo val model={best_pt} data={data_yaml}")
    print("\nInference:")
    print(f"  python predict_yolo.py --weights {best_pt} <image_or_video>")

    # ------------------------------------------------------------------
    # Optional export
    # ------------------------------------------------------------------
    if args.export:
        print("\nExporting to OpenVINO ...")
        export_model = YOLO(str(best_pt))
        export_model.export(format="openvino", half=False, simplify=True)
        print("✅ OpenVINO export saved alongside best.pt")


def main() -> None:
    args = parse_args()
    init_seeds(args.seed)
    train_yolo(args)


if __name__ == "__main__":
    main()
