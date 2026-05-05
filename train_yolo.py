"""
Fine-tune YOLOv8 on the BD3 building-defects dataset.

Defaults to ``yolov8s.pt`` -- a good speed/accuracy balance on CPU and
sufficient for the 8-class BD3 detection task.  Augmentation is tuned
for real-world inspection footage:

  * **HSV jitter** -- different lighting, wall paints, time of day.
  * **Mosaic + mixup** -- forces multiple co-occurring defects per
    training image, which matches the production "multiple defects in
    one frame" use-case.
  * **Affine + flip** -- camera angle robustness.
  * **Erasing** -- forces the model to use multiple cues, not just one
    patch (helps avoid overfitting on the BD3 sample being small).
  * Albumentations (Blur, MedianBlur, ToGray, CLAHE) is auto-applied by
    Ultralytics if installed -- this is how blurry / low-light frames
    are handled.  ``pip install albumentations`` to enable.

Training optimisations:
  * **Cosine LR schedule** + warmup (smoother than step decay on small
    datasets).
  * **Early stop on mAP50** with patience 25 -- stops when the model
    has stopped improving rather than over-fitting another 50 epochs.

Pre-flight checks before training:
  * data.yaml exists and is valid.
  * dataset/{train,val}/images contain image files.
  * at least one label .txt file is non-empty (the most common
    "I forgot to draw boxes" mistake).

Usage (with venv activated):
    python train_yolo.py
    python train_yolo.py --epochs 150 --batch 16 --imgsz 640
    python train_yolo.py --model yolov8m.pt   # try medium for higher mAP
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    p.add_argument("--model", default="yolov8m.pt",
                   help="Starting weights. yolov8n.pt is the smallest/fastest; "
                        "yolov8s.pt is a CPU-friendly balance; "
                        "yolov8m.pt (default) gives the best mAP and is what "
                        "the inference pipeline in defect_analyzer.py expects.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size. Lower this if you run out of memory.")
    p.add_argument("--device", default="",
                   help="'' = auto, 'cpu' = force CPU, '0' = first CUDA GPU.")
    p.add_argument("--project", default=None,
                   help="Where Ultralytics writes the run folder. "
                        "Defaults to runs/detect.")
    p.add_argument("--name", default="bd3_yolov8m",
                   help="Run subfolder name; auto-incremented if it exists.")
    p.add_argument("--patience", type=int, default=25,
                   help="Early-stopping patience (epochs without mAP50 "
                        "improvement on the val set).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr0", type=float, default=0.01,
                   help="Initial learning rate.")
    p.add_argument("--lrf", type=float, default=0.01,
                   help="Final LR multiplier (cosine schedule's floor).")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--mosaic", type=float, default=1.0,
                   help="Mosaic probability per image (0..1).")
    p.add_argument("--mixup", type=float, default=0.10,
                   help="Mixup probability per image (0..1).")
    p.add_argument("--erasing", type=float, default=0.40,
                   help="Random erasing probability (0..1) — forces the "
                        "model to use multiple cues per defect.")
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.70)
    p.add_argument("--hsv-v", type=float, default=0.40,
                   help="Value (brightness) jitter — main lever for "
                        "low-light robustness.")
    p.add_argument("--degrees", type=float, default=10.0,
                   help="Rotation range in degrees.")
    p.add_argument("--translate", type=float, default=0.10)
    p.add_argument("--scale", type=float, default=0.50)
    p.add_argument("--shear", type=float, default=2.0)
    p.add_argument("--flipud", type=float, default=0.0,
                   help="Vertical flip probability. Building inspections "
                        "are gravity-correct, so this defaults to 0.")
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--cos-lr", action="store_true", default=True,
                   help="Use cosine LR schedule (default true).")
    p.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    p.add_argument("--skip-checks", action="store_true",
                   help="Skip the pre-flight dataset sanity checks.")
    return p.parse_args()


def preflight(data_yaml_path: Path) -> None:
    if not data_yaml_path.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml_path}")

    with data_yaml_path.open() as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", ".")).expanduser()
    if not base.is_absolute():
        base = (data_yaml_path.parent / base).resolve()
    train_imgs = base / cfg["train"]
    val_imgs = base / cfg["val"]
    train_lbls = Path(str(train_imgs).replace("/images", "/labels"))
    val_lbls = Path(str(val_imgs).replace("/images", "/labels"))
    names = cfg.get("names", {})

    print("Pre-flight checks:")
    print(f"  data.yaml      : {data_yaml_path}")
    print(f"  dataset root   : {base}")
    print(f"  train images   : {train_imgs}")
    print(f"  train labels   : {train_lbls}")
    print(f"  val images     : {val_imgs}")
    print(f"  val labels     : {val_lbls}")
    print(f"  classes ({len(names)}) : {list(names.values())}")

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for label, p in [("train images", train_imgs), ("val images", val_imgs)]:
        if not p.exists():
            raise SystemExit(f"Missing folder: {p}")
        n = sum(1 for f in p.iterdir() if f.suffix.lower() in EXTS)
        if n == 0:
            raise SystemExit(
                f"No images in {p}. Run: python prepare_yolo_dataset.py"
            )
        print(f"  {label:<14}: {n} files")

    non_empty = 0
    if train_lbls.exists():
        for f in train_lbls.iterdir():
            if f.suffix == ".txt" and f.stat().st_size > 0:
                non_empty += 1
    print(f"  train labels   : {non_empty} non-empty .txt file(s)")
    if non_empty == 0:
        raise SystemExit(
            "\nNo annotated training images found.\n"
            "Open labelImg and draw boxes first:\n"
            "    ./annotate.sh train"
        )


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    if not args.skip_checks:
        preflight(data_yaml)
    print()

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        plots=True,
        cos_lr=args.cos_lr,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        # augmentation
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
        degrees=args.degrees, translate=args.translate,
        scale=args.scale, shear=args.shear,
        flipud=args.flipud, fliplr=args.fliplr,
        mosaic=args.mosaic, mixup=args.mixup, erasing=args.erasing,
    )
    if args.project:
        train_kwargs["project"] = args.project
    results = model.train(**train_kwargs)

    save_dir = Path(
        results.save_dir if hasattr(results, "save_dir") else
        (args.project or "runs/detect") + "/" + args.name
    )
    print("\nTraining complete. Best weights are at:")
    print(f"  {save_dir}/weights/best.pt")
    print("\nValidate on the val set with:")
    print(f"  yolo val model={save_dir}/weights/best.pt data={data_yaml}")
    print("Or run inference with:")
    print(f"  python predict_yolo.py --weights {save_dir}/weights/best.pt "
          f"path/to/image_or_video")


if __name__ == "__main__":
    main()
