"""
Build a YOLOv8 detection dataset layout from BD3-Dataset/sample images/class_images.

Output:
    dataset/
      train/
        images/   <- ~80% of images (mixed classes, flat folder)
        labels/   <- one .txt file per image, populated via labelImg
      val/
        images/   <- ~20% of images
        labels/   <- one .txt file per image, populated via labelImg

YOLO label format (one box per line, normalized 0..1):
    <class_id> <x_center> <y_center> <width> <height>

The class id mapping lives in yolo_classes.txt (used by labelImg) and
data.yaml (used by ultralytics). They must be kept in sync.

This script is APPEND-SAFE by default:
  * Images already present in train/ or val/ stay where they are -- their
    label files are NEVER overwritten, so your hand-annotated boxes are
    preserved across re-runs.
  * Brand-new images in BD3-Dataset/sample images/class_images/<cls>/
    get added with an 80/20 stratified split.

Pass --reset to wipe and rebuild from scratch (DESTRUCTIVE).
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DEFAULT = ROOT / "BD3-Dataset" / "sample images" / "class_images"
OUT_DEFAULT = ROOT / "dataset"
TRAIN_RATIO = 0.8
SEED = 42
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=str(SRC_DEFAULT),
                   help="Source folder of per-class image subfolders.")
    p.add_argument("--out", default=str(OUT_DEFAULT),
                   help="Output dataset folder.")
    p.add_argument("--reset", action="store_true",
                   help="DESTRUCTIVE: wipe the output folder and rebuild "
                        "from scratch. Your existing label .txt files will "
                        "be lost. Use only when you want to redo the split.")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def existing_image_basenames(out: Path) -> set[str]:
    seen: set[str] = set()
    for split in ("train", "val"):
        d = out / split / "images"
        if d.exists():
            for f in d.iterdir():
                if f.suffix.lower() in EXTS:
                    seen.add(f.name)
    return seen


def main() -> None:
    args = parse_args()
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    if not src.exists():
        raise SystemExit(f"Source folder not found: {src}")

    random.seed(args.seed)

    if args.reset and out.exists():
        print(f"--reset given: removing {out}")
        shutil.rmtree(out)

    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    already = existing_image_basenames(out)
    if already:
        print(f"Found {len(already)} image(s) already in {out}; "
              f"these will be left in place (labels preserved).")

    by_class: dict[str, list[Path]] = {}
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in EXTS)
        if imgs:
            by_class[cls_dir.name] = imgs

    added_summary: list[tuple[str, int, int]] = []

    for cls_name, imgs in by_class.items():
        new_imgs = [p for p in imgs if p.name not in already]
        if not new_imgs:
            added_summary.append((cls_name, 0, 0))
            continue

        random.shuffle(new_imgs)
        n = len(new_imgs)
        n_train = max(1, round(n * TRAIN_RATIO))
        if n >= 2 and n_train >= n:
            n_train = n - 1
        train_imgs = new_imgs[:n_train]
        val_imgs = new_imgs[n_train:]

        for split_name, items in (("train", train_imgs), ("val", val_imgs)):
            for s in items:
                dst_img = out / split_name / "images" / s.name
                dst_lbl = out / split_name / "labels" / (s.stem + ".txt")
                shutil.copy2(s, dst_img)
                if not dst_lbl.exists():
                    # Empty file = "image has no objects" in YOLO format,
                    # which is also valid (treated as a background sample).
                    dst_lbl.touch()

        added_summary.append((cls_name, len(train_imgs), len(val_imgs)))

    # Final totals across ALL files in the dataset (including ones
    # preserved from previous runs).
    final = {}
    for split in ("train", "val"):
        d = out / split / "images"
        final[split] = sum(1 for f in d.iterdir() if f.suffix.lower() in EXTS)
    non_empty_train = sum(
        1 for f in (out / "train" / "labels").iterdir()
        if f.suffix == ".txt" and f.stat().st_size > 0
    ) if (out / "train" / "labels").exists() else 0
    non_empty_val = sum(
        1 for f in (out / "val" / "labels").iterdir()
        if f.suffix == ".txt" and f.stat().st_size > 0
    ) if (out / "val" / "labels").exists() else 0

    print(f"\nDataset root: {out}\n")
    print(f"{'class':<14}{'newly added (train)':>22}{'newly added (val)':>20}")
    for cls, a, b in added_summary:
        print(f"{cls:<14}{a:>22}{b:>20}")
    print()
    print(f"Total images now in train/    : {final['train']}")
    print(f"Total images now in val/      : {final['val']}")
    print(f"Annotated train labels (>0 B) : {non_empty_train}")
    print(f"Annotated val   labels (>0 B) : {non_empty_val}")
    if non_empty_train == 0:
        print("\nNo annotations yet. Open labelImg with: ./annotate.sh train")
    else:
        print("\nReady to train: python train_yolo.py")


if __name__ == "__main__":
    main()
