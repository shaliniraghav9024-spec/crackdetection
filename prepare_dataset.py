"""
Build a train/val/test split from BD3-Dataset/sample images/class_images.

Output structure (matches the BD3 notebooks' expectation):
    BD3-Dataset/dataset/train-curat-dataset/
        train/<class>/...
        val/<class>/...
        test/<class>/...

Class folder names are normalized to use underscores (e.g. "major_crack")
so they match classes.txt.

NOTE: With only ~5-7 samples per class in the repo, this is just enough to
verify the pipeline runs. Replace the source folder with the full BD3
dataset (3,965 images) for real benchmarking.
"""
from __future__ import annotations

import os
import random
import shutil
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "BD3-Dataset" / "sample images" / "class_images"
OUT = ROOT / "BD3-Dataset" / "dataset" / "train-curat-dataset"

TRAIN_RATIO, VAL_RATIO = 0.6, 0.2  # remainder goes to test
SEED = 42

CLASS_RENAME = {
    "Algae": "algae",
    "major crack": "major_crack",
    "minor crack": "minor_crack",
    "normal": "normal",
    "peeling": "peeling",
    "spalling": "spalling",
    "stain": "stain",
}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source folder not found: {SRC}")

    random.seed(SEED)

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (OUT / split).mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, int, int, int]] = []

    for cls_dir in sorted(SRC.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_out = CLASS_RENAME.get(cls_dir.name, cls_dir.name.replace(" ", "_"))

        images = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        random.shuffle(images)

        n = len(images)
        if n == 0:
            print(f"[skip] {cls_dir.name}: no images")
            continue

        # ImageFolder requires every class to be present in every split,
        # otherwise class indices won't line up across train/val/test.
        # For tiny classes (n < 3) we duplicate images so each split has
        # at least one sample; this lets the pipeline run, but real
        # generalization needs more images.
        if n >= 3:
            n_train = max(1, ceil(n * TRAIN_RATIO))
            n_val = max(1, ceil(n * VAL_RATIO))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            n_test = n - n_train - n_val
            train_items = images[:n_train]
            val_items = images[n_train:n_train + n_val]
            test_items = images[n_train + n_val:]
        elif n == 2:
            train_items, val_items, test_items = [images[0]], [images[1]], [images[0]]
        else:  # n == 1
            train_items = val_items = test_items = [images[0]]

        splits = {"train": train_items, "val": val_items, "test": test_items}

        for split_name, items in splits.items():
            dst_dir = OUT / split_name / cls_out
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in items:
                shutil.copy2(src, dst_dir / src.name)

        summary.append((cls_out, len(train_items), len(val_items), len(test_items)))

    print(f"\nWrote dataset to: {OUT}")
    print(f"{'class':<14}{'train':>8}{'val':>6}{'test':>6}")
    for cls, a, b, c in summary:
        print(f"{cls:<14}{a:>8}{b:>6}{c:>6}")
    totals = tuple(sum(x) for x in zip(*((a, b, c) for _, a, b, c in summary)))
    print(f"{'TOTAL':<14}{totals[0]:>8}{totals[1]:>6}{totals[2]:>6}")


if __name__ == "__main__":
    main()
