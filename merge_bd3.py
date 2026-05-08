"""
Merge BD3 (single-class 'crack' dataset) into dataset_remapped/ as major_crack.

Source:
    ~/property_inspector/bd3_dataset/{train,valid,test}/{images,labels}/
    BD3 has one class:  0 = crack

Destination (must already exist — created by remap_labels.py):
    ~/building-defect-detection/dataset_remapped/{train,val}/{images,labels}/

Per the project's 6-class taxonomy:
    0=algae, 1=major_crack, 2=minor_crack, 3=peeling, 4=spalling, 5=stain
BD3's crack (id 0) is mapped to major_crack (id 1) in the new taxonomy.
This is the *only* source of major_crack data.

Behavior:
  - Pre-seeds the kept-hash set with every image already in dataset_remapped/
    so we don't introduce duplicates of merged_dataset content.
  - BD3 train -> dataset_remapped/train
  - BD3 valid -> dataset_remapped/val
  - BD3 test  -> SKIPPED (held back for held-out evaluation; uncomment below
                 to include).
  - Bboxes are clamped to [0,1] for safety (cheap; BD3 looked clean).
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path

BD3_ROOT = Path.home() / "property_inspector/bd3_dataset"
DST_ROOT = Path.home() / "building-defect-detection/dataset_remapped"

# BD3 has 1 class. Map BD3 source ids -> our taxonomy ids.
MAPPING: dict[int, int] = {
    0: 1,  # BD3 'crack' -> our 'major_crack'
}

# BD3 split -> dataset_remapped split. test is intentionally not mapped.
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    # "test":  "val",   # uncomment to also include BD3 test in val
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_BOX_SIDE = 1e-3


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def clamp_bbox(cx: float, cy: float, w: float, h: float
               ) -> tuple[float, float, float, float] | None:
    xmin = max(0.0, cx - w / 2.0)
    xmax = min(1.0, cx + w / 2.0)
    ymin = max(0.0, cy - h / 2.0)
    ymax = min(1.0, cy + h / 2.0)
    new_w = xmax - xmin
    new_h = ymax - ymin
    if new_w < MIN_BOX_SIDE or new_h < MIN_BOX_SIDE:
        return None
    return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, new_w, new_h


def preload_dst_hashes(root: Path) -> set[str]:
    """Hash every image already in dataset_remapped so BD3 dups are skipped."""
    seen: set[str] = set()
    for split in ("train", "val"):
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            continue
        for p in img_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                try:
                    seen.add(file_md5(p))
                except OSError as e:
                    print(f"  warn: cannot hash {p.name}: {e}")
    return seen


def merge_split(bd3_split: str, dst_split: str,
                seen: set[str], stats: Counter) -> None:
    src_img = BD3_ROOT / bd3_split / "images"
    src_lbl = BD3_ROOT / bd3_split / "labels"
    dst_img = DST_ROOT / dst_split / "images"
    dst_lbl = DST_ROOT / dst_split / "labels"

    if not src_img.is_dir():
        print(f"  (skip) {src_img} missing")
        return

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    img_files = sorted(p for p in src_img.iterdir()
                       if p.suffix.lower() in IMG_EXTS)
    print(f"  [{bd3_split} -> {dst_split}] {len(img_files)} BD3 images")

    for img_path in img_files:
        try:
            h = file_md5(img_path)
        except OSError as e:
            print(f"    warn: cannot hash {img_path.name}: {e}")
            stats[f"{dst_split}_skip_unreadable"] += 1
            continue
        if h in seen:
            stats[f"{dst_split}_dup_skipped"] += 1
            continue
        seen.add(h)

        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            stats[f"{dst_split}_skip_no_label"] += 1
            continue

        # Filename collisions vs merged_dataset are unlikely (BD3 names look
        # like roboflow hashes, merged_dataset uses 'mbdd_*' / human names),
        # but prefix anyway to be safe.
        out_stem = f"bd3_{img_path.stem}"
        out_img = dst_img / (out_stem + img_path.suffix.lower())
        out_lbl = dst_lbl / (out_stem + ".txt")

        new_lines: list[str] = []
        with open(lbl_path) as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    stats[f"{dst_split}_bad_row_arity"] += 1
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, hh = (float(v) for v in parts[1:])
                except ValueError:
                    stats[f"{dst_split}_bad_row_numeric"] += 1
                    continue

                if cls not in MAPPING:
                    stats[f"{dst_split}_unknown_class"] += 1
                    continue
                target = MAPPING[cls]

                clamped = clamp_bbox(cx, cy, w, hh)
                if clamped is None:
                    stats[f"{dst_split}_clamped_to_zero"] += 1
                    continue
                ncx, ncy, nw, nh = clamped
                if (nw, nh) != (w, hh) or (ncx, ncy) != (cx, cy):
                    stats[f"{dst_split}_bboxes_clamped"] += 1

                new_lines.append(f"{target} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                stats[f"{dst_split}_rows_kept"] += 1
                stats[f"class_{target}"] += 1

        with open(out_lbl, "w") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")
        shutil.copy2(img_path, out_img)
        stats[f"{dst_split}_images_copied"] += 1
        if not new_lines:
            stats[f"{dst_split}_negatives_kept"] += 1


def main() -> None:
    print(f"BD3 source: {BD3_ROOT}")
    print(f"Dest:       {DST_ROOT}\n")

    if not BD3_ROOT.is_dir():
        raise SystemExit(f"BD3 source not found: {BD3_ROOT}")
    if not DST_ROOT.is_dir():
        raise SystemExit(
            f"{DST_ROOT} does not exist. Run remap_labels.py first."
        )

    print("Pre-hashing existing dataset_remapped/ images for dedup...")
    seen = preload_dst_hashes(DST_ROOT)
    print(f"  pre-seeded {len(seen)} hashes\n")

    stats: Counter = Counter()
    for bd3_split, dst_split in SPLIT_MAP.items():
        print(f"Processing BD3 {bd3_split} -> {dst_split}")
        merge_split(bd3_split, dst_split, seen, stats)
    print()

    print("--- Stats ---")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print()
    print("Next steps:")
    print(f"  python validate_dataset.py --root {DST_ROOT} --num-classes 6 \\")
    print(f"      --report dataset_remapped_report.json")


if __name__ == "__main__":
    main()
