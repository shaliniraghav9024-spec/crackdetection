"""
Remap class IDs in YOLO labels from merged_dataset's taxonomy to ours,
with bbox edge-clamping and exact-duplicate image dedup.

Source (raw merged_dataset class IDs, preserved untouched):
    ~/building-defect-detection/dataset_old/{train,val}/{images,labels}/

Output (new, remapped, deduped, clamped):
    ~/building-defect-detection/dataset_remapped/{train,val}/{images,labels}/

This script does NOT touch ~/building-defect-detection/dataset/ or dataset_old/.

Mapping (source -> target IDs, against the new 6-class taxonomy):
  0 crack       -> 2 minor_crack
  1 leakage     -> 5 stain
  2 abscission  -> 3 peeling
  3 corrosion   -> 4 spalling
  4 bulge       -> 3 peeling   (no source instances anyway)
  5 algae       -> 0 algae

Target taxonomy (data.yaml / classes.txt, 6 classes):
    0=algae, 1=major_crack, 2=minor_crack,
    3=peeling, 4=spalling,  5=stain

(major_crack=1 is filled by merge_bd3.py — not by this script.)
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path

SRC_ROOT = Path.home() / "building-defect-detection/dataset_old"
DST_ROOT = Path.home() / "building-defect-detection/dataset_remapped"

# source_class_id -> target_class_id (None means drop)
MAPPING: dict[int, int | None] = {
    0: 2,  # crack       -> minor_crack
    1: 5,  # leakage     -> stain
    2: 3,  # abscission  -> peeling
    3: 4,  # corrosion   -> spalling
    4: 3,  # bulge       -> peeling
    5: 0,  # algae       -> algae
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Drop a clamped bbox if either dimension shrinks below this fraction.
# 1e-3 of image = ~1 pixel on a 1000px image. Anything smaller is noise.
MIN_BOX_SIDE = 1e-3


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def clamp_bbox(cx: float, cy: float, w: float, h: float
               ) -> tuple[float, float, float, float] | None:
    """Clip a YOLO bbox to [0,1]; return None if it collapses."""
    xmin = max(0.0, cx - w / 2.0)
    xmax = min(1.0, cx + w / 2.0)
    ymin = max(0.0, cy - h / 2.0)
    ymax = min(1.0, cy + h / 2.0)
    new_w = xmax - xmin
    new_h = ymax - ymin
    if new_w < MIN_BOX_SIDE or new_h < MIN_BOX_SIDE:
        return None
    return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, new_w, new_h


def remap_split(split: str, kept_hashes: set[str], stats: Counter) -> None:
    src_img = SRC_ROOT / split / "images"
    src_lbl = SRC_ROOT / split / "labels"
    dst_img = DST_ROOT / split / "images"
    dst_lbl = DST_ROOT / split / "labels"

    if not src_lbl.exists():
        print(f"  (skip) {src_lbl} missing")
        return

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    img_files = [p for p in src_img.iterdir()
                 if p.suffix.lower() in IMG_EXTS]
    img_files.sort()  # deterministic dedup behavior
    print(f"  [{split}] {len(img_files)} source images")

    for img_path in img_files:
        # ---- dedup ----
        try:
            h = file_md5(img_path)
        except OSError as e:
            print(f"    warn: cannot hash {img_path.name}: {e}")
            stats[f"{split}_skip_unreadable"] += 1
            continue
        if h in kept_hashes:
            stats[f"{split}_dup_skipped"] += 1
            continue
        kept_hashes.add(h)

        # ---- label exists? ----
        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            stats[f"{split}_skip_no_label"] += 1
            continue

        # ---- remap rows ----
        new_lines: list[str] = []
        with open(lbl_path) as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    stats[f"{split}_bad_row_arity"] += 1
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, hh = (float(v) for v in parts[1:])
                except ValueError:
                    stats[f"{split}_bad_row_numeric"] += 1
                    continue

                if cls not in MAPPING:
                    stats[f"{split}_unknown_class"] += 1
                    continue
                target = MAPPING[cls]
                if target is None:
                    stats[f"{split}_dropped_by_mapping"] += 1
                    continue

                clamped = clamp_bbox(cx, cy, w, hh)
                if clamped is None:
                    stats[f"{split}_clamped_to_zero"] += 1
                    continue
                ncx, ncy, nw, nh = clamped
                if (nw, nh) != (w, hh) or (ncx, ncy) != (cx, cy):
                    stats[f"{split}_bboxes_clamped"] += 1

                new_lines.append(f"{target} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                stats[f"{split}_rows_kept"] += 1
                stats[f"class_{target}"] += 1

        # ---- write outputs (always copy image; empty label = negative) ----
        with open(dst_lbl / lbl_path.name, "w") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")
        shutil.copy2(img_path, dst_img / img_path.name)
        stats[f"{split}_images_copied"] += 1
        if not new_lines:
            stats[f"{split}_negatives_kept"] += 1

    print(f"  [{split}] done.")


def main() -> None:
    print(f"Source: {SRC_ROOT}")
    print(f"Dest:   {DST_ROOT}\n")

    if not SRC_ROOT.is_dir():
        raise SystemExit(f"Source not found: {SRC_ROOT}")
    if DST_ROOT.exists() and any(DST_ROOT.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite non-empty {DST_ROOT}.\n"
            f"Delete or move it first: rm -rf {DST_ROOT}"
        )

    src_names = ["crack", "leakage", "abscission", "corrosion", "bulge", "algae"]
    dst_names = ["algae", "major_crack", "minor_crack",
                 "peeling", "spalling", "stain"]
    print("Mapping (src -> dst):")
    for src_id, tgt in MAPPING.items():
        if tgt is None:
            print(f"  {src_id} {src_names[src_id]:<10} -> DROP")
        else:
            print(f"  {src_id} {src_names[src_id]:<10} -> {tgt} {dst_names[tgt]}")
    print()

    # Hashes of every kept image are shared across splits — duplicates that
    # span splits would be a leak from train into val, so we want to catch
    # them. Pre-seed val first so val wins ties (val is smaller and
    # train-leakage is the worse failure mode).
    kept_hashes: set[str] = set()
    stats: Counter = Counter()

    for split in ("val", "train"):
        print(f"Processing split: {split}")
        remap_split(split, kept_hashes, stats)
    print()

    print("--- Stats ---")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print()
    print("Class counts in output (class_<id>):")
    for k in sorted(stats):
        if k.startswith("class_"):
            cid = int(k.split("_", 1)[1])
            print(f"  {cid} {dst_names[cid]:<12}: {stats[k]}")
    print()
    print(f"Output ready at: {DST_ROOT}")
    print("Next steps:")
    print(f"  1) python merge_bd3.py            # add BD3 cracks as major_crack")
    print(f"  2) python validate_dataset.py --root {DST_ROOT} --num-classes 6")
    print( "  3) review the report, then swap dataset_remapped/ -> dataset/")


if __name__ == "__main__":
    main()
