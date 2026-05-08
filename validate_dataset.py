"""
Read-only validator for a YOLO-format dataset.

Checks (no files are modified):
  1. Structure        — expected train/val splits with images/ + labels/
  2. Duplicates       — MD5 hash collisions across all images in the dataset
  3. Corruption       — images that fail to decode (PIL verify)
  4. Missing labels   — image with no matching .txt
  5. Orphan labels    — .txt with no matching image
  6. Invalid YOLO     — bad rows: wrong arity, non-numeric, coords outside [0,1],
                       zero/negative w/h, class id outside the declared range
  7. Class distribution — instance count per class id, per split

Usage:
    python validate_dataset.py --root ~/building-defect-detection/dataset_old \
                               --num-classes 6 --workers 8 \
                               --report dataset_old_report.json

Exit code is 0 even when issues are found — this script reports, it does
not fix. A non-zero exit only indicates the script itself failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    raise SystemExit("Pillow is required. Run: pip install pillow")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------- worker functions (top-level for ProcessPoolExecutor) ----------
def _hash_and_check_image(path_str: str) -> tuple[str, str | None, str | None]:
    """
    Returns (path, md5_hex_or_None, error_or_None).
    md5 is None if the file failed to read at all.
    """
    p = Path(path_str)
    md5 = None
    err = None
    try:
        h = hashlib.md5()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        md5 = h.hexdigest()
    except OSError as e:
        return path_str, None, f"read_error: {e}"

    try:
        with Image.open(p) as im:
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        err = f"decode_error: {type(e).__name__}: {e}"

    return path_str, md5, err


def _scan_label(path_str: str, num_classes: int) -> dict:
    """Inspect one label file. Returns row stats + per-row errors."""
    p = Path(path_str)
    rows = 0
    bad_rows: list[str] = []
    class_counts: Counter[int] = Counter()

    try:
        text = p.read_text()
    except OSError as e:
        return {
            "path": path_str, "rows": 0, "bad_rows": [f"read_error: {e}"],
            "class_counts": {},
        }

    for ln_idx, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        rows += 1
        parts = line.split()
        if len(parts) != 5:
            bad_rows.append(f"line {ln_idx}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            bad_rows.append(f"line {ln_idx}: non-numeric tokens")
            continue
        if not (0 <= cls < num_classes):
            bad_rows.append(f"line {ln_idx}: class_id={cls} out of range [0,{num_classes})")
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            bad_rows.append(f"line {ln_idx}: center outside [0,1]: x={x} y={y}")
            continue
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            bad_rows.append(f"line {ln_idx}: w/h outside (0,1]: w={w} h={h}")
            continue
        # Bbox must fit inside image: center +/- half-extent stays in [0,1]
        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 \
           or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            bad_rows.append(
                f"line {ln_idx}: bbox extends outside image "
                f"(x={x},y={y},w={w},h={h})"
            )
            continue
        class_counts[cls] += 1

    return {
        "path": path_str, "rows": rows, "bad_rows": bad_rows,
        "class_counts": dict(class_counts),
    }


# ---------- main validator ----------
def collect_split(split_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Return (image_stem -> img_path, label_stem -> lbl_path) for a split."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    if img_dir.is_dir():
        for p in img_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                images[p.stem] = p
    if lbl_dir.is_dir():
        for p in lbl_dir.iterdir():
            if p.suffix.lower() == ".txt" and p.is_file():
                labels[p.stem] = p
    return images, labels


def validate_root(root: Path, num_classes: int, workers: int) -> dict:
    report: dict = {
        "root": str(root),
        "num_classes": num_classes,
        "splits": {},
        "duplicates": [],          # filled in across splits
        "summary": {},
    }

    splits_present: list[str] = []
    for split in ("train", "val", "valid", "test"):
        if (root / split).is_dir():
            splits_present.append(split)
    if not splits_present:
        report["error"] = f"No train/val/valid/test dirs under {root}"
        return report
    print(f"[structure] splits present: {splits_present}")

    all_image_paths: list[Path] = []
    all_image_hashes: dict[str, list[str]] = defaultdict(list)  # md5 -> [paths]

    # Per-split structural / orphan / missing checks (cheap, sequential)
    for split in splits_present:
        sdir = root / split
        images, labels = collect_split(sdir)
        missing_labels = sorted(set(images) - set(labels))
        orphan_labels  = sorted(set(labels) - set(images))
        report["splits"][split] = {
            "image_count": len(images),
            "label_count": len(labels),
            "missing_labels": missing_labels[:50],
            "missing_labels_total": len(missing_labels),
            "orphan_labels":  orphan_labels[:50],
            "orphan_labels_total":  len(orphan_labels),
        }
        all_image_paths.extend(images.values())
        print(f"[{split}] images={len(images)}  labels={len(labels)}  "
              f"missing_labels={len(missing_labels)}  orphan_labels={len(orphan_labels)}")

    # ----- parallel: hash + image decode -----
    print(f"\n[hash+decode] running on {len(all_image_paths)} images "
          f"with {workers} workers...")
    decode_errors: list[tuple[str, str]] = []
    read_errors: list[tuple[str, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_hash_and_check_image, str(p)) for p in all_image_paths]
        for fut in as_completed(futures):
            path, md5, err = fut.result()
            if md5 is not None:
                all_image_hashes[md5].append(path)
            if err and err.startswith("read_error"):
                read_errors.append((path, err))
            elif err:
                decode_errors.append((path, err))
            done += 1
            if done % 5000 == 0 or done == len(all_image_paths):
                print(f"    {done}/{len(all_image_paths)}")

    duplicates = [
        {"hash": h, "paths": paths}
        for h, paths in all_image_hashes.items()
        if len(paths) > 1
    ]
    report["duplicates"] = duplicates
    report["duplicate_image_count"] = sum(len(d["paths"]) - 1 for d in duplicates)
    report["decode_errors"] = decode_errors[:200]
    report["decode_error_total"] = len(decode_errors)
    report["read_errors"] = read_errors[:200]
    report["read_error_total"] = len(read_errors)
    print(f"    duplicate hash groups: {len(duplicates)}  "
          f"redundant images: {report['duplicate_image_count']}")
    print(f"    decode errors: {len(decode_errors)}  read errors: {len(read_errors)}")

    # ----- parallel: label scan -----
    label_paths: list[Path] = []
    for split in splits_present:
        lbl_dir = root / split / "labels"
        if lbl_dir.is_dir():
            label_paths.extend(p for p in lbl_dir.iterdir()
                               if p.suffix.lower() == ".txt")

    print(f"\n[labels] scanning {len(label_paths)} label files...")
    per_split_class_counts: dict[str, Counter] = defaultdict(Counter)
    bad_label_files: list[dict] = []
    empty_label_files = 0
    total_rows = 0
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_scan_label, str(p), num_classes): p for p in label_paths
        }
        for fut in as_completed(futures):
            res = fut.result()
            p = Path(res["path"])
            split = p.parts[-3]   # .../<split>/labels/<file>
            total_rows += res["rows"]
            if res["rows"] == 0:
                empty_label_files += 1
            for cls, n in res["class_counts"].items():
                per_split_class_counts[split][cls] += n
            if res["bad_rows"]:
                bad_label_files.append({
                    "path": res["path"],
                    "errors": res["bad_rows"][:5],
                    "error_count": len(res["bad_rows"]),
                })
            done += 1
            if done % 10000 == 0 or done == len(label_paths):
                print(f"    {done}/{len(label_paths)}")

    report["total_label_rows"] = total_rows
    report["empty_label_files"] = empty_label_files
    report["bad_label_files"] = bad_label_files[:200]
    report["bad_label_file_total"] = len(bad_label_files)
    report["class_distribution"] = {
        split: dict(sorted(c.items())) for split, c in per_split_class_counts.items()
    }
    print(f"    total annotation rows: {total_rows}")
    print(f"    empty label files (negatives): {empty_label_files}")
    print(f"    label files with errors: {len(bad_label_files)}")

    # Summary
    report["summary"] = {
        "splits": {s: report["splits"][s]["image_count"] for s in splits_present},
        "duplicate_redundant_images": report["duplicate_image_count"],
        "decode_errors": report["decode_error_total"],
        "read_errors": report["read_error_total"],
        "missing_labels": sum(report["splits"][s]["missing_labels_total"]
                              for s in splits_present),
        "orphan_labels":  sum(report["splits"][s]["orphan_labels_total"]
                              for s in splits_present),
        "bad_label_files": report["bad_label_file_total"],
        "empty_label_files": empty_label_files,
        "total_annotation_rows": total_rows,
        "class_distribution": report["class_distribution"],
    }
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="Dataset root containing train/val (or valid/test).")
    ap.add_argument("--num-classes", required=True, type=int,
                    help="Expected class count for class-id range check.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", type=Path, default=None,
                    help="Optional path to write JSON report.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    print(f"Validating: {root}")
    print(f"Expected num_classes: {args.num_classes}\n")
    report = validate_root(root, args.num_classes, args.workers)

    print("\n========== SUMMARY ==========")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print("=============================")

    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nFull JSON report: {args.report}")


if __name__ == "__main__":
    main()
