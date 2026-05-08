"""
Merge a YOLO dataset into this project's training set, with deduplication.

Source:      ~/property_inspector/merged_dataset
Destination: ~/building-defect-detection/dataset
Pre-seed:    ~/property_inspector/bd3_dataset  (its images are treated as already-seen)
"""

from pathlib import Path
import shutil
import hashlib

# ---------- Paths ----------
SRC = Path.home() / "property_inspector/merged_dataset"
DST = Path.home() / "building-defect-detection/dataset"
BD3 = Path.home() / "property_inspector/bd3_dataset"

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Hashes of every image we've decided to keep (or already exist elsewhere).
# Any new image whose hash is in this set gets skipped.
seen_hashes: set[str] = set()


# ---------- Helpers ----------
def file_hash(path: Path) -> str:
    """Return MD5 hex digest of a file's bytes (streamed in 8 KB chunks)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def preload_hashes(root: Path) -> int:
    """Add the MD5 of every image under root/<split>/images to seen_hashes."""
    if not root.exists():
        print(f"  (skip) {root} does not exist")
        return 0
    n = 0
    for split in ["train", "valid", "val", "test"]:
        img_dir = root / split / "images"
        if not img_dir.exists():
            continue
        for img in img_dir.glob("*"):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            try:
                seen_hashes.add(file_hash(img))
                n += 1
            except OSError as e:
                print(f"  warn: could not read {img}: {e}")
    return n


def copy_split(src_split: str, dst_split: str) -> tuple[int, int, int]:
    """
    Copy unique images + matching labels from SRC/<src_split> to DST/<dst_split>.
    Returns (copied, duplicates_skipped, missing_label_skipped).
    """
    src_img = SRC / src_split / "images"
    src_lbl = SRC / src_split / "labels"
    dst_img = DST / dst_split / "images"
    dst_lbl = DST / dst_split / "labels"

    if not src_img.exists():
        print(f"  (skip) source images dir missing: {src_img}")
        return 0, 0, 0

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    copied, dup_skipped, missing_label = 0, 0, 0

    for img_path in src_img.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        h = file_hash(img_path)
        if h in seen_hashes:
            dup_skipped += 1
            continue

        # Each image must have a matching YOLO label file (same stem, .txt).
        # Skipping unlabeled images keeps dataset/ clean for training.
        label_path = src_lbl / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_label += 1
            continue

        seen_hashes.add(h)
        shutil.copy2(img_path, dst_img / img_path.name)
        shutil.copy2(label_path, dst_lbl / label_path.name)
        copied += 1

    print(
        f"  {src_split:>5} -> {dst_split:<5} | "
        f"copied={copied}  dup_skipped={dup_skipped}  no_label={missing_label}"
    )
    return copied, dup_skipped, missing_label


# ---------- Main ----------
def main() -> None:
    print("Pre-seeding hashes from existing bd3_dataset...")
    preseeded = preload_hashes(BD3)
    print(f"  pre-seeded {preseeded} hashes\n")

    print("Copying merged_dataset -> dataset/ (deduped)...")
    t_copied, t_dup, t_no_lbl = 0, 0, 0
    for src_split, dst_split in [("train", "train"), ("valid", "val")]:
        c, d, m = copy_split(src_split, dst_split)
        t_copied += c
        t_dup += d
        t_no_lbl += m

    print("\n--- Summary ---")
    print(f"Pre-seeded hashes (bd3_dataset): {preseeded}")
    print(f"Copied images:                   {t_copied}")
    print(f"Skipped duplicates:              {t_dup}")
    print(f"Skipped (no matching label):     {t_no_lbl}")
    print("DONE")


if __name__ == "__main__":
    main()
