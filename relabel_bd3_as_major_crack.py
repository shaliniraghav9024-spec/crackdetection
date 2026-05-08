"""
One-shot fix: rewrite class id 2 -> 1 in every bd3_*.txt label file.

Context
-------
Your dataset_old/ already contained BD3 images (prefixed `bd3_`) that were
labeled with BD3's single class id 0 ("crack"). When remap_labels.py ran, it
applied the merged_dataset mapping (0 -> 2 minor_crack) to every file
indiscriminately, so BD3 cracks ended up tagged as minor_crack.

This script restores the intended split:
    bd3_*  files: class 2 (minor_crack)  ->  class 1 (major_crack)
    everything else: untouched

It is idempotent — if a file's lines already say "1 ...", they're left alone.
Only label files are touched; images are not moved or renamed.

Safety
------
- Only files matching bd3_*.txt under dataset/{train,val}/labels are read.
- Each file is rewritten atomically (write to .tmp + os.replace).
- A backup copy of every modified file is dropped under
  dataset/.bd3_relabel_backup/<split>/<filename>.txt before the first write.
  Delete that directory once you've verified the result.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "building-defect-detection/dataset"
BACKUP_ROOT = ROOT / ".bd3_relabel_backup"

# Class id remap inside bd3_*.txt files only.
RELABEL = {"2": "1"}


def process_file(lbl: Path, split: str, stats: Counter) -> None:
    text = lbl.read_text()
    lines_in = text.splitlines()
    lines_out: list[str] = []
    file_modified = False

    for raw in lines_in:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        old_cls = parts[0]
        if old_cls in RELABEL:
            parts[0] = RELABEL[old_cls]
            stats[f"{split}_rows_relabeled_{old_cls}_to_{parts[0]}"] += 1
            file_modified = True
        lines_out.append(" ".join(parts))

    if not file_modified:
        stats[f"{split}_files_unchanged"] += 1
        return

    # Backup before first write
    backup_path = BACKUP_ROOT / split / lbl.name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(lbl, backup_path)

    tmp = lbl.with_suffix(lbl.suffix + ".tmp")
    tmp.write_text("\n".join(lines_out) + "\n")
    os.replace(tmp, lbl)
    stats[f"{split}_files_modified"] += 1


def main() -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"{ROOT} not found")

    print(f"Target: {ROOT}")
    print(f"Backups: {BACKUP_ROOT}")
    print(f"Rule: in bd3_*.txt, rewrite leading class id  {RELABEL}\n")

    stats: Counter = Counter()
    for split in ("train", "val"):
        lbl_dir = ROOT / split / "labels"
        if not lbl_dir.is_dir():
            print(f"  (skip) {lbl_dir} missing")
            continue
        files = sorted(lbl_dir.glob("bd3_*.txt"))
        print(f"  [{split}] {len(files)} bd3_*.txt label files")
        for f in files:
            process_file(f, split, stats)

    print("\n--- Stats ---")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"\nIf everything looks right, you can remove the backup with:")
    print(f"  rm -rf {BACKUP_ROOT}")


if __name__ == "__main__":
    main()
