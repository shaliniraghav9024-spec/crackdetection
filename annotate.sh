#!/usr/bin/env bash
# Launch labelImg in YOLO format on the BD3 dataset.
#
# Usage:
#   ./annotate.sh                # opens train split (default)
#   ./annotate.sh train          # opens train split
#   ./annotate.sh val            # opens val split
#
# In labelImg:
#   1. The label format must say "YOLO" in the left toolbar (toggle with
#      the "PascalVOC" / "YOLO" button if it doesn't). This script tries
#      to launch with that already selected.
#   2. Use 'w' to draw a box, type/select the class, then 'd' for the
#      next image. Files auto-save when you press 'Ctrl+S' or move on.
#   3. Output .txt files are written next to the image by default; we
#      redirect them to ../labels/ via the second positional argument.
set -euo pipefail

cd "$(dirname "$0")"

SPLIT="${1:-train}"
case "$SPLIT" in
  train|val) ;;
  *) echo "Unknown split: $SPLIT (use train or val)"; exit 1 ;;
esac

IMAGES="$(pwd)/dataset/${SPLIT}/images"
LABELS="$(pwd)/dataset/${SPLIT}/labels"
CLASSES="$(pwd)/yolo_classes.txt"

if [[ ! -d "$IMAGES" ]]; then
  echo "Image folder not found: $IMAGES"
  echo "Run: python prepare_yolo_dataset.py"
  exit 1
fi
if [[ ! -f "$CLASSES" ]]; then
  echo "Classes file not found: $CLASSES"
  exit 1
fi

mkdir -p "$LABELS"

# shellcheck source=/dev/null
source "$(pwd)/venv/bin/activate"

# labelImg picks up the save dir from a flag we set later via UI, but we
# also pre-create labels/ so the YOLO output goes next to images and you
# can move them. Easier: just point labelImg at the labels folder via the
# 'Change Save Dir' menu (Ctrl+R) the first time you launch.
echo "Opening labelImg on:"
echo "  images : $IMAGES"
echo "  labels : $LABELS"
echo "  classes: $CLASSES"
echo
echo "Inside labelImg:"
echo "  - Make sure the format toggle on the left says YOLO (not PascalVOC)."
echo "  - Press Ctrl+R and pick:  $LABELS"
echo "  - 'w' to draw a box, 'd' next image, 'a' previous image."
echo

exec python -m labelImg.labelImg "$IMAGES" "$CLASSES" "$LABELS"
