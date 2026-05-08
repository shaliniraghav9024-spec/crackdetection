#!/usr/bin/env bash
# pack_dataset.sh — tar up the project for cloud GPU training.
#
# Output: bd3_cloud_pkg.tar in the parent directory (so we don't include
# our own output file in the archive).
#
# Includes:  dataset/, data.yaml, classes.txt, training scripts
# Excludes:  venv, runs/, old dataset copies, models/, git, caches, reports
#
# JPEGs don't compress — using plain tar (no gzip) so packing is ~30s,
# not ~10min, with a near-identical size.
#
# Usage:
#   ./pack_dataset.sh                          # default name
#   ./pack_dataset.sh /tmp/bd3_v1.tar          # custom path

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$(dirname "$PROJ_DIR")/bd3_cloud_pkg.tar}"

# What to ship to the cloud machine (relative to PROJ_DIR)
INCLUDE=(
  dataset
  data.yaml
  classes.txt
  yolo11_emc.yaml
  train_yolo.py
  augment_config.py
  validate_dataset.py
  validate_and_compare.py
  tune_hyperparams.py
)

# Pretrained weights — ship if present, saves a download
for w in yolo11n.pt yolo11s.pt yolo11m.pt; do
  if [[ -f "$PROJ_DIR/$w" ]]; then INCLUDE+=("$w"); fi
done

# Sanity: every INCLUDE entry must exist
for entry in "${INCLUDE[@]}"; do
  if [[ ! -e "$PROJ_DIR/$entry" ]]; then
    echo "error: missing required entry: $entry" >&2
    exit 1
  fi
done

EXCLUDES=(
  --exclude='dataset/.bd3_relabel_backup'   # local-only backup
  --exclude='__pycache__'
  --exclude='*.pyc'
)

echo "Packing $PROJ_DIR -> $OUT"
echo "Entries:"
printf '  %s\n' "${INCLUDE[@]}"
echo

cd "$PROJ_DIR"
tar "${EXCLUDES[@]}" -cf "$OUT" "${INCLUDE[@]}"

SIZE=$(du -h "$OUT" | awk '{print $1}')
SHA=$(sha256sum "$OUT" | awk '{print $1}')

echo "Done."
echo "  path : $OUT"
echo "  size : $SIZE"
echo "  sha256: $SHA"
echo
echo "Next:"
echo "  Colab:    upload $OUT to Google Drive, open bd3_train_colab.ipynb"
echo "  vast.ai:  ./deploy_vastai.sh user@HOST:PORT $OUT"
