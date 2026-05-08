#!/usr/bin/env bash
# vastai_remote_train.sh — runs INSIDE the vast.ai instance.
#
# Invoked by deploy_vastai.sh inside a tmux session at /workspace/bd3.
# Edit the args below to change run config; this is intentionally simple
# so you can `vim` it on the remote between runs without touching the
# deploy script.

set -euo pipefail
cd /workspace/bd3

# ---- Patch data.yaml to use the remote path ----
python - <<'PY'
import yaml, pathlib
p = pathlib.Path('data.yaml')
cfg = yaml.safe_load(p.read_text())
cfg['path'] = '/workspace/bd3/dataset'
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print('data.yaml path ->', cfg['path'])
PY

# ---- Quick re-validation (catches upload corruption) ----
python validate_dataset.py --root dataset --num-classes 6 --workers 8 \
    --report cloud_validate_report.json | tail -25

# ---- Configurable training args ----
MODEL="${MODEL:-yolo11s.pt}"
EPOCHS="${EPOCHS:-200}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
NAME="${NAME:-bd3_yolo11s_v1}"
PATIENCE="${PATIENCE:-30}"

echo
echo "=== Training config ==="
echo "  model=$MODEL  epochs=$EPOCHS  imgsz=$IMGSZ  batch=$BATCH  name=$NAME"
echo "  GPU(s):"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo

python train_yolo.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device 0 \
    --workers "$WORKERS" \
    --patience "$PATIENCE" \
    --save-period 25 \
    --name "$NAME"

# ---- Post-training eval ----
BEST="runs/detect/$NAME/weights/best.pt"
if [[ -f "$BEST" ]]; then
    echo
    echo "=== Final eval ==="
    python validate_and_compare.py --weights "$BEST" --imgsz "$IMGSZ" --device 0 \
        --labels "$NAME" || true
fi

echo
echo "DONE. Pull results from the deploy host with:"
echo "  rsync -avh -e \"ssh -p \$PORT\" \$USER@\$HOST:/workspace/bd3/runs/ ./runs_vastai/"
