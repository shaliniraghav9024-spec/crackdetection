#!/usr/bin/env bash
# deploy_vastai.sh — ship the BD3 package to a vast.ai instance and start training.
#
# Workflow this assumes:
#   1. You provisioned a vast.ai instance (image: pytorch/pytorch or similar
#      CUDA-ready Ubuntu image with python3 + pip available).
#   2. You have its SSH endpoint, e.g.  ssh -p 12345 root@ssh4.vast.ai
#   3. You ran ./pack_dataset.sh so bd3_cloud_pkg.tar exists.
#
# Usage:
#   ./deploy_vastai.sh <user@host> <port> [tarball]
#   ./deploy_vastai.sh root@ssh4.vast.ai 12345
#   ./deploy_vastai.sh root@ssh4.vast.ai 12345 /tmp/bd3_v2.tar
#
# What it does:
#   - rsync the tarball to /workspace/bd3_pkg.tar on the remote
#   - copy vastai_remote_train.sh to /workspace/run.sh
#   - launch training inside `tmux` so SSH disconnects don't kill it
#   - print the command to attach to the tmux session for live logs
#
# After training, pull results back with:
#   rsync -av -e "ssh -p PORT" USER@HOST:/workspace/runs/ ./runs/

set -euo pipefail

SSH_HOST="${1:-}"
SSH_PORT="${2:-22}"
TAR="${3:-$(dirname "$(realpath "$0")")/../bd3_cloud_pkg.tar}"

if [[ -z "$SSH_HOST" ]]; then
  echo "usage: $0 <user@host> <port> [tarball]" >&2
  exit 2
fi
if [[ ! -f "$TAR" ]]; then
  echo "error: tarball not found: $TAR" >&2
  echo "       run ./pack_dataset.sh first." >&2
  exit 1
fi

REMOTE_TAR="/workspace/bd3_pkg.tar"
REMOTE_DIR="/workspace/bd3"
REMOTE_RUN="/workspace/run.sh"

SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new $SSH_HOST"
RSYNC_E="ssh -p $SSH_PORT"

echo "[1/5] testing SSH connection ..."
$SSH "echo ok && nvidia-smi --query-gpu=name,memory.total --format=csv | head -3"

echo
echo "[2/5] rsyncing $TAR  ->  $SSH_HOST:$REMOTE_TAR"
echo "       size: $(du -h "$TAR" | awk '{print $1}')"
rsync -avh --progress -e "$RSYNC_E" "$TAR" "$SSH_HOST:$REMOTE_TAR"

echo
echo "[3/5] uploading remote run script"
LOCAL_RUN="$(dirname "$(realpath "$0")")/vastai_remote_train.sh"
if [[ ! -f "$LOCAL_RUN" ]]; then
  echo "error: $LOCAL_RUN missing — keep it next to this script." >&2
  exit 1
fi
rsync -av -e "$RSYNC_E" "$LOCAL_RUN" "$SSH_HOST:$REMOTE_RUN"
$SSH "chmod +x $REMOTE_RUN"

echo
echo "[4/5] extracting + installing on remote"
$SSH "set -e; \
      mkdir -p $REMOTE_DIR && \
      tar -xf $REMOTE_TAR -C $REMOTE_DIR && \
      pip install -q --upgrade ultralytics albumentations pyyaml && \
      ls $REMOTE_DIR | head"

echo
echo "[5/5] launching training inside tmux"
$SSH "command -v tmux >/dev/null || apt-get install -y tmux"
$SSH "tmux new-session -d -s bd3 'cd $REMOTE_DIR && $REMOTE_RUN 2>&1 | tee /workspace/train.log'"

cat <<EOF

================================================================
Launched.

  Tail the log without attaching:
    $SSH 'tail -f /workspace/train.log'

  Attach interactively (Ctrl-b d to detach):
    ssh -p $SSH_PORT $SSH_HOST -t tmux attach -t bd3

  When done, pull results back:
    rsync -avh --progress -e "ssh -p $SSH_PORT" \\
        $SSH_HOST:$REMOTE_DIR/runs/ ./runs_vastai/
================================================================
EOF
