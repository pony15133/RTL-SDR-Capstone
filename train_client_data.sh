#!/usr/bin/env bash
# Train both models on the client's labelled pass recordings.
# Usage: ./train_client_data.sh [--root /path/to/client_pass_recordings] [--promote]
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run ./setup.sh first."; exit 1; }
mkdir -p logs
LOG="logs/client_data_$(date +%Y%m%d_%H%M).txt"
echo "Log: $LOG"
export PYTHONUNBUFFERED=1
.venv/bin/python train_client_data.py "$@" 2>&1 | tee "$LOG"
echo "Finished. Log saved to $LOG"
