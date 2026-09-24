#!/usr/bin/env bash
# Overnight training (macOS/Linux): large SatNOGS download (waits through the
# hourly API limit), retrain, test on RSP-03. Keeps the computer awake while it
# runs; log in logs/overnight_<date>.txt. Ctrl+C stops; run again to resume.
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run ./setup.sh first."; exit 1; }
mkdir -p logs
LOG="logs/overnight_$(date +%Y%m%d_%H%M).txt"
echo "Log: $LOG"
export PYTHONUNBUFFERED=1
CMD=(.venv/bin/python setup_station.py --fetch-satnogs --big-data --skip-doctor "$@")
if command -v caffeinate >/dev/null; then CMD=(caffeinate -i "${CMD[@]}");
elif command -v systemd-inhibit >/dev/null; then CMD=(systemd-inhibit --what=sleep --why="SatNOGS training" "${CMD[@]}"); fi
"${CMD[@]}" 2>&1 | tee "$LOG"
echo "Finished. Log saved to $LOG"
