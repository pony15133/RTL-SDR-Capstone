#!/usr/bin/env sh
# Starts the live dashboard (background + browser) and the automatic capture loop.
# Extra options go to auto_capture.py, e.g.  ./run_station.sh --simulate   or   --hours 12
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || { echo "Run ./setup.sh first."; exit 1; }
[ -f capture_config.json ] || cp capture_config.example.json capture_config.json
.venv/bin/python dashboard.py --config capture_config.json &
DASH=$!
trap 'kill $DASH 2>/dev/null' EXIT INT TERM
sleep 2
( command -v open >/dev/null && open http://localhost:8050 ) || ( command -v xdg-open >/dev/null && xdg-open http://localhost:8050 ) || echo "Open http://localhost:8050"
.venv/bin/python auto_capture.py --config capture_config.json --forever "$@"
