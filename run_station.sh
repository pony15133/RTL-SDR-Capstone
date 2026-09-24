#!/usr/bin/env sh
# Starts the station web page (background + browser) and the automatic capture loop.
# The capture loop is restarted automatically if it crashes; the computer is kept
# awake while it runs (caffeinate on macOS, systemd-inhibit on Linux). Logs: logs/station.log
# Extra options go to auto_capture.py, e.g.  ./run_station.sh --simulate   or   --hours 12
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || { echo "Run ./setup.sh first."; exit 1; }
[ -f capture_config.json ] || cp capture_config.example.json capture_config.json
.venv/bin/python dashboard.py --config capture_config.json &
DASH=$!
trap 'kill $DASH 2>/dev/null; exit 0' EXIT INT TERM
sleep 2
( command -v open >/dev/null && open http://localhost:8050 ) || ( command -v xdg-open >/dev/null && xdg-open http://localhost:8050 ) || echo "Open http://localhost:8050"
AWAKE=""
if command -v caffeinate >/dev/null; then AWAKE="caffeinate -i";
elif command -v systemd-inhibit >/dev/null; then AWAKE="systemd-inhibit --what=sleep --why=satellite-station"; fi
RESTARTS=0
while true; do
  $AWAKE .venv/bin/python auto_capture.py --config capture_config.json --forever "$@" && break
  RESTARTS=$((RESTARTS + 1))
  echo "Capture loop stopped with an error (restart $RESTARTS). Restarting in 60 s - Ctrl+C to stop."
  sleep 60
done
