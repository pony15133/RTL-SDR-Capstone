#!/usr/bin/env sh
# One-time setup on macOS / Linux: virtual environment, packages, models, environment check.
#   ./setup.sh                   quick setup
#   ./setup.sh --fetch-satnogs   also download SatNOGS training data (10-20 min)
set -e
cd "$(dirname "$0")"
if [ -x .venv/bin/python ] && ! .venv/bin/python -c "import sys" >/dev/null 2>&1; then
  echo "The existing .venv is broken - rebuilding it."; rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv || { echo "Install Python 3.10+ first (macOS: brew install python python-tk; Linux: sudo apt install python3-venv python3-tk)"; exit 1; }
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r sdr-doppler-prototype/requirements.txt -r iq-recorder/requirements.txt
exec .venv/bin/python setup_station.py "$@"
