#!/usr/bin/env sh
# 1-minute RTL-SDR hardware test. Extra options: --freq 95.0e6  --gain 40  --simulate
cd "$(dirname "$0")" && exec .venv/bin/python check_dongle.py "$@"
