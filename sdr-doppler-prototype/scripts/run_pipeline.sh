#!/usr/bin/env bash
set -euo pipefail

python src/main.py --input data/raw/sample.npy --output data/results/ --save-image
