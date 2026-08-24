#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, "src")
from database import init_db

init_db(Path("data/results/captures.sqlite3"))
print("Initialized data/results/captures.sqlite3")
PY
