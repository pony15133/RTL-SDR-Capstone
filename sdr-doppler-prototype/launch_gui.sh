#!/usr/bin/env sh
# macOS / Linux launcher for the desktop GUI (Windows: launch_gui.bat).
# Uses the repo's .venv if present, otherwise python3 on PATH.
cd "$(dirname "$0")" || exit 1
for py in ../.venv/bin/python .venv/bin/python python3 python; do
  if command -v "$py" >/dev/null 2>&1; then
    exec "$py" gui_app.py "$@"
  fi
done
echo "Python 3 not found. Install it (macOS: brew install python-tk; Linux: sudo apt install python3 python3-tk)." >&2
exit 1
