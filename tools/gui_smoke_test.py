#!/usr/bin/env python3
"""Automated start-up smoke test of the desktop GUI (sdr-doppler-prototype/gui_app.py).

    python tools/gui_smoke_test.py                 # on a desktop (Windows / macOS / Linux)
    xvfb-run -a python tools/gui_smoke_test.py     # headless Linux, virtual display

Opens the real Tk window (no stubs), visits every function tab, then presses
the same code paths as the "Run Demo Capture" and "Show History" buttons -
pointed at a temporary database and results folder, so your real data is
not touched - and waits for them to finish. Screenshots each step when the
platform allows it (Pillow ImageGrab), and writes evidence/gui/.

This checks that the GUI starts, builds every tab and drives the pipeline
without errors. It is NOT a substitute for a person clicking through it:
layout/readability, file dialogs and the results pop-up windows still need
the manual checklist in LIVE_HARDWARE_TEST.md.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP_DIR = REPO / "sdr-doppler-prototype"
OUT = REPO / "evidence" / "gui"
sys.path.insert(0, str(APP_DIR))


def screenshot(root, name: str, shots: list) -> None:
    root.update()
    time.sleep(0.3)
    root.update()
    try:
        from PIL import ImageGrab

        x, y = root.winfo_rootx(), root.winfo_rooty()
        w, h = root.winfo_width(), root.winfo_height()
        path = OUT / f"{len(shots) + 1:02d}_{name}.png"
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(path)
        shots.append(path.name)
    except Exception as exc:  # no screen grabbing on this platform/build - keep testing
        shots.append(f"(no screenshot: {exc.__class__.__name__}: {exc})")


def run_and_wait(root, app, function: str, timeout: float) -> str:
    """Start a GUI function (as its button does) and keep the REAL Tk main loop
    running until the background command reports complete/failed. gui_app's
    worker threads call root.after(), which Tk only allows while mainloop() runs."""
    outcome = {"status": "timed out"}
    deadline = time.monotonic() + timeout

    def poll():
        status = app.status_var.get()
        if status.endswith("complete") or status.endswith("failed"):
            outcome["status"] = status
            root.after(1000, root.quit)  # give the after(0, ...) output callback time to draw
        elif time.monotonic() > deadline:
            root.quit()
        else:
            root.after(200, poll)

    root.after(0, lambda: app._run_selected_function(function))
    root.after(200, poll)
    root.mainloop()
    return outcome["status"]


def main() -> int:
    from tkinter import Tk

    import gui_app

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    checks, shots = [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    with tempfile.TemporaryDirectory(prefix="sdr_gui_") as tmp:
        tmp = Path(tmp)
        root = Tk()
        app = gui_app.SDRDopplerGUI(root)
        app.db_path_var.set(str(tmp / "gui.sqlite3"))
        app.output_dir_var.set(str(tmp / "results"))
        root.update()
        check("main window opens", root.winfo_viewable(), root.title())

        tabs = [app.function_tabs.tab(t, "text") for t in app.function_tabs.tabs()]
        check("all function tabs built", len(tabs) == 7, ", ".join(tabs))
        for i, name in enumerate(tabs):
            app.function_tabs.select(i)
            screenshot(root, "tab_" + name.split(" (")[0].lower().replace(" ", "_"), shots)

        app.function_tabs.select(tabs.index("Satellite Passes"))
        status = run_and_wait(root, app, "Demo Capture", timeout=180)  # the "Run Demo Capture" button
        check("Run Demo Capture completes", status.endswith("complete"), status)
        import sqlite3

        with sqlite3.connect(tmp / "gui.sqlite3") as conn:
            rows = conn.execute("SELECT satellite_name, recording_status, processing_status FROM capture_results").fetchall()
        check("demo capture stored in the GUI's database", rows and rows[0][1:] == ("SUCCESS", "DETECTED"), rows)
        check("results pop-up window opened", any(w.winfo_class() == "Toplevel" for w in root.winfo_children()))
        screenshot(root, "after_demo_capture", shots)
        for w in root.winfo_children():
            if w.winfo_class() == "Toplevel":
                w.destroy()

        app.function_tabs.select(tabs.index("Capture History"))
        status = run_and_wait(root, app, "Capture History", timeout=60)  # the "Show History" button
        history = app.results_box.get("1.0", "end")
        check("Show History lists the demo capture", status.endswith("complete") and "METEOR-M2-3" in history, status)
        screenshot(root, "capture_history", shots)
        root.destroy()

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = None
    import tkinter

    env = {"python": platform.python_version(), "platform": platform.platform(), "tk": tkinter.TkVersion,
           "git_commit": sha, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ok = all(c["ok"] for c in checks)
    (OUT / "gui_smoke.json").write_text(json.dumps({"all_passed": ok, "checks": checks, "screenshots": shots,
                                                    "environment": env}, indent=2), encoding="utf-8")
    md = ["# GUI start-up smoke test", "",
          f"Generated {env['generated_utc']} on commit `{sha}` by `tools/gui_smoke_test.py` - Python {env['python']}, "
          f"Tk {env['tk']}, {env['platform']}.", "",
          f"**Result: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}**", "",
          "Automated: the real Tk window, every tab, and the Run Demo Capture / Show History code paths against a "
          "temporary database. Not a manual usability check.", "", "| Check | Result | Detail |", "|---|---|---|"]
    md += [f"| {c['check']} | {'PASS' if c['ok'] else 'FAIL'} | {str(c['detail'])[:150]} |" for c in checks]
    md += ["", "Screenshots:", ""] + [f"- {s}" for s in shots]
    (OUT / "GUI_SMOKE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} - wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
