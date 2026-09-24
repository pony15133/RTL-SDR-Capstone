#!/usr/bin/env python3
"""Label our own station's captures and add them to the waterfall training data.

Every pass the station processes leaves a Doppler-corrected waterfall
(<name>_waterfall.png + the 128 x 128 matrix <name>_waterfall.npy) and a
database row. This tool walks through the captures that aren't labelled
yet, shows each waterfall, and asks: satellite signal visible (1), no
signal (0), skip (s) or quit (q). Labelled captures are appended to
data/training/station_waterfall_features.csv, which trains together with
the SatNOGS set:

    python scripts/label_station_captures.py                 # label new captures
    python scripts/label_station_captures.py --list          # what's there, no prompts
    python train_model.py --feature-set waterfall --output models/waterfall_rf.joblib \\
        --dataset data/training/satnogs_waterfall_features.csv data/training/station_waterfall_features.csv

Label by LOOKING at the waterfall, not by what the model said - otherwise
the model just learns its own mistakes. A trace near the centre line (after
Doppler correction it's close to vertical) = 1. Only noise, or only
interference that is clearly not the satellite = 0. Not sure = s.

This is the most valuable data you can add: it's our antenna, our dongle,
our location and our satellites, which no online dataset has.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import DB_PATH  # noqa: E402
from database import list_results  # noqa: E402
from features.waterfall_features import WATERFALL_FEATURE_NAMES  # noqa: E402,F401

from review import COLUMNS, META, append, labelled_ids, make_row, matrix_path, resolve  # noqa: E402,F401  (shared with the web UI)

DEFAULT_OUTPUT = ROOT / "data" / "training" / "station_waterfall_features.csv"


def open_image(path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606 - opening the user's own image
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # a missing viewer is not fatal
        print(f"  (could not open the image: {exc})")


def pending(db_path: Path, csv_path: Path, limit: int, include_simulated: bool) -> list:
    done = labelled_ids(csv_path)
    rows = []
    for r in list_results(db_path, limit=limit):
        if not include_simulated and r.get("simulated"):
            continue
        if f"station_{r['id']}" in done or matrix_path(r) is None:
            continue
        rows.append(r)
    return list(reversed(rows))  # oldest first


def ask(prompt: str, input_fn=input) -> str:
    while True:
        answer = input_fn(prompt).strip().lower()
        if answer in {"1", "0", "s", "q"}:
            return answer
        print("  please type 1, 0, s or q")


def main(argv=None, input_fn=input) -> int:
    p = argparse.ArgumentParser(description="Label our own captures for the waterfall model.")
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--limit", type=int, default=1000, help="Look at this many most recent captures")
    p.add_argument("--list", action="store_true", help="Only list unlabelled captures")
    p.add_argument("--no-open", action="store_true", help="Don't open each image automatically")
    p.add_argument("--include-simulated", action="store_true", help="Also offer simulated (demo) captures")
    p.add_argument("--labeller", default=os.environ.get("USERNAME") or os.environ.get("USER") or "unknown")
    p.add_argument("--policy", default="archive-negatives", choices=("keep-all", "archive-negatives", "delete-negatives"),
                   help="What to do with the raw IQ of a capture you label 0 (default archive-negatives)")
    args = p.parse_args(argv)

    todo = pending(args.db, args.output, args.limit, args.include_simulated)
    already = len(labelled_ids(args.output))
    print(f"{len(todo)} capture(s) to label ({already} already in {args.output.name})")
    if args.list or not todo:
        for r in todo:
            print(f"  #{r['id']}  {r.get('satellite_name') or '?':14s} {r.get('actual_recording_start') or r.get('timestamp_utc')}"
                  f"  model {r.get('wf_ml_confidence_score')}  {r.get('waterfall_image_path')}")
        return 0

    print("For each waterfall: 1 = satellite signal visible, 0 = no signal, s = skip, q = quit\n")
    added = {0: 0, 1: 0}
    for r in todo:
        print(f"#{r['id']}  {r.get('satellite_name') or '?'}  {r.get('actual_recording_start') or r.get('timestamp_utc')}")
        print(f"  image: {r.get('waterfall_image_path')}")
        if not args.no_open:
            open_image(r["waterfall_image_path"])
        answer = ask("  label [1/0/s/q]: ", input_fn)
        if answer == "q":
            break
        if answer == "s":
            continue
        # Same as the web UI's Review page: stores the label on the capture row, moves an
        # uncertain IQ file out of uncertain/ (signal) or applies the policy (noise),
        # and appends the training row.
        resolve(r["id"], int(answer), db_path=args.db, reviewer=args.labeller, policy=args.policy,
                training_csv=args.output)
        added[int(answer)] += 1
    print(f"\nAdded {added[1]} with signal, {added[0]} without, to {args.output}")
    print("Retrain with it:  python train_model.py --feature-set waterfall --output models/waterfall_rf.joblib "
          f"--dataset data/training/satnogs_waterfall_features.csv {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
