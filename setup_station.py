#!/usr/bin/env python3
"""One-time station setup - run by setup.bat / setup.sh (or directly).

1. Creates capture_config.json from the example if you don't have one.
2. Trains the IQ model from the committed real dataset (CAMRAS RSP-03),
   so every computer gets a model built with its own scikit-learn version.
3. Optionally downloads SatNOGS observations and trains the waterfall
   model (--fetch-satnogs, 10-20 min, needs internet).
4. Runs the environment doctor (RTL-SDR tools, dongle, packages).

    python setup_station.py                       # quick setup
    python setup_station.py --fetch-satnogs       # + real multi-satellite training data
    python setup_station.py --fetch-satnogs --per-class 300
    python setup_station.py --fetch-satnogs --big-data     # ~3000 observations over 120 days (a few hours)
    python setup_station.py --refresh-labels               # relabel saved SatNOGS data by waterfall vetting

The waterfall model trains on every labelled set that exists:
data/training/satnogs_waterfall_features.csv and, once you've labelled
your own passes (scripts/label_station_captures.py),
data/training/station_waterfall_features.csv.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PROTO = REPO / "sdr-doppler-prototype"
PY = sys.executable


def step(title: str) -> None:
    print(f"\n=== {title}")


def run(cmd, cwd=REPO) -> int:
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], cwd=str(cwd))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Set up this computer as a capture station.")
    p.add_argument("--fetch-satnogs", action="store_true", help="Download SatNOGS training data and train the waterfall model")
    p.add_argument("--per-class", type=int, default=None,
                   help="SatNOGS observations per class (signal / no signal); default 200, or 1500 with --big-data")
    p.add_argument("--big-data", action="store_true",
                   help="With --fetch-satnogs: a large download (1500 per class over 120 days, METEOR/ISS first). "
                        "A few hours: it pauses when SatNOGS's hourly API allowance runs out and continues by "
                        "itself. Resumable - stop with Ctrl+C and run again to continue")
    p.add_argument("--refresh-labels", action="store_true",
                   help="Relabel the saved SatNOGS observations from their waterfall vetting before training")
    p.add_argument("--skip-doctor", action="store_true")
    args = p.parse_args(argv)
    problems = []

    step("1/4 Station config")
    cfg = REPO / "capture_config.json"
    if cfg.exists():
        print(f"  keeping your existing {cfg.name}")
    else:
        shutil.copy(REPO / "capture_config.example.json", cfg)
        print(f"  created {cfg.name} from the example - edit 'station' (lat/lon) if you're not at the Singapore campus")

    step("2/4 IQ model (real CAMRAS RSP-03 data)")
    dataset = PROTO / "data" / "training" / "rsp03_camras_features.csv"
    if run([PY, "train_model.py", "--dataset", dataset, "--output", "models/random_forest.joblib",
            "--model-version", "rsp03_iq_rf"], cwd=PROTO) != 0:
        problems.append("IQ model training failed (see above)")

    step("3/4 Waterfall model (SatNOGS)")
    satnogs = PROTO / "data" / "training" / "satnogs_waterfall_features.csv"
    station = PROTO / "data" / "training" / "station_waterfall_features.csv"
    fetch = [PY, "scripts/fetch_satnogs_dataset.py"]
    if satnogs.exists():
        # Older datasets lack newer feature columns: recompute from the saved arrays (seconds, no download).
        run(fetch + ["--rebuild-features"], cwd=PROTO)
    if args.refresh_labels and satnogs.exists():
        if run(fetch + ["--refresh-labels"], cwd=PROTO) != 0:
            problems.append("relabelling stopped early - run setup with --refresh-labels again to continue")
    if args.fetch_satnogs:
        per_class = args.per_class or (1500 if args.big_data else 200)
        cmd = fetch + ["--per-class", per_class]
        if args.big_data:
            cmd += ["--days-back", 120, "--preset", "station"]
        if run(cmd, cwd=PROTO) != 0:
            problems.append("SatNOGS download stopped early - rerun setup with --fetch-satnogs to resume")
    if satnogs.exists():
        datasets = [satnogs] + ([station] if station.exists() else [])
        if station.exists():
            print(f"  + your own labelled passes: {station.name}")
        if run([PY, "train_model.py", "--feature-set", "waterfall", "--dataset", *datasets,
                "--output", "models/waterfall_rf.joblib", "--model-version", "satnogs_waterfall_rf"], cwd=PROTO) != 0:
            problems.append("waterfall model training failed (see above)")
        else:
            print("\n  External test - real CAMRAS RSP-03 pass the waterfall model has never seen:")
            run([PY, "scripts/evaluate_model.py", "--model", "models/waterfall_rf.joblib",
                 "--dataset", "data/training/rsp03_waterfall_features.csv"], cwd=PROTO)
    else:
        print("  no SatNOGS dataset yet - run setup again with --fetch-satnogs to add the waterfall model.")
        print("  (Everything works without it; the IQ model and rule detector are used instead.)")

    step("4/4 Environment check")
    if not args.skip_doctor:
        code = run([PY, "-m", "rtl_recorder.doctor", "--output-dir", str(REPO / "recordings")], cwd=REPO / "iq-recorder")
        if code != 0:
            problems.append("doctor reported missing items - simulation mode still works; see its instructions")

    print("\n" + ("Setup finished." if not problems else "Setup finished with notes:\n  - " + "\n  - ".join(problems)))
    print("Next:\n  1. Plug in the dongle and run check_dongle (1 minute hardware test)\n"
          "  2. Run run_station to start the dashboard + automatic capture")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
