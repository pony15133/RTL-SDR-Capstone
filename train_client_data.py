#!/usr/bin/env python3
"""Train both models on the client's labelled pass recordings.

    python train_client_data.py                       # finds ..\\..\\client_pass_recordings next to CODE
    python train_client_data.py --root E:\\CAPSTONE\\CAP2\\client_pass_recordings

Steps (all results are also added to sdr-doppler-prototype/models/ml_history.csv):
  1. Build the datasets (scripts/build_client_spectrogram_set.py) - a few minutes, cached after.
  2. Score the CURRENT models on the client passes they have never seen (baseline).
  3. Train new models with the client data added, saved beside the old ones:
       models/iq_rf_client.joblib        (RSP-03 + client snapshots [+ your own passes])
       models/waterfall_rf_client.joblib (SatNOGS + client windows [+ your own passes])
  4. Score the new models on the same held-out client passes.
The working models are NOT replaced. If a new one is better, run with --promote iq|waterfall|both.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PROTO = REPO / "sdr-doppler-prototype"
TRAIN = PROTO / "data" / "training"
MODELS = PROTO / "models"
PY = sys.executable

PAIRS = {"iq": ("random_forest", "iq_rf_client"), "waterfall": ("waterfall_rf", "waterfall_rf_client")}


def run(cmd) -> int:
    cmd = [str(c) for c in cmd]
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=PROTO)


def evaluate(model: str, dataset: Path, notes: str) -> None:
    if (MODELS / f"{model}.joblib").exists() and dataset.exists():
        run([PY, "scripts/evaluate_model.py", "--model", f"models/{model}.joblib", "--dataset", dataset, "--notes", notes])


def promote(which: str = "both") -> int:
    pairs = [PAIRS[k] for k in (("iq", "waterfall") if which == "both" else (which,))]
    for old, new in pairs:
        if not (MODELS / f"{new}.joblib").exists():
            print(f"FAILED: {new}.joblib not found - train first")
            return 1
    for old, new in pairs:
        for ext in (".joblib", ".json", "_feature_importance.csv", "_splits.csv"):
            src, dst = MODELS / f"{new}{ext}", MODELS / f"{old}{ext}"
            if src.exists():
                if dst.exists():
                    shutil.copy2(dst, MODELS / f"{old}_before_client{ext}")
                shutil.copy2(src, dst)
        print(f"  {new} -> {old}  (previous kept as {old}_before_client)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO.parent.parent / "client_pass_recordings",
                    help="Folder with one sub-folder per pass")
    ap.add_argument("--promote", nargs="?", const="both", choices=("iq", "waterfall", "both"),
                    help="Make the *_client model(s) the working model(s): iq, waterfall or both (default)")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args(argv)
    if args.promote:
        return promote(args.promote)

    iq_train, iq_ext = TRAIN / "client_passes_iq_train.csv", TRAIN / "client_passes_iq_external.csv"
    wf_train, wf_ext = TRAIN / "client_passes_waterfall_train.csv", TRAIN / "client_passes_waterfall_external.csv"
    problems = []

    print("\n=== 1/4 Build datasets from", args.root)
    if not args.skip_build:
        if not args.root.exists():
            print(f"FAILED: {args.root} not found - pass --root <path to client_pass_recordings>")
            return 1
        if run([PY, "scripts/build_client_spectrogram_set.py", "--root", args.root]) != 0:
            return 1

    print("\n=== 2/4 Baseline: current models on the held-out client passes")
    evaluate("random_forest", iq_ext, "baseline: current IQ model on held-out client passes")
    evaluate("waterfall_rf", wf_ext, "baseline: current waterfall model on held-out client passes")

    print("\n=== 3/4 Train with the client data added")
    station_iq = TRAIN / "station_iq_features.csv"
    iq_sets = [TRAIN / "rsp03_camras_features.csv", iq_train] + ([station_iq] if station_iq.exists() else [])
    if run([PY, "train_model.py", "--dataset", *iq_sets, "--output", "models/iq_rf_client.joblib",
            "--model-version", "rsp03_client_passes_iq_rf", "--notes", "RSP-03 + client pass snapshots"]) != 0:
        problems.append("IQ model training failed")
    satnogs, station_wf = TRAIN / "satnogs_waterfall_features.csv", TRAIN / "station_waterfall_features.csv"
    wf_sets = [satnogs, wf_train] + ([station_wf] if station_wf.exists() else [])
    if run([PY, "train_model.py", "--feature-set", "waterfall", "--dataset", *wf_sets,
            "--output", "models/waterfall_rf_client.joblib", "--model-version", "satnogs_client_passes_waterfall_rf",
            "--notes", "SatNOGS + client pass windows"]) != 0:
        problems.append("waterfall model training failed")

    print("\n=== 4/4 New models on the same held-out client passes (and RSP-03)")
    evaluate("iq_rf_client", iq_ext, "new IQ model on held-out client passes")
    evaluate("waterfall_rf_client", wf_ext, "new waterfall model on held-out client passes")
    evaluate("waterfall_rf_client", TRAIN / "rsp03_waterfall_features.csv", "new waterfall model on RSP-03")

    print("\nCompare the 'baseline' and 'new' lines above (also in models/ml_history.csv).")
    print("Look at the previews in sdr-doppler-prototype/data/client_passes/preview/ first.")
    print("If a new model is better:  python train_client_data.py --promote iq | waterfall | both")
    for p in problems:
        print("  NOTE:", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
