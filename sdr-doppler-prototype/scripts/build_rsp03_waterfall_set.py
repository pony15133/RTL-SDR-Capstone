#!/usr/bin/env python3
"""Rebuild the external test set data/training/rsp03_waterfall_features.csv.

Each labelled 1-second window of the client's CAMRAS RSP-03 recordings
(rows of rsp03_camras_features.csv, labelled by scripts/label_known_carrier.py)
is turned into the standard 128 x 128 waterfall the station pipeline uses
(doppler.standard_waterfall, +/-24 kHz) and described with the waterfall
features. Labels and recording ids are copied from the IQ dataset, so both
test sets describe exactly the same windows.

    python scripts/build_rsp03_waterfall_set.py --data-root ..      # folder that contains Data/

Needs the raw files Data/Satellite_Data_snapshots/RSP-03_*.raw (~490 MB,
git-ignored). Run it again whenever the waterfall features change.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doppler import standard_waterfall  # noqa: E402
from features.waterfall_features import WATERFALL_FEATURE_NAMES, waterfall_features  # noqa: E402

SAMPLE_RATE = 1_000_000.0  # CAMRAS RSP-03 snapshots: 1 Msps ci16


def window_iq(raw_path: Path, start: int, stop: int) -> np.ndarray:
    raw = np.memmap(raw_path, dtype="<i2", mode="r")[2 * start:2 * stop]
    return (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 32768.0


def build(iq_dataset: Path, data_root: Path, arrays_dir=None) -> pd.DataFrame:
    src = pd.read_csv(iq_dataset)
    rows = []
    for _, r in src.iterrows():
        m = re.match(r"(.*)#samples=(\d+)-(\d+)", str(r["source_file"]))
        if not m:
            raise SystemExit(f"FAILED: {r['capture_id']} has no #samples= window in source_file")
        path, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        raw_path = (data_root / path)
        if not raw_path.exists():
            raise SystemExit(f"FAILED: {raw_path} not found - point --data-root at the folder containing Data/")
        matrix = standard_waterfall(window_iq(raw_path, a, b), SAMPLE_RATE)
        if arrays_dir is not None:
            np.save(Path(arrays_dir) / f"{r['capture_id']}.npy", matrix.astype(np.float16))
        feats = waterfall_features(matrix)
        rows.append({"capture_id": r["capture_id"], **{k: feats[k] for k in WATERFALL_FEATURE_NAMES},
                     "label": int(r["label"]), "recording_id": r["recording_id"], "is_synthetic": 0,
                     "source_file": r["source_file"]})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=ROOT.parent, help="Folder that contains Data/ (default: repo root)")
    p.add_argument("--iq-dataset", type=Path, default=ROOT / "data" / "training" / "rsp03_camras_features.csv")
    p.add_argument("--output", type=Path, default=ROOT / "data" / "training" / "rsp03_waterfall_features.csv")
    p.add_argument("--arrays-dir", type=Path, default=None, help="Also save each 128x128 waterfall here (.npy)")
    args = p.parse_args(argv)
    if args.arrays_dir:
        args.arrays_dir.mkdir(parents=True, exist_ok=True)
    df = build(args.iq_dataset, args.data_root, args.arrays_dir)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} windows ({int(df.label.sum())} with signal) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
