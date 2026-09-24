"""Human review of captures: the "uncertain" queue and labelling our own passes.

A capture the model wasn't sure about (verdict "uncertain") keeps its raw IQ
in <output_dir>/uncertain/ and gets review_status "pending". A person looks
at the waterfall (web UI Review page, or scripts/label_station_captures.py)
and says "signal" or "noise":

    signal  -> the IQ moves back next to the other kept recordings
    noise   -> the retention policy runs (archive / delete / keep-all)

Either way the answer is stored on the capture row (human_label, reviewed_by,
reviewed_at) and, when the waterfall matrix exists, appended to
data/training/station_waterfall_features.csv - so every review also makes the
waterfall model better at our own station.

Any capture can be labelled this way, not only uncertain ones: a person's
label always wins over the model.
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from config import DB_PATH, TRAINING_DATA_DIR
from database import get_result, update_result
from features.waterfall_features import WATERFALL_FEATURE_NAMES, waterfall_features
from retention import POLICIES, REJECTED_DIR, UNCERTAIN_DIR

STATION_DATASET = TRAINING_DATA_DIR / "station_waterfall_features.csv"
META = ["db_id", "satellite_name", "norad_id", "scheduled_aos", "actual_recording_start", "wf_ml_confidence_score",
        "waterfall_image_path", "labelled_by", "notes"]
COLUMNS = ["capture_id", *WATERFALL_FEATURE_NAMES, "label", "is_synthetic", "recording_id", "source_file", *META]
LABEL_NAMES = {1: "signal", 0: "noise"}


# --------------------------------------------------------------------------- training rows

def matrix_path(row: dict) -> Optional[Path]:
    img = row.get("waterfall_image_path")
    if not img:
        return None
    p = Path(img).with_suffix(".npy")
    return p if p.exists() else None


def labelled_ids(csv_path: Optional[Path] = None) -> set:
    csv_path = Path(csv_path) if csv_path else STATION_DATASET
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {r["capture_id"] for r in csv.DictReader(f)}


def append(csv_path: Path, row: dict) -> None:
    csv_path = Path(csv_path)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def make_row(db_row: dict, matrix: np.ndarray, label: int, labelled_by: str, notes: str = "") -> dict:
    feats = waterfall_features(np.asarray(matrix, dtype=np.float64))
    capture_id = f"station_{db_row['id']}"
    return {
        "capture_id": capture_id, **feats, "label": int(label), "is_synthetic": 0,
        # one recording = one pass, so a pass never lands in two splits
        "recording_id": capture_id, "source_file": db_row.get("raw_iq_file_path") or db_row.get("input_file"),
        "db_id": db_row["id"], "satellite_name": db_row.get("satellite_name"), "norad_id": db_row.get("norad_id"),
        "scheduled_aos": db_row.get("scheduled_aos"), "actual_recording_start": db_row.get("actual_recording_start"),
        "wf_ml_confidence_score": db_row.get("wf_ml_confidence_score"),
        "waterfall_image_path": db_row.get("waterfall_image_path"), "labelled_by": labelled_by, "notes": notes,
    }


def add_training_row(db_row: dict, label: int, labelled_by: str, csv_path: Optional[Path] = None,
                     notes: str = "") -> bool:
    """Append this capture to the station training set. False if it has no
    saved waterfall matrix or is already in the set (a label is final)."""
    csv_path = Path(csv_path) if csv_path else STATION_DATASET
    m = matrix_path(db_row)
    if m is None or f"station_{db_row['id']}" in labelled_ids(csv_path):
        return False
    append(csv_path, make_row(db_row, np.load(m), label, labelled_by, notes))
    return True


# --------------------------------------------------------------------------- review decisions

def _move(path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    shutil.move(str(path), str(target))
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        shutil.move(str(sidecar), str(target_dir / sidecar.name))
    return target


def _kept_dir(iq: Path) -> Path:
    """Where a confirmed recording lives: the output dir (parent of uncertain/ or rejected/)."""
    return iq.parent.parent if iq.parent.name in (UNCERTAIN_DIR, REJECTED_DIR) else iq.parent


def resolve(capture_id: int, label: int, *, db_path: Optional[Path] = None, reviewer: str = "unknown",
            policy: str = "archive-negatives", add_to_training: bool = True,
            training_csv: Optional[Path] = None, notes: str = "") -> dict:
    """Record a person's verdict on one capture and act on its raw IQ.

    Returns {"capture_id", "label", "iq_action", "iq_path", "training_row_added"}.
    Never raises for file problems - the row says what happened.
    """
    if label not in (0, 1):
        raise ValueError("label must be 1 (signal) or 0 (noise)")
    if policy not in POLICIES:
        raise ValueError(f"Unknown retention policy {policy!r}")
    db_path = Path(db_path) if db_path else DB_PATH
    row = get_result(db_path, int(capture_id))
    if row is None:
        raise KeyError(f"No capture with id {capture_id}")

    iq_path = row.get("raw_iq_file_path")
    iq = Path(iq_path) if iq_path else None
    action, final = "no IQ file", iq_path
    try:
        if iq is not None and iq.exists():
            if label == 1:
                target_dir = _kept_dir(iq)
                if iq.parent != target_dir:
                    final = str(_move(iq, target_dir))
                    action = "kept (confirmed by review, moved out of " + iq.parent.name + "/)"
                else:
                    action = "kept (confirmed by review)"
            elif policy == "keep-all":
                action = "kept (policy keep-all)"
            elif policy == "archive-negatives":
                target_dir = _kept_dir(iq) / REJECTED_DIR
                final = str(_move(iq, target_dir)) if iq.parent != target_dir else str(iq)
                action = "archived (reviewed as noise)"
            else:
                iq.unlink()
                final, action = None, "deleted (reviewed as noise)"
    except OSError as exc:
        action = f"unchanged (error: {exc})"

    added = add_training_row(row, label, reviewer, training_csv, notes) if add_to_training else False
    update_result(db_path, int(capture_id), {
        "review_status": LABEL_NAMES[label],
        "human_label": label,
        "reviewed_by": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "iq_retention": action,
        "raw_iq_file_path": final,
    })
    return {"capture_id": int(capture_id), "label": label, "iq_action": action, "iq_path": final,
            "training_row_added": added}
