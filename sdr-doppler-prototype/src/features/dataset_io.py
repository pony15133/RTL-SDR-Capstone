"""Shared training-CSV I/O: the canonical row schema, append/dedup helpers,
and the interactive label prompt - used by both ``scripts/label_capture.py``
(one capture at a time) and ``scripts/build_training_dataset.py`` (batch,
from a folder of raw captures), so the row schema and append logic exist
in exactly one place.
"""

from pathlib import Path
from typing import Optional
import csv

from features.extractor import FEATURE_NAMES, FeatureVector, feature_vector_to_array

#: The training CSV schema - see data/training/README.md for the full
#: column-by-column documentation. Kept here (not duplicated) so every
#: writer produces rows train_model.py can read unmodified.
CSV_COLUMNS = [
    "capture_id", *FEATURE_NAMES, "label", "is_synthetic",
    "source_file", "sample_rate_hz", "nperseg", "noverlap", "notes",
]


def feature_row_values(features: FeatureVector) -> dict:
    """The 9 ML feature values as {name: value}, sanitised for CSV/training.

    A capture with fewer than 3 valid trace points legitimately produces
    smoothness_score = inf (see calculate_smoothness()) - correct for the
    rule detector, but literal inf written to the CSV fails
    ml/train.py's own validation (and can't be fed to RandomForest
    directly). This applies the same finite-sentinel substitution
    feature_vector_to_array() already uses at inference time, so a row
    written here is always safe to train on and matches what the model
    will actually see - use this instead of FeatureVector.as_dict() when
    building a CSV row.
    """
    array = feature_vector_to_array(features)
    return dict(zip(FEATURE_NAMES, (float(v) for v in array)))


def append_row(csv_path: Path, row: dict) -> None:
    """Append one row to the training CSV, writing the header first if needed."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_has_content = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not file_has_content:
            writer.writeheader()
        writer.writerow(row)


def existing_source_files(csv_path: Path) -> set:
    """The set of `source_file` values already present in the dataset.

    Used to skip re-processing a capture that's already been labelled and
    appended, so re-running a batch import against a growing folder of
    captures is safe by default.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "source_file" not in reader.fieldnames:
            return set()
        return {row["source_file"] for row in reader if row.get("source_file")}


def prompt_for_label(context: str = "this capture") -> Optional[int]:
    """Interactively ask for a 0/1 label (or None to skip). Never guesses."""
    while True:
        raw = input(f"Label {context} [1 = satellite candidate / 0 = not / s = skip]: ").strip().lower()
        if raw in ("1", "y", "yes"):
            return 1
        if raw in ("0", "n", "no"):
            return 0
        if raw in ("s", "skip", ""):
            return None
        print("Please enter 1, 0, or s to skip.")
