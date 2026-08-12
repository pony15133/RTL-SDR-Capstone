#!/usr/bin/env python3
"""Batch-build (or extend) a training CSV from a folder of raw capture files.

This is the batch counterpart to scripts/label_capture.py: instead of one
file per command with an interactive label prompt, point this at a
directory of real captures (.bin/.iq/.dat/.npy by default) and it appends
a labelled feature row per file to the training CSV - using the exact
same feature extraction (features.extractor.extract_features) as the rest
of the pipeline, so the resulting CSV trains with train_model.py exactly
as before. Reading the file and extracting features is fully automatic;
the label itself is not - it has to come from a human decision somewhere,
since this is supervised learning.

Label sources, checked in this order for each file:
  1. --labels-csv: a manifest CSV with columns "filename,label". filename
     is matched first against the file's path relative to --input-dir,
     then against its bare basename.
  2. Folder-name convention: if the file's path (relative to --input-dir)
     passes through a directory named for a class in DEFAULT_CLASS_MAP
     (e.g. "satellite"/"1"/"positive"/"pos"/"candidate" -> 1;
     "noise"/"0"/"negative"/"neg" -> 0), it's labelled accordingly. Add or
     override folder-name -> label mappings with --class-map instead of
     editing this file, e.g. --class-map "interference=0,unknown_sat=1".
  3. --label: apply this single label to every file processed in this
     run (use when an entire folder is confirmed one way, e.g. a folder
     of known non-satellite background noise).
  4. Interactive prompt (same wording as label_capture.py), unless
     --skip-unlabeled is given, in which case unresolved files are
     skipped rather than prompted for.

Files already present in the dataset (matched by their full source path)
are skipped by default, so re-running this against a growing folder of
captures is safe. Pass --reprocess to add them again anyway. A capture
that fails to load/process (corrupt file, wrong dtype, ...) is logged and
skipped - it never aborts the rest of the batch.

One row per input file. A raw-IQ recording is not automatically split
into multiple rows here - see scripts/chunk_bin_to_dataset.py if a
recording's real signal only occupies a small fraction of a much longer
capture (that tool windows a single file into many scored rows instead).

Example - the "satellite/" and "noise/" folder convention out of the box:
    python scripts/build_training_dataset.py \\
        --input-dir training_data --recursive \\
        --dataset data/training/features.csv \\
        --sample-rate 2400000 --center-freq 137900000 --binary-dtype complex64

Example - a folder of confirmed-positive real satellite-pass captures:
    python scripts/build_training_dataset.py \\
        --input-dir /path/to/captures/positive_passes \\
        --label 1 \\
        --dataset data/training/features.csv \\
        --sample-rate 2400000 --center-freq 137900000 --binary-dtype complex64

Example - a folder labelled via a manifest CSV (filename,label):
    python scripts/build_training_dataset.py \\
        --input-dir /path/to/captures \\
        --labels-csv data/training/my_labels.csv \\
        --dataset data/training/features.csv

Then train exactly as before:
    python train_model.py --dataset data/training/features.csv --output models/random_forest.joblib
"""

import argparse
import csv
import sys
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (  # noqa: E402
    DEFAULT_CENTER_FREQ_HZ,
    DEFAULT_NOVERLAP,
    DEFAULT_NPERSEG,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SNR_THRESHOLD_DB,
)
from features.dataset_io import (  # noqa: E402
    DatasetSchemaError,
    append_row,
    existing_source_files,
    feature_row_values,
    prompt_for_label,
)
from features.extractor import FEATURE_NAMES, extract_features  # noqa: E402
from load_data import load_input  # noqa: E402
from spectrogram import iq_to_spectrogram, matrix_to_spectrogram  # noqa: E402
from storage import safe_stem, utc_timestamp  # noqa: E402

DEFAULT_EXTENSIONS = ".bin,.iq,.dat,.npy"

#: Default folder-name -> label convention. Not exhaustive by design - a
#: project's own class-folder names (e.g. per-satellite subfolders) don't
#: need to be added here; use --class-map instead of hard-wiring more
#: names into this module.
DEFAULT_CLASS_MAP: Dict[str, int] = {
    "1": 1, "positive": 1, "pos": 1, "candidate": 1, "satellite": 1,
    "0": 0, "negative": 0, "neg": 0, "noise": 0,
}

#: If the larger class outnumbers the smaller one by at least this ratio
#: (in the resulting dataset CSV), warn about it. Not a hard limit -
#: RandomForestClassifier(class_weight="balanced") and other techniques
#: can compensate - just something a user should notice and decide about.
IMBALANCE_WARNING_RATIO = 3.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-build a training CSV from a folder of raw capture files.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing raw capture files")
    parser.add_argument("--dataset", type=Path, default=Path("data/training/features.csv"), help="Training CSV to append to")
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS, help=f"Comma-separated file extensions to treat as captures (default: {DEFAULT_EXTENSIONS})")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories under --input-dir")
    parser.add_argument("--labels-csv", type=Path, default=None, help="Manifest CSV with columns: filename,label")
    parser.add_argument(
        "--class-map",
        default=None,
        help=(
            "Comma-separated folder-name=label pairs, added to (or overriding) the default "
            f"folder-name convention ({', '.join(f'{k}={v}' for k, v in DEFAULT_CLASS_MAP.items())}). "
            "Example: --class-map interference=0,unknown_sat=1"
        ),
    )
    parser.add_argument("--label", type=int, choices=[0, 1], default=None, help="Apply this single label to every file in this run")
    parser.add_argument("--skip-unlabeled", action="store_true", help="Skip files with no resolvable label instead of prompting interactively")
    parser.add_argument("--reprocess", action="store_true", help="Reprocess files even if already present in --dataset (by source path)")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--center-freq", type=float, default=DEFAULT_CENTER_FREQ_HZ)
    parser.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG)
    parser.add_argument("--noverlap", type=int, default=DEFAULT_NOVERLAP)
    parser.add_argument("--binary-dtype", default="complex64", help="dtype for raw binary IQ files (.bin/.iq/.dat)")
    parser.add_argument("--snr-threshold-db", type=float, default=DEFAULT_SNR_THRESHOLD_DB)
    parser.add_argument("--notes", default="", help="Free-text note stored with every row appended in this run")
    return parser


def parse_extensions(raw: str) -> set:
    return {ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}" for ext in raw.split(",") if ext.strip()}


def parse_class_map(raw: Optional[str]) -> Dict[str, int]:
    """Parse --class-map into a {folder_name: label} dict layered on DEFAULT_CLASS_MAP.

    Entries are merged onto (not replacing) the built-in convention, so
    --class-map only needs to name the classes it adds or overrides.
    """
    class_map = dict(DEFAULT_CLASS_MAP)
    if not raw:
        return class_map
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise SystemExit(f"Invalid --class-map entry {entry!r}; expected folder_name=label")
        name, _, raw_label = entry.partition("=")
        name = name.strip().lower()
        try:
            label = int(raw_label.strip())
        except ValueError:
            raise SystemExit(f"Invalid label {raw_label!r} for {name!r} in --class-map")
        if label not in (0, 1):
            raise SystemExit(f"Label must be 0 or 1, got {label} for {name!r} in --class-map")
        if not name:
            raise SystemExit(f"Empty folder name in --class-map entry {entry!r}")
        class_map[name] = label
    return class_map


def load_labels_manifest(path: Optional[Path]) -> Dict[str, int]:
    """Load a {filename_or_relpath: 0/1} map from a manifest CSV."""
    if path is None:
        return {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "filename" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit(f"Labels manifest {path} must have 'filename' and 'label' columns")
        labels: Dict[str, int] = {}
        for row in reader:
            raw_label = row["label"]
            try:
                label = int(raw_label)
            except (TypeError, ValueError):
                raise SystemExit(f"Invalid label {raw_label!r} for {row['filename']!r} in {path}")
            if label not in (0, 1):
                raise SystemExit(f"Label must be 0 or 1, got {label} for {row['filename']!r} in {path}")
            labels[row["filename"]] = label
        return labels


def resolve_label_with_source(
    path: Path,
    input_dir: Path,
    labels_map: Dict[str, int],
    forced_label: Optional[int],
    class_map: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[int], Optional[str]]:
    """Resolve a file's label plus where it came from ("manifest", "folder:<name>", "forced", or None).

    Priority: manifest > folder-name convention (class_map, default
    DEFAULT_CLASS_MAP) > forced --label > (caller decides interactive/skip).
    The source string is used only for reporting (e.g. the "by source
    folder" class-distribution breakdown) - it never changes which label
    is chosen.
    """
    class_map = DEFAULT_CLASS_MAP if class_map is None else class_map
    rel_path = path.relative_to(input_dir)
    rel_key = str(rel_path)
    if rel_key in labels_map:
        return labels_map[rel_key], "manifest"
    if path.name in labels_map:
        return labels_map[path.name], "manifest"

    for part in rel_path.parts[:-1]:
        lowered = part.lower()
        if lowered in class_map:
            return class_map[lowered], f"folder:{lowered}"

    if forced_label is not None:
        return forced_label, "forced"

    return None, None


def resolve_label(
    path: Path,
    input_dir: Path,
    labels_map: Dict[str, int],
    forced_label: Optional[int],
    class_map: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    """Backward-compatible wrapper around resolve_label_with_source() that drops the source."""
    label, _source = resolve_label_with_source(path, input_dir, labels_map, forced_label, class_map)
    return label


def iter_capture_files(input_dir: Path, extensions: set, recursive: bool):
    walker = input_dir.rglob("*") if recursive else input_dir.glob("*")
    for path in sorted(walker):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def _validate_row(row: dict) -> None:
    """Defence-in-depth check that a row is safe to append/train on.

    feature_row_values() (via feature_vector_to_array()) already sanitises
    non-finite values before this point, and CSV_COLUMNS is built from
    FEATURE_NAMES, so this should never actually fire - it exists so a
    future change to either can't silently write a broken row.
    """
    missing = [name for name in FEATURE_NAMES if name not in row]
    if missing:
        raise ValueError(f"row is missing required feature column(s): {missing}")
    non_finite = [name for name in FEATURE_NAMES if not isfinite(float(row[name]))]
    if non_finite:
        raise ValueError(f"row has non-finite value(s) for: {non_finite}")
    if row.get("label") not in (0, 1):
        raise ValueError(f"row has invalid label: {row.get('label')!r}")


def extract_row_for_file(path: Path, args: argparse.Namespace, label: int) -> dict:
    loaded = load_input(path, binary_dtype=args.binary_dtype)
    time_axis_is_synthetic = loaded.kind != "iq"
    if loaded.kind == "iq":
        spec = iq_to_spectrogram(
            loaded.values,
            sample_rate_hz=args.sample_rate,
            center_freq_hz=args.center_freq,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
        )
    else:
        spec = matrix_to_spectrogram(loaded.values, sample_rate_hz=args.sample_rate, center_freq_hz=args.center_freq)

    features = extract_features(spec, snr_threshold_db=args.snr_threshold_db, time_axis_is_synthetic=time_axis_is_synthetic)
    timestamp = utc_timestamp()
    capture_id = f"{safe_stem(path)}_{timestamp.replace(':', '')}"
    row = {
        "capture_id": capture_id,
        **feature_row_values(features),
        "label": label,
        "is_synthetic": 0,
        "source_file": str(path),
        "sample_rate_hz": args.sample_rate,
        "nperseg": args.nperseg,
        "noverlap": args.noverlap,
        "notes": args.notes,
    }
    _validate_row(row)
    return row


def dataset_label_counts(csv_path: Path) -> Counter:
    """{label_value: row_count} across the *entire* dataset CSV (not just this run).

    Used for the end-of-run single-class/imbalance warnings, since what
    matters for training is the composition of the whole file, not only
    the rows this invocation happened to add.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return Counter()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            return Counter()
        counts: Counter = Counter()
        for row in reader:
            try:
                counts[int(row["label"])] += 1
            except (TypeError, ValueError):
                continue  # malformed row - not this function's job to validate schema
        return counts


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"FAILED: --input-dir is not a directory: {input_dir}", file=sys.stderr)
        return 1

    extensions = parse_extensions(args.extensions)
    labels_map = load_labels_manifest(args.labels_csv)
    class_map = parse_class_map(args.class_map)
    already_processed = set() if args.reprocess else existing_source_files(args.dataset)

    files = list(iter_capture_files(input_dir, extensions, args.recursive))
    if not files:
        print(f"No files with extensions {sorted(extensions)} found under {input_dir}")
        return 0

    n_appended = n_skipped_duplicate = n_skipped_unlabeled = n_failed = 0
    label_counts_this_run: Counter = Counter()
    folder_counts_this_run: Counter = Counter()

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}")
        source_key = str(path)
        if source_key in already_processed:
            print("  skipped (already in dataset)")
            n_skipped_duplicate += 1
            continue

        label, source = resolve_label_with_source(path, input_dir, labels_map, args.label, class_map)
        if label is None:
            if args.skip_unlabeled:
                print("  skipped (no label resolved from manifest/folder/--label)")
                n_skipped_unlabeled += 1
                continue
            label = prompt_for_label(context=str(path))
            source = "interactive"
            if label is None:
                print("  skipped (no label given)")
                n_skipped_unlabeled += 1
                continue

        try:
            row = extract_row_for_file(path, args, label)
        except Exception as exc:  # a corrupt/unreadable capture must not abort the whole batch
            print(f"  FAILED to process ({exc}) - skipping", file=sys.stderr)
            n_failed += 1
            continue

        try:
            append_row(args.dataset, row)
        except DatasetSchemaError as exc:
            # A schema mismatch is a dataset-level problem, not a per-file
            # one - every subsequent append would fail identically, so
            # abort the batch instead of grinding through the rest.
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1

        already_processed.add(source_key)
        n_appended += 1
        label_counts_this_run[label] += 1
        if source and source.startswith("folder:"):
            folder_counts_this_run[source.split(":", 1)[1]] += 1
        print(f"  appended capture_id={row['capture_id']} label={label}")

    print()
    print(f"Processed: {len(files)}")
    print(f"Successful: {n_appended}")
    print(f"Failed: {n_failed}")
    if n_skipped_duplicate:
        print(f"Skipped (already in dataset): {n_skipped_duplicate}")
    if n_skipped_unlabeled:
        print(f"Skipped (no label resolved): {n_skipped_unlabeled}")

    if label_counts_this_run:
        print()
        print("Class distribution (this run, by label):")
        for label_value in sorted(label_counts_this_run):
            print(f"  {label_value}: {label_counts_this_run[label_value]}")
        if folder_counts_this_run:
            print()
            print("Class distribution (this run, by source folder):")
            for name in sorted(folder_counts_this_run):
                print(f"  {name}: {folder_counts_this_run[name]}")

    full_counts = dataset_label_counts(args.dataset)
    if len(full_counts) == 1:
        (only_label,) = full_counts.keys()
        print()
        print(
            f"WARNING: {args.dataset} currently contains only one class (label={only_label}, "
            f"{full_counts[only_label]} row(s)). A binary classifier needs examples of both "
            "classes to train meaningfully."
        )
    elif len(full_counts) >= 2:
        smallest = min(full_counts.values())
        largest = max(full_counts.values())
        if smallest > 0 and (largest / smallest) >= IMBALANCE_WARNING_RATIO:
            print()
            print(
                f"WARNING: {args.dataset} is imbalanced ({dict(sorted(full_counts.items()))}, "
                f"{largest / smallest:.1f}:1). Consider collecting more of the minority class, "
                "or pass --class-weight balanced to train_model.py."
            )

    print()
    print(f"Output: {args.dataset}")
    if n_appended:
        print(f"Train with: python train_model.py --dataset {args.dataset} --output models/random_forest.joblib")
    return 1 if n_failed and n_appended == 0 else 0


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
