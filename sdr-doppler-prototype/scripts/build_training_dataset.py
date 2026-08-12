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
     passes through a directory named "1"/"positive"/"pos"/"candidate",
     it's labelled 1; "0"/"negative"/"neg"/"noise" -> labelled 0.
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
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (  # noqa: E402
    DEFAULT_CENTER_FREQ_HZ,
    DEFAULT_NOVERLAP,
    DEFAULT_NPERSEG,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SNR_THRESHOLD_DB,
)
from features.dataset_io import append_row, existing_source_files, feature_row_values, prompt_for_label  # noqa: E402
from features.extractor import extract_features  # noqa: E402
from load_data import load_input  # noqa: E402
from spectrogram import iq_to_spectrogram, matrix_to_spectrogram  # noqa: E402
from storage import safe_stem, utc_timestamp  # noqa: E402

DEFAULT_EXTENSIONS = ".bin,.iq,.dat,.npy"
_POSITIVE_DIR_NAMES = {"1", "positive", "pos", "candidate"}
_NEGATIVE_DIR_NAMES = {"0", "negative", "neg", "noise"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-build a training CSV from a folder of raw capture files.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing raw capture files")
    parser.add_argument("--dataset", type=Path, default=Path("data/training/features.csv"), help="Training CSV to append to")
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS, help=f"Comma-separated file extensions to treat as captures (default: {DEFAULT_EXTENSIONS})")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories under --input-dir")
    parser.add_argument("--labels-csv", type=Path, default=None, help="Manifest CSV with columns: filename,label")
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


def resolve_label(path: Path, input_dir: Path, labels_map: Dict[str, int], forced_label: Optional[int]) -> Optional[int]:
    """Priority: manifest > folder-name convention > forced --label > (caller decides interactive/skip)."""
    rel_path = path.relative_to(input_dir)
    rel_key = str(rel_path)
    if rel_key in labels_map:
        return labels_map[rel_key]
    if path.name in labels_map:
        return labels_map[path.name]

    for part in rel_path.parts[:-1]:
        lowered = part.lower()
        if lowered in _POSITIVE_DIR_NAMES:
            return 1
        if lowered in _NEGATIVE_DIR_NAMES:
            return 0

    if forced_label is not None:
        return forced_label

    return None


def iter_capture_files(input_dir: Path, extensions: set, recursive: bool):
    walker = input_dir.rglob("*") if recursive else input_dir.glob("*")
    for path in sorted(walker):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


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
    return {
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


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"FAILED: --input-dir is not a directory: {input_dir}", file=sys.stderr)
        return 1

    extensions = parse_extensions(args.extensions)
    labels_map = load_labels_manifest(args.labels_csv)
    already_processed = set() if args.reprocess else existing_source_files(args.dataset)

    files = list(iter_capture_files(input_dir, extensions, args.recursive))
    if not files:
        print(f"No files with extensions {sorted(extensions)} found under {input_dir}")
        return 0

    n_appended = n_skipped_duplicate = n_skipped_unlabeled = n_failed = 0

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}")
        source_key = str(path)
        if source_key in already_processed:
            print("  skipped (already in dataset)")
            n_skipped_duplicate += 1
            continue

        label = resolve_label(path, input_dir, labels_map, args.label)
        if label is None:
            if args.skip_unlabeled:
                print("  skipped (no label resolved from manifest/folder/--label)")
                n_skipped_unlabeled += 1
                continue
            label = prompt_for_label(context=str(path))
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

        append_row(args.dataset, row)
        already_processed.add(source_key)
        n_appended += 1
        print(f"  appended capture_id={row['capture_id']} label={label}")

    print()
    print(f"Done: {n_appended} appended, {n_skipped_duplicate} already present, {n_skipped_unlabeled} unlabeled, {n_failed} failed")
    print(f"Dataset: {args.dataset}")
    if n_appended:
        print(f"Train with: python train_model.py --dataset {args.dataset} --output models/random_forest.joblib")
    return 1 if n_failed and n_appended == 0 else 0


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
