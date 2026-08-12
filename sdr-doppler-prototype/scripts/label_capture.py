#!/usr/bin/env python3
"""Run the detection pipeline on one capture and append a human-labelled
row to a real training dataset CSV.

This is the low-friction path to building `data/training/features.csv`
from real RTL-SDR captures (see data/training/README.md "Collecting Real
Labelled Data"): it runs the exact same feature extraction the rest of
the pipeline uses, shows you the rule detector's (and, if available, the
ML detector's) opinion plus the raw feature values, and appends your
label to the dataset - always with `is_synthetic=0`, since this script is
for real data.

Example:
    python scripts/label_capture.py --input data/raw/sample.npy \\
        --dataset data/training/features.csv --save-image --output data/results/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (  # noqa: E402
    DEFAULT_CENTER_FREQ_HZ,
    DEFAULT_MAX_SMOOTHNESS_HZ,
    DEFAULT_ML_MODEL_PATH,
    DEFAULT_MIN_DRIFT_HZ,
    DEFAULT_MIN_VALID_RATIO,
    DEFAULT_NOVERLAP,
    DEFAULT_NPERSEG,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SNR_THRESHOLD_DB,
)
from detect import detect_candidate  # noqa: E402
from detection.ml_detector import run_ml_detection  # noqa: E402
from features.dataset_io import append_row, prompt_for_label  # noqa: E402
from features.extractor import extract_features  # noqa: E402
from load_data import load_input  # noqa: E402
from spectrogram import iq_to_spectrogram, matrix_to_spectrogram, save_spectrogram_image  # noqa: E402
from storage import safe_stem, utc_timestamp  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract features from a capture and append a labelled row to a training CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Input IQ file or spectrogram matrix")
    parser.add_argument("--dataset", type=Path, default=Path("data/training/features.csv"), help="Training CSV to append to")
    parser.add_argument("--output", type=Path, default=Path("data/results"), help="Directory for the spectrogram image, if --save-image is used")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--center-freq", type=float, default=DEFAULT_CENTER_FREQ_HZ)
    parser.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG)
    parser.add_argument("--noverlap", type=int, default=DEFAULT_NOVERLAP)
    parser.add_argument("--binary-dtype", default="complex64")
    parser.add_argument("--snr-threshold-db", type=float, default=DEFAULT_SNR_THRESHOLD_DB)
    parser.add_argument("--min-valid-ratio", type=float, default=DEFAULT_MIN_VALID_RATIO)
    parser.add_argument("--min-drift-hz", type=float, default=DEFAULT_MIN_DRIFT_HZ)
    parser.add_argument("--max-smoothness-hz", type=float, default=DEFAULT_MAX_SMOOTHNESS_HZ)
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--ml-model", type=Path, default=DEFAULT_ML_MODEL_PATH)
    parser.add_argument("--no-ml", action="store_true", help="Skip showing the ML detector's opinion")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None, help="Provide the label directly instead of being prompted (1=satellite candidate, 0=not)")
    parser.add_argument("--notes", default="", help="Free-text note stored with this row")
    return parser


def run(args: argparse.Namespace) -> int:
    loaded = load_input(args.input, binary_dtype=args.binary_dtype)
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
    detection = detect_candidate(
        spec,
        min_valid_ratio=args.min_valid_ratio,
        min_drift_hz=args.min_drift_hz,
        max_smoothness_hz=args.max_smoothness_hz,
        snr_threshold_db=args.snr_threshold_db,
        features=features,
    )
    ml_detection = None if args.no_ml else run_ml_detection(args.ml_model, features)

    timestamp = utc_timestamp()
    image_path = None
    if args.save_image:
        image_path = save_spectrogram_image(
            spec, args.output / f"{safe_stem(args.input)}_{timestamp.replace(':', '')}_spectrogram.png"
        )

    print(f"Input: {args.input}")
    print(f"Rule detector:  detected={detection.detected}  confidence={detection.confidence_score:.3f}")
    if ml_detection is not None:
        if ml_detection.available:
            print(f"ML detector:    detected={ml_detection.ml_detection_result}  confidence={ml_detection.ml_confidence_score:.3f}")
        else:
            print(f"ML detector:    {ml_detection.status} ({ml_detection.reason})")
    print("Extracted features:")
    for name, value in features.as_dict().items():
        print(f"  {name}: {value}")
    if features.time_axis_is_synthetic:
        print("  NOTE: input had no real time axis - signal_duration_seconds/drift_rate_hz_per_second are in time-bin units, not seconds.")
    if image_path:
        print(f"Spectrogram image: {image_path}")

    label = args.label if args.label is not None else prompt_for_label()
    if label is None:
        print("Skipped - no row appended.")
        return 0

    capture_id = f"{safe_stem(args.input)}_{timestamp.replace(':', '')}"
    row = {
        "capture_id": capture_id,
        **features.as_dict(),
        "label": label,
        "is_synthetic": 0,
        "source_file": str(args.input),
        "sample_rate_hz": args.sample_rate,
        "nperseg": args.nperseg,
        "noverlap": args.noverlap,
        "notes": args.notes,
    }
    append_row(args.dataset, row)
    print(f"Appended capture_id={capture_id} label={label} to {args.dataset}")
    return 0


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
