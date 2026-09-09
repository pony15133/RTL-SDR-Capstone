#!/usr/bin/env python3
"""Slice a large raw-IQ capture into short windows and label each one for
`data/training/features.csv`, instead of labeling the whole file as a
single example.

Built for sparse-burst data (e.g. the LoRadar satellite-LoRa dataset,
~3% packet occupancy): labeling one 100+MB file as a single row throws
away everything but one bit of information. This script instead:

1. Memory-maps the file (no need to load a multi-GB capture into RAM)
   and steps through it in fixed-length windows.
2. Extracts the same feature set the rest of the pipeline uses for each
   window.
3. Uses `valid_signal_ratio` (fraction of time slices with SNR above
   --snr-threshold-db) as a cheap activity filter: windows with no
   above-threshold energy are almost certainly silence, so they're
   auto-labeled 0 and appended without asking (tagged in `notes` as
   unreviewed) unless --no-auto-negatives is passed. Windows that show
   activity are shown to you (features + optional spectrogram image) and
   prompted for a real label - same 1/0/s convention as label_capture.py.
4. Appends every row to the same CSV schema label_capture.py uses, so
   train_model.py can consume it directly.

The scanning loop itself lives in `scan_and_label()`, which takes a
`label_fn` callback instead of calling input() directly - this CLI
(`run()`/`main()`) supplies a terminal-prompting one, and
sdr-doppler-prototype/gui_app.py supplies one that pops up a review
dialog instead, so both share the exact same windowing/feature/CSV logic.

Example (LoRadar-style capture: complex64, 4 MHz sample rate, satellite
uplink band centered around 401 MHz):

    python scripts/label_lora_chunks.py \\
        --input /path/to/session.bin \\
        --dataset data/training/features.csv \\
        --sample-rate 4000000 --center-freq 401000000 \\
        --window-seconds 1.0 --save-image --output data/results/lora_review/

CALIBRATE --snr-threshold-db FIRST, on your own data. The `snr_db` this
pipeline reports is a peak-bin-over-median-bin estimate per time slice; for
pure noise (no signal at all) that ratio is *not* ~0 dB - with the default
1024-bin FFT it's naturally around 8-12 dB, purely from spectral-peak order
statistics. The pipeline-wide default (--snr-threshold-db 6, see
config.DEFAULT_SNR_THRESHOLD_DB) is tuned for smoother satellite Doppler
traces and is too low to gate LoRa burst activity: at 6 dB, noise-only
windows will show valid_signal_ratio=1.0 and get shown to you as if they
were active. Run a first pass over a small slice of your file (--max-windows
20 or so) at the default threshold, look at the snr_db values it prints for
windows you know are just noise, and set --snr-threshold-db a few dB above
that baseline (commonly 15-20 dB) before doing a full run.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (  # noqa: E402
    DEFAULT_NOVERLAP,
    DEFAULT_NPERSEG,
    DEFAULT_SNR_THRESHOLD_DB,
)
from detect import DetectionResult, detect_candidate  # noqa: E402
from features.extractor import FEATURE_NAMES, FeatureVector, extract_features, feature_vector_to_csv_dict  # noqa: E402
from spectrogram import SpectrogramData, iq_to_spectrogram, save_spectrogram_image  # noqa: E402
from storage import safe_stem  # noqa: E402

# Same schema label_capture.py writes - keeps both scripts' output
# interchangeable in the same training CSV.
CSV_COLUMNS = ["capture_id", *FEATURE_NAMES, "label", "is_synthetic", "source_file", "sample_rate_hz", "nperseg", "noverlap", "notes"]

_DTYPE_BYTES = {"complex64": 8, "complex128": 16}


@dataclass
class WindowReview:
    """One window whose activity crossed --activity-ratio and needs a real label."""

    window_index: int
    window_start_s: float
    capture_id: str
    features: FeatureVector
    detection: DetectionResult
    spec: SpectrogramData
    n_windows_estimate: int


@dataclass
class ScanStats:
    windows_scanned: int = 0
    reviewed: int = 0
    auto_negative: int = 0
    skipped_existing: int = 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk a large IQ capture into windows and label each one into a training CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Raw IQ .bin file (interleaved I/Q, no header)")
    parser.add_argument("--dataset", type=Path, default=Path("data/training/features.csv"), help="Training CSV to append to")
    parser.add_argument("--output", type=Path, default=Path("data/results/lora_review"), help="Directory for spectrogram images of reviewed windows")
    parser.add_argument("--sample-rate", type=float, required=True, help="IQ sample rate in Hz")
    parser.add_argument("--center-freq", type=float, required=True, help="Center frequency in Hz")
    parser.add_argument("--binary-dtype", default="complex64", choices=sorted(_DTYPE_BYTES), help="Raw sample dtype")
    parser.add_argument("--window-seconds", type=float, default=1.0, help="Length of each labeled window, in seconds")
    parser.add_argument("--step-seconds", type=float, default=None, help="Hop between window starts, in seconds (default: --window-seconds, i.e. no overlap)")
    parser.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG)
    parser.add_argument("--noverlap", type=int, default=DEFAULT_NOVERLAP)
    parser.add_argument("--snr-threshold-db", type=float, default=DEFAULT_SNR_THRESHOLD_DB)
    parser.add_argument(
        "--activity-ratio",
        type=float,
        default=0.01,
        help="Minimum valid_signal_ratio (fraction of time slices above --snr-threshold-db) for a window to be shown to you for labeling. Below this, it's auto-labeled 0 unread.",
    )
    parser.add_argument("--no-auto-negatives", action="store_true", help="Don't append auto-labeled-0 rows for inactive windows - just skip them")
    parser.add_argument("--save-image", action="store_true", help="Save a spectrogram PNG for each window you're actually prompted on")
    parser.add_argument("--min-drift-hz", type=float, default=0.0, help="Passed through to the rule detector's opinion shown while reviewing (informational only)")
    parser.add_argument("--max-smoothness-hz", type=float, default=float("inf"), help="Passed through to the rule detector's opinion shown while reviewing (informational only)")
    parser.add_argument("--min-valid-ratio", type=float, default=0.0, help="Passed through to the rule detector's opinion shown while reviewing (informational only)")
    parser.add_argument("--max-windows", type=int, default=None, help="Stop after this many windows (for a quick first pass)")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Skip to this offset into the file before starting")
    parser.add_argument("--notes", default="", help="Free-text note appended to every row's notes field")
    return parser


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_has_content = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not file_has_content:
            writer.writeheader()
        writer.writerow(row)


def existing_capture_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {row["capture_id"] for row in csv.DictReader(handle)}


def prompt_for_label() -> "int | None":
    while True:
        raw = input("Label this window [1 = signal present / 0 = noise only / s = skip]: ").strip().lower()
        if raw in ("1", "y", "yes"):
            return 1
        if raw in ("0", "n", "no"):
            return 0
        if raw in ("s", "skip", ""):
            return None
        print("Please enter 1, 0, or s to skip.")


def scan_and_label(
    args: argparse.Namespace,
    label_fn: Callable[[WindowReview], Optional[int]],
    *,
    stop_check: Optional[Callable[[], bool]] = None,
) -> ScanStats:
    """Step through --input in windows, auto-labeling inactive ones and
    calling ``label_fn(review)`` for each active one.

    ``label_fn`` returns 1/0 to record that label, or None to skip the
    window without appending a row. ``stop_check``, if given, is polled
    between windows and stops the scan early when it returns True (e.g. a
    GUI "Stop" button) without losing rows already appended.
    """
    bytes_per_sample = _DTYPE_BYTES[args.binary_dtype]
    total_bytes = args.input.stat().st_size
    total_samples = total_bytes // bytes_per_sample

    window_samples = int(round(args.window_seconds * args.sample_rate))
    step_seconds = args.step_seconds if args.step_seconds is not None else args.window_seconds
    step_samples = int(round(step_seconds * args.sample_rate))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("--window-seconds and --step-seconds must both be positive")

    iq = np.memmap(args.input, dtype=np.dtype(args.binary_dtype), mode="r")
    start_sample = int(round(args.start_seconds * args.sample_rate))

    dataset_path = args.dataset
    already_done = existing_capture_ids(dataset_path)
    stem = safe_stem(args.input)

    n_windows_total = max(0, (total_samples - start_sample - window_samples) // step_samples + 1)
    stats = ScanStats()
    window_index = 0

    while start_sample + window_samples <= total_samples:
        if args.max_windows is not None and window_index >= args.max_windows:
            break
        if stop_check is not None and stop_check():
            break

        end_sample = start_sample + window_samples
        window_start_s = start_sample / args.sample_rate
        capture_id = f"{stem}_w{window_index:05d}_{window_start_s:.3f}s"

        if capture_id in already_done:
            stats.skipped_existing += 1
            window_index += 1
            start_sample += step_samples
            continue

        segment = np.asarray(iq[start_sample:end_sample])
        spec = iq_to_spectrogram(
            segment,
            sample_rate_hz=args.sample_rate,
            center_freq_hz=args.center_freq,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
        )
        features = extract_features(spec, snr_threshold_db=args.snr_threshold_db)

        label = None
        auto = False
        if features.valid_signal_ratio < args.activity_ratio:
            auto = True
            if not args.no_auto_negatives:
                label = 0
        else:
            stats.reviewed += 1
            detection = detect_candidate(
                spec,
                min_valid_ratio=args.min_valid_ratio,
                min_drift_hz=args.min_drift_hz,
                max_smoothness_hz=args.max_smoothness_hz,
                snr_threshold_db=args.snr_threshold_db,
                features=features,
            )
            review = WindowReview(
                window_index=window_index,
                window_start_s=window_start_s,
                capture_id=capture_id,
                features=features,
                detection=detection,
                spec=spec,
                n_windows_estimate=n_windows_total,
            )
            label = label_fn(review)

        if label is not None:
            row = {
                "capture_id": capture_id,
                **feature_vector_to_csv_dict(features),
                "label": label,
                "is_synthetic": 0,
                "source_file": f"{args.input}#window{window_index}@{window_start_s:.3f}s",
                "sample_rate_hz": args.sample_rate,
                "nperseg": args.nperseg,
                "noverlap": args.noverlap,
                "notes": ("auto (below activity threshold, unreviewed)" if auto else "reviewed") + (f"; {args.notes}" if args.notes else ""),
            }
            append_row(dataset_path, row)
            if auto:
                stats.auto_negative += 1

        window_index += 1
        stats.windows_scanned = window_index
        start_sample += step_samples

    return stats


def run(args: argparse.Namespace) -> int:
    bytes_per_sample = _DTYPE_BYTES[args.binary_dtype]
    total_bytes = args.input.stat().st_size
    total_samples = total_bytes // bytes_per_sample
    duration_s = total_samples / args.sample_rate
    print(f"Input: {args.input} ({total_bytes / 1e6:.1f} MB, {total_samples:,} samples, {duration_s:.3f}s at {args.sample_rate:,.0f} Hz)")

    def label_fn(review: WindowReview) -> "int | None":
        print(f"\n--- window {review.window_index} @ {review.window_start_s:.3f}s ({review.window_index + 1} of ~{review.n_windows_estimate}) ---")
        print(f"Rule detector opinion: detected={review.detection.detected} confidence={review.detection.confidence_score:.3f}")
        for name, value in review.features.as_dict().items():
            print(f"  {name}: {value}")
        if args.save_image:
            image_path = save_spectrogram_image(review.spec, args.output / f"{review.capture_id}_spectrogram.png")
            print(f"  spectrogram: {image_path}")
        return prompt_for_label()

    stats = scan_and_label(args, label_fn)

    print(
        f"\nDone. {stats.windows_scanned} windows scanned, {stats.reviewed} shown for review, "
        f"{stats.auto_negative} auto-labeled 0, {stats.skipped_existing} already in {args.dataset} (skipped)."
    )
    return 0


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
