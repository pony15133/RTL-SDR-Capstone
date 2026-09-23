#!/usr/bin/env python3
"""Chunk one large raw IQ capture into fixed-length, overlapping time
windows, run the existing feature extraction + rule detector on each
window, and append a labelled row per window to the training CSV - same
schema as scripts/label_capture.py / scripts/build_training_dataset.py.

Built for captures where a real signal occupies only a small fraction of
the file - e.g. the LoRadar satellite-LoRa dataset (4 MHz, complex64,
~3% packet duty cycle). Treating a whole multi-second/minute file as one
row/one label would wash the few real packets out into statistical
noise; this instead scores every window individually. See
data/training/README.md for when to use this vs. the other two tools.

Labelling per window:
    - The rule detector (detect.detect_candidate(), same thresholds and
      code as the rest of the pipeline) runs on every window - nothing
      new is invented here.
    - A window it rejects (detected=False) is auto-labelled 0. At this
      duty cycle the overwhelming majority of windows are genuinely
      empty, so this is a safe bulk shortcut - rows are tagged in
      `notes` as auto-negative, staying honestly distinguishable from a
      human-reviewed label.
    - A window it flags (detected=True) is a *candidate*: by default you
      are shown its features (and, with --save-candidate-images, a
      spectrogram PNG) and asked to confirm 1/0/skip. Pass
      --auto-accept-candidates to trust the rule detector's label
      directly without stopping - those rows are tagged "auto-accepted,
      not manually reviewed" in `notes`, so a weak label is never
      silently indistinguishable from a reviewed one.

Windows are keyed by (file path, start sample, end sample) for dedup, so
re-running this against the same file with the same --window-seconds/
--stride-seconds is safe - already-processed windows are skipped.

Example (LoRadar-style capture: 4 MHz, complex64, band centred ~401.3 MHz):
    python scripts/chunk_bin_to_dataset.py \\
        --input /path/to/session_001.bin \\
        --dataset data/training/features.csv \\
        --sample-rate 4000000 --center-freq 401300000 --binary-dtype complex64 \\
        --window-seconds 2.0 --stride-seconds 1.0 \\
        --snr-threshold-db 15 --min-valid-ratio 0.2 \\
        --save-candidate-images --output data/results/lora_review

IMPORTANT - the rule detector's DEFAULT thresholds (--snr-threshold-db 6,
--min-valid-ratio 0.45) were tuned for a signal spanning nearly an entire
capture (a satellite pass), not a short burst inside a much longer
window. Verified empirically on a synthetic burst-in-noise window: at
the default 6 dB threshold, pure noise alone crosses the strongest-bin-
vs-median-power check on almost every time slice once there are ~1000+
frequency bins (nperseg=1024), so valid_signal_ratio saturates near 1.0
for noise AND for real signal alike - useless as a discriminator - and
the real burst's clean, smooth trace gets diluted by the noise-dominated
rest of the window, so it fails max_smoothness_hz too. Raising
--snr-threshold-db to ~15 dB fixed the false-"valid" rate under noise in
that test, and lowering --min-valid-ratio to ~0.2 correctly allowed a
burst occupying under a third of the window to still register. These are
a validated *starting point* on synthetic data, not a tuned result on
your actual captures - confirm against a couple of known-real windows
before trusting --auto-accept-candidates on a full file.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (  # noqa: E402
    DEFAULT_MAX_SMOOTHNESS_HZ,
    DEFAULT_MIN_DRIFT_HZ,
    DEFAULT_MIN_VALID_RATIO,
    DEFAULT_NOVERLAP,
    DEFAULT_NPERSEG,
    DEFAULT_SNR_THRESHOLD_DB,
)
from detect import detect_candidate  # noqa: E402
from features.dataset_io import append_row, existing_source_files, feature_row_values, prompt_for_label  # noqa: E402
from features.extractor import extract_features  # noqa: E402
from load_data import load_input  # noqa: E402
from spectrogram import iq_to_spectrogram, save_spectrogram_image  # noqa: E402
from storage import safe_stem, utc_timestamp  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk a large raw IQ capture into windows and append labelled rows to a training CSV."
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw IQ capture file (.bin/.iq/.dat/.npy)")
    parser.add_argument("--dataset", type=Path, default=Path("data/training/features.csv"), help="Training CSV to append to")
    parser.add_argument("--sample-rate", type=float, required=True, help="IQ sample rate in Hz (e.g. 4000000 for the LoRadar dataset)")
    parser.add_argument("--center-freq", type=float, required=True, help="Centre frequency in Hz the capture was tuned to")
    parser.add_argument("--binary-dtype", default="complex64", help="dtype for raw binary IQ files (.bin/.iq/.dat)")
    parser.add_argument("--window-seconds", type=float, default=2.0, help="Window length in seconds (default: 2.0)")
    parser.add_argument(
        "--stride-seconds", type=float, default=1.0,
        help="Step between window starts in seconds (default: 1.0, i.e. 50%% overlap at the default window length)",
    )
    parser.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG)
    parser.add_argument("--noverlap", type=int, default=DEFAULT_NOVERLAP)
    parser.add_argument("--snr-threshold-db", type=float, default=DEFAULT_SNR_THRESHOLD_DB)
    parser.add_argument("--min-valid-ratio", type=float, default=DEFAULT_MIN_VALID_RATIO)
    parser.add_argument("--min-drift-hz", type=float, default=DEFAULT_MIN_DRIFT_HZ)
    parser.add_argument("--max-smoothness-hz", type=float, default=DEFAULT_MAX_SMOOTHNESS_HZ)
    parser.add_argument(
        "--auto-accept-candidates", action="store_true",
        help="Trust the rule detector's label on candidate windows instead of prompting interactively",
    )
    parser.add_argument("--save-candidate-images", action="store_true", help="Save a spectrogram PNG for each candidate window")
    parser.add_argument("--output", type=Path, default=Path("data/results"), help="Directory for candidate spectrogram images")
    parser.add_argument("--max-windows", type=int, default=None, help="Stop after this many windows (useful for a quick first test on a new file)")
    parser.add_argument("--reprocess", action="store_true", help="Reprocess windows even if already present in --dataset")
    parser.add_argument("--notes", default="", help="Extra free-text appended to every row's notes")
    return parser


def iter_windows(total_samples: int, window_samples: int, stride_samples: int) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) sample index pairs. Yields nothing if the capture is shorter than one window."""
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window/stride sizes must be positive")
    start = 0
    while start + window_samples <= total_samples:
        yield start, start + window_samples
        start += stride_samples


def window_source_key(path: Path, start: int, end: int) -> str:
    """Deterministic dedup key - independent of wall-clock time, so a
    re-run with the same window/stride skips already-processed windows."""
    return f"{path}#samples={start}-{end}"


def run(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"FAILED: --input file not found: {args.input}", file=sys.stderr)
        return 1

    loaded = load_input(args.input, binary_dtype=args.binary_dtype)
    if loaded.kind != "iq":
        print(
            f"FAILED: {args.input} was loaded as a spectrogram matrix, not raw IQ - "
            "this tool chunks raw IQ captures only.",
            file=sys.stderr,
        )
        return 1

    total_samples = loaded.values.shape[0]
    window_samples = int(round(args.window_seconds * args.sample_rate))
    stride_samples = int(round(args.stride_seconds * args.sample_rate))
    if window_samples <= 0 or stride_samples <= 0:
        print("FAILED: --window-seconds and --stride-seconds must both be positive.", file=sys.stderr)
        return 1

    windows = list(iter_windows(total_samples, window_samples, stride_samples))
    if args.max_windows:
        windows = windows[: args.max_windows]
    if not windows:
        duration_s = total_samples / args.sample_rate
        print(
            f"FAILED: capture is only {duration_s:.3f}s long at {args.sample_rate:.0f} Hz - "
            f"shorter than one {args.window_seconds}s window. Reduce --window-seconds.",
            file=sys.stderr,
        )
        return 1

    already_processed = set() if args.reprocess else existing_source_files(args.dataset)

    n_auto_negative = n_reviewed_positive = n_reviewed_negative = n_auto_accepted = 0
    n_skipped_duplicate = n_skipped_review = n_failed = 0

    for index, (start, end) in enumerate(windows, start=1):
        source_key = window_source_key(args.input, start, end)
        print(f"[{index}/{len(windows)}] samples {start}-{end} ({start / args.sample_rate:.2f}s-{end / args.sample_rate:.2f}s)")
        if source_key in already_processed:
            print("  skipped (already in dataset)")
            n_skipped_duplicate += 1
            continue

        try:
            window_iq = loaded.values[start:end]
            spec = iq_to_spectrogram(
                window_iq,
                sample_rate_hz=args.sample_rate,
                center_freq_hz=args.center_freq,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
            )
            features = extract_features(spec, snr_threshold_db=args.snr_threshold_db, time_axis_is_synthetic=False)
            detection = detect_candidate(
                spec,
                min_valid_ratio=args.min_valid_ratio,
                min_drift_hz=args.min_drift_hz,
                max_smoothness_hz=args.max_smoothness_hz,
                snr_threshold_db=args.snr_threshold_db,
                features=features,
            )
        except Exception as exc:  # a bad window must not abort the rest of the file
            print(f"  FAILED to process ({exc}) - skipping", file=sys.stderr)
            n_failed += 1
            continue

        if not detection.detected:
            label = 0
            note = "auto-negative (rule detector: no signal)"
            n_auto_negative += 1
        else:
            if args.save_candidate_images:
                timestamp = utc_timestamp()
                image_path = save_spectrogram_image(
                    spec,
                    args.output / f"{safe_stem(args.input)}_w{index:05d}_{timestamp.replace(':', '')}_spectrogram.png",
                )
                print(f"  candidate spectrogram: {image_path}")

            print(
                f"  CANDIDATE: rule confidence={detection.confidence_score:.3f}, "
                f"drift={detection.frequency_drift_hz:.1f} Hz, valid_ratio={detection.valid_signal_ratio:.3f}"
            )

            if args.auto_accept_candidates:
                label = 1
                note = "auto-accepted candidate (rule detector), not manually reviewed"
                n_auto_accepted += 1
            else:
                chosen = prompt_for_label(context=f"window {start}-{end} of {args.input}")
                if chosen is None:
                    print("  skipped (no label given)")
                    n_skipped_review += 1
                    continue
                label = chosen
                note = "manually reviewed candidate"
                if label == 1:
                    n_reviewed_positive += 1
                else:
                    n_reviewed_negative += 1

        if args.notes:
            note = f"{note}; {args.notes}"

        timestamp = utc_timestamp()
        row = {
            "capture_id": f"{safe_stem(args.input)}_w{index:05d}_{timestamp.replace(':', '')}",
            **feature_row_values(features),
            "label": label,
            "is_synthetic": 0,
            "source_file": source_key,
            "sample_rate_hz": args.sample_rate,
            "nperseg": args.nperseg,
            "noverlap": args.noverlap,
            "notes": note,
        }
        append_row(args.dataset, row)
        already_processed.add(source_key)
        print(f"  appended capture_id={row['capture_id']} label={label}")

    total_appended = n_auto_negative + n_reviewed_positive + n_reviewed_negative + n_auto_accepted
    print()
    print(
        f"Done: {total_appended} appended ({n_auto_negative} auto-negative, {n_reviewed_positive} reviewed-positive, "
        f"{n_reviewed_negative} reviewed-negative, {n_auto_accepted} auto-accepted-candidate), "
        f"{n_skipped_duplicate} already present, {n_skipped_review} skipped at review, {n_failed} failed"
    )
    print(f"Dataset: {args.dataset}")
    if total_appended:
        print(f"Train with: python train_model.py --dataset {args.dataset} --output models/random_forest.joblib")
    return 1 if n_failed and total_appended == 0 else 0


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
