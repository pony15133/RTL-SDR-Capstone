import argparse
from pathlib import Path

from config import (
    DB_PATH,
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
from database import insert_result, init_db
from detect import detect_candidate
from detection.ml_detector import run_ml_detection
from features.extractor import extract_features
from iq_io import is_raw_iq_file, probe
from load_data import is_raw_iq_path, load_input
from spectrogram import (
    iq_to_spectrogram,
    matrix_to_spectrogram,
    save_chunked_spectrogram_images,
    save_spectrogram_image,
)
from storage import (
    ensure_session_output_dir,
    format_detection_summary,
    safe_stem,
    save_summary,
    utc_timestamp,
)
<<<<<<< HEAD


def print_progress(label: str, current: int, total: int) -> None:
    if total <= 0:
        total = 1
    percent = min(100, max(0, int((current / total) * 100)))
    width = 20
    filled = int(width * percent / 100)
    bar = "#" * filled + "-" * (width - filled)
    print(f"{label}: {percent:3d}% |{bar}| {current}/{total}")
=======
>>>>>>> 482f8559d715eca22d15970253af36deb909de15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect candidate satellite Doppler traces in SDR data.")
    parser.add_argument("--input", required=True, type=Path, help="Input IQ file (.npy/.bin/.iq) or spectrogram matrix (.npy/.txt/.csv)")
    parser.add_argument("--output", required=True, type=Path, help="Directory for result summaries and images")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path")
    parser.add_argument("--sample-rate", type=float, default=None,
                        help=f"IQ sample rate in Hz (default: from SigMF/recorder metadata or file name, else {DEFAULT_SAMPLE_RATE_HZ:g})")
    parser.add_argument("--center-freq", type=float, default=None,
                        help=f"Center frequency in Hz (default: from metadata or file name, else {DEFAULT_CENTER_FREQ_HZ:g})")
    parser.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG, help="Spectrogram FFT segment length")
    parser.add_argument("--noverlap", type=int, default=DEFAULT_NOVERLAP, help="Spectrogram overlap")
    parser.add_argument("--binary-dtype", default="auto",
                        help="IQ format for raw files: auto (default), cu8 (rtl_sdr .iq), ci16 (SigMF/CAMRAS), complex64, wav")
    parser.add_argument("--snr-threshold-db", type=float, default=DEFAULT_SNR_THRESHOLD_DB)
    parser.add_argument("--min-valid-ratio", type=float, default=DEFAULT_MIN_VALID_RATIO)
    parser.add_argument("--min-drift-hz", type=float, default=DEFAULT_MIN_DRIFT_HZ)
    parser.add_argument("--max-smoothness-hz", type=float, default=DEFAULT_MAX_SMOOTHNESS_HZ)
    parser.add_argument("--save-image", action="store_true", help="Save a PNG spectrogram image")
    parser.add_argument(
        "--image-chunk-size",
        type=int,
        default=None,
        help="If set, render the spectrogram in chunks of this many time rows to avoid memory blow-ups on large captures.",
    )
    parser.add_argument(
        "--image-chunk-overlap",
        type=int,
        default=0,
        help="Overlap between successive spectrogram chunks when using --image-chunk-size.",
    )
    parser.add_argument("--ml-model", type=Path, default=DEFAULT_ML_MODEL_PATH, help="Path to a trained Random Forest model (.joblib)")
    parser.add_argument("--no-ml", action="store_true", help="Skip ML detection even if a trained model is available")
    return parser


def resolve_capture_parameters(args: argparse.Namespace) -> None:
    """Fill in --sample-rate / --center-freq from the recording's own
    metadata when they weren't given, so the frequency axis is right."""
    info = probe(args.input) if is_raw_iq_file(args.input) else None
    if args.sample_rate is None:
        args.sample_rate = (info.sample_rate_hz if info and info.sample_rate_hz else DEFAULT_SAMPLE_RATE_HZ)
    if args.center_freq is None:
        args.center_freq = (info.center_freq_hz if info and info.center_freq_hz else DEFAULT_CENTER_FREQ_HZ)


def run(args: argparse.Namespace) -> int:
    resolve_capture_parameters(args)
    print_progress("Loading input", 10, 100)
    loaded = load_input(args.input, binary_dtype=args.binary_dtype)
    time_axis_is_synthetic = loaded.kind != "iq"
    if loaded.kind == "iq":
        print_progress("Building spectrogram", 25, 100)
        spec = iq_to_spectrogram(
            loaded.values,
            sample_rate_hz=args.sample_rate,
            center_freq_hz=args.center_freq,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
        )
    else:
        print_progress("Loading spectrogram matrix", 25, 100)
        spec = matrix_to_spectrogram(
            loaded.values,
            sample_rate_hz=args.sample_rate,
            center_freq_hz=args.center_freq,
        )

    print_progress("Extracting signal features", 50, 100)
    features = extract_features(spec, snr_threshold_db=args.snr_threshold_db, time_axis_is_synthetic=time_axis_is_synthetic)

    print_progress("Running rule-based detection", 70, 100)
    detection = detect_candidate(
        spec,
        min_valid_ratio=args.min_valid_ratio,
        min_drift_hz=args.min_drift_hz,
        max_smoothness_hz=args.max_smoothness_hz,
        snr_threshold_db=args.snr_threshold_db,
        features=features,
    )

    print_progress("Running ML detection", 80, 100)
    ml_detection = None if args.no_ml else run_ml_detection(args.ml_model, features)

    timestamp = utc_timestamp()
<<<<<<< HEAD
    session_dir = ensure_session_output_dir(args.output)
    image_path = None
    if args.save_image:
        print_progress("Saving spectrogram image", 90, 100)
=======
    # Groups this run's summary/image(s) under their own timestamped folder
    # instead of a flat output dir - what the GUI's results/image-preview
    # panels browse as "today's session".
    session_dir = ensure_session_output_dir(args.output)
    image_path = None
    if args.save_image:
>>>>>>> 482f8559d715eca22d15970253af36deb909de15
        if args.image_chunk_size and args.image_chunk_size > 0:
            chunk_paths = save_chunked_spectrogram_images(
                spec,
                session_dir / "spectrogram_chunks",
                chunk_size=args.image_chunk_size,
                overlap=args.image_chunk_overlap,
                prefix=f"{safe_stem(args.input)}_{timestamp.replace(':', '')}_spectrogram",
            )
            image_path = chunk_paths[0] if chunk_paths else None
        else:
            image_path = save_spectrogram_image(
                spec,
                session_dir / f"{safe_stem(args.input)}_{timestamp.replace(':', '')}_spectrogram.png",
            )

    summary_path = save_summary(session_dir, args.input, timestamp, detection, ml_detection)

    raw_iq_file_path = str(args.input) if detection.detected and loaded.kind == "iq" and is_raw_iq_path(args.input) else None
    row = {
        "input_file": str(args.input),
        "timestamp_utc": timestamp,
        "detection_result": int(detection.detected),
        "confidence_score": detection.confidence_score,
        "valid_signal_ratio": detection.valid_signal_ratio,
        "frequency_drift_hz": detection.frequency_drift_hz,
        "smoothness_score": detection.smoothness_score,
        "spectrogram_image_path": str(image_path) if image_path else None,
        "raw_iq_file_path": raw_iq_file_path,
        "notes": detection.notes,
        "rule_detection_result": int(detection.detected),
        "rule_confidence_score": detection.confidence_score,
        "ml_detection_result": None if ml_detection is None or ml_detection.ml_detection_result is None else int(ml_detection.ml_detection_result),
        "ml_confidence_score": ml_detection.ml_confidence_score if ml_detection else None,
        "model_version": ml_detection.model_version if ml_detection else None,
    }
    init_db(args.db)
    result_id = insert_result(args.db, row)

    print_progress("Finalising result", 100, 100)
    print()
    print(format_detection_summary({
        "rule_detection_result": int(detection.detected),
        "rule_confidence_score": detection.confidence_score,
        "valid_signal_ratio": detection.valid_signal_ratio,
        "frequency_drift_hz": detection.frequency_drift_hz,
        "smoothness_score": detection.smoothness_score,
        "ml_detection_result": None if ml_detection is None or ml_detection.ml_detection_result is None else int(ml_detection.ml_detection_result),
        "ml_confidence_score": ml_detection.ml_confidence_score if ml_detection else None,
        "notes": detection.notes,
    }))
    print()
    print(f"Result ID: {result_id}")
    print(f"Summary JSON: {summary_path}")
    if image_path:
<<<<<<< HEAD
        print(f"Spectrogram image: {image_path}")
=======
        print(f"spectrogram_image={image_path}")
    print()
    print(format_detection_summary(row))
>>>>>>> 482f8559d715eca22d15970253af36deb909de15
    return 0


def main() -> int:
    parser = build_parser()
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
