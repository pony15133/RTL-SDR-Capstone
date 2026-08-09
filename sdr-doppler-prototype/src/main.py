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
from load_data import is_raw_iq_path, load_input
from spectrogram import iq_to_spectrogram, matrix_to_spectrogram, save_spectrogram_image
from storage import safe_stem, save_summary, utc_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect candidate satellite Doppler traces in SDR data.")
    parser.add_argument("--input", required=True, type=Path, help="Input IQ file (.npy/.bin/.iq) or spectrogram matrix (.npy/.txt/.csv)")
    parser.add_argument("--output", required=True, type=Path, help="Directory for result summaries and images")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ, help="IQ sample rate in Hz")
    parser.add_argument("--center-freq", type=float, default=DEFAULT_CENTER_FREQ_HZ, help="Center frequency in Hz")
    parser.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG, help="Spectrogram FFT segment length")
    parser.add_argument("--noverlap", type=int, default=DEFAULT_NOVERLAP, help="Spectrogram overlap")
    parser.add_argument("--binary-dtype", default="complex64", help="dtype for raw binary IQ files")
    parser.add_argument("--snr-threshold-db", type=float, default=DEFAULT_SNR_THRESHOLD_DB)
    parser.add_argument("--min-valid-ratio", type=float, default=DEFAULT_MIN_VALID_RATIO)
    parser.add_argument("--min-drift-hz", type=float, default=DEFAULT_MIN_DRIFT_HZ)
    parser.add_argument("--max-smoothness-hz", type=float, default=DEFAULT_MAX_SMOOTHNESS_HZ)
    parser.add_argument("--save-image", action="store_true", help="Save a PNG spectrogram image")
    parser.add_argument("--ml-model", type=Path, default=DEFAULT_ML_MODEL_PATH, help="Path to a trained Random Forest model (.joblib)")
    parser.add_argument("--no-ml", action="store_true", help="Skip ML detection even if a trained model is available")
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
        spec = matrix_to_spectrogram(
            loaded.values,
            sample_rate_hz=args.sample_rate,
            center_freq_hz=args.center_freq,
        )

    # Computed once, consumed by both the rule-based baseline and the ML
    # detector - see features/extractor.py.
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
            spec,
            args.output / f"{safe_stem(args.input)}_{timestamp.replace(':', '')}_spectrogram.png",
        )

    summary_path = save_summary(args.output, args.input, timestamp, detection, ml_detection)

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

    print(f"result_id={result_id}")
    print("--- rule-based baseline ---")
    print(f"detected={detection.detected}")
    print(f"confidence_score={detection.confidence_score:.3f}")
    print(f"valid_signal_ratio={detection.valid_signal_ratio:.3f}")
    print(f"frequency_drift_hz={detection.frequency_drift_hz:.1f}")
    print(f"smoothness_score={detection.smoothness_score:.1f}")
    print("--- machine learning (Random Forest) ---")
    if ml_detection is None:
        print("ml_status=SKIPPED (--no-ml)")
    else:
        print(f"ml_status={ml_detection.status}")
        if ml_detection.available:
            print(f"ml_detection_result={ml_detection.ml_detection_result}")
            print(f"ml_confidence_score={ml_detection.ml_confidence_score:.3f}")
            print(f"model_version={ml_detection.model_version}")
        else:
            print(f"ml_reason={ml_detection.reason}")
    print(f"summary={summary_path}")
    if image_path:
        print(f"spectrogram_image={image_path}")
    return 0


def main() -> int:
    parser = build_parser()
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
