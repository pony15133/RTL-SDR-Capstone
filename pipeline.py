#!/usr/bin/env python3
"""Connects iq-recorder (auto capture) to the sdr-doppler-prototype
detection pipeline and its SQLite results database.

Before this module existed, the two subprojects were completely
disconnected: ``iq-recorder`` wrote a raw ``.iq`` file + JSON metadata
sidecar and stopped, and ``sdr-doppler-prototype`` only ever ran manually
via ``src/main.py`` against a file you pointed it at by hand - nothing
wrote a capture into ``capture_results`` (see
sdr-doppler-prototype/src/database.py) automatically.

This module is the connector: :func:`capture_and_detect` drives one
RTL-SDR recording through ``rtl_recorder.RTLSDRRecorder`` and, on
success, feeds the resulting IQ file straight into the doppler
prototype's spectrogram -> feature extraction -> rule/ML detection ->
``capture_results`` row, exactly as ``main.py`` would for a manually
supplied file. :func:`process_recording` does just the second half, for
callers that already have a ``RecordingResult`` (e.g. a future
scheduler calling ``record_pass()``).

Both subprojects keep their own independent ``src`` layout (see their
READMEs) - this module only adds their directories to ``sys.path`` at
import time so it can import both without turning either into an
installable package.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
IQ_RECORDER_ROOT = REPO_ROOT / "iq-recorder"
DOPPLER_SRC = REPO_ROOT / "sdr-doppler-prototype" / "src"

for _path in (IQ_RECORDER_ROOT, DOPPLER_SRC):
    _str_path = str(_path)
    if _str_path not in sys.path:
        sys.path.insert(0, _str_path)

from rtl_recorder.config import RecorderConfig  # noqa: E402
from rtl_recorder.metadata import RecordingResult  # noqa: E402
from rtl_recorder.recorder import RTLSDRRecorder  # noqa: E402

from config import DB_PATH, DEFAULT_ML_MODEL_PATH  # noqa: E402
from database import init_db, insert_result  # noqa: E402
from detect import detect_candidate  # noqa: E402
from detection.ml_detector import run_ml_detection  # noqa: E402
from features.extractor import extract_features  # noqa: E402
from spectrogram import iq_to_spectrogram, save_spectrogram_image  # noqa: E402
from storage import safe_stem, save_summary, utc_timestamp  # noqa: E402

#: rtl_sdr's native raw output: interleaved unsigned 8-bit I/Q samples,
#: offset-binary around 127.5 (see rtl_recorder README / rtl_sdr(1)). This
#: is what the recorder actually writes in real (non-simulated) mode.
RAW_IQ_DTYPE = "uint8"


@dataclass
class PipelineResult:
    """What one recorder-to-database run produced."""

    recording: RecordingResult
    result_id: Optional[int] = None
    detected: Optional[bool] = None
    confidence_score: Optional[float] = None
    ml_status: Optional[str] = None
    summary_path: Optional[str] = None
    spectrogram_image: Optional[str] = None
    skipped_reason: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.recording.success and self.result_id is not None


def load_raw_iq_as_complex(iq_path: Path) -> "np.ndarray":  # noqa: F821 - numpy imported lazily below
    """Load an rtl_sdr raw ``.iq`` capture as a complex64 sample array.

    rtl_sdr writes interleaved uint8 I/Q pairs centered on 127.5, not the
    complex64 float format ``sdr-doppler-prototype/src/load_data.py``
    otherwise assumes for ``.iq``/``.bin`` files - so the conversion is
    done explicitly here rather than routing through ``load_input``.
    """
    import numpy as np

    raw = np.fromfile(iq_path, dtype=RAW_IQ_DTYPE)
    if raw.size % 2 != 0:
        raw = raw[:-1]  # drop a trailing unpaired byte from a truncated capture
    raw = raw.astype(np.float32)
    iq = ((raw[0::2] - 127.5) + 1j * (raw[1::2] - 127.5)) / 127.5
    return iq.astype(np.complex64)


def process_recording(
    result: RecordingResult,
    *,
    frequency_hz: float,
    sample_rate_hz: float,
    db_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    nperseg: int = 1024,
    noverlap: int = 512,
    snr_threshold_db: float = 6.0,
    min_valid_ratio: float = 0.45,
    min_drift_hz: float = 1_500.0,
    max_smoothness_hz: float = 8_000.0,
    ml_model_path: Optional[Path] = None,
    no_ml: bool = False,
    save_image: bool = False,
) -> PipelineResult:
    """Run detection on a completed recording and store the result in the DB.

    Mirrors sdr-doppler-prototype/src/main.py's pipeline, but takes a
    ``RecordingResult`` from ``rtl_recorder`` instead of a CLI-supplied
    ``--input`` path. A recording that didn't succeed (device busy,
    cancelled, failed) is skipped - there's nothing to detect.
    """
    pr = PipelineResult(recording=result)
    if not result.success or not result.output_file:
        pr.skipped_reason = f"recording did not succeed (status={result.status.value}); nothing to detect"
        return pr

    db_path = Path(db_path) if db_path else DB_PATH
    output_dir = Path(output_dir) if output_dir else db_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    ml_model_path = Path(ml_model_path) if ml_model_path else DEFAULT_ML_MODEL_PATH
    input_path = Path(result.output_file)

    iq_samples = load_raw_iq_as_complex(input_path)
    spec = iq_to_spectrogram(
        iq_samples,
        sample_rate_hz=sample_rate_hz,
        center_freq_hz=frequency_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    features = extract_features(spec, snr_threshold_db=snr_threshold_db)
    detection = detect_candidate(
        spec,
        min_valid_ratio=min_valid_ratio,
        min_drift_hz=min_drift_hz,
        max_smoothness_hz=max_smoothness_hz,
        snr_threshold_db=snr_threshold_db,
        features=features,
    )
    ml_detection = None if no_ml else run_ml_detection(ml_model_path, features)

    timestamp = utc_timestamp()
    image_path = None
    if save_image:
        image_path = save_spectrogram_image(
            spec,
            output_dir / f"{safe_stem(input_path)}_{timestamp.replace(':', '')}_spectrogram.png",
        )
    summary_path = save_summary(output_dir, input_path, timestamp, detection, ml_detection)

    row = {
        "input_file": str(input_path),
        "timestamp_utc": timestamp,
        "detection_result": int(detection.detected),
        "confidence_score": detection.confidence_score,
        "valid_signal_ratio": detection.valid_signal_ratio,
        "frequency_drift_hz": detection.frequency_drift_hz,
        "smoothness_score": detection.smoothness_score,
        "spectrogram_image_path": str(image_path) if image_path else None,
        "raw_iq_file_path": str(input_path),
        "notes": detection.notes,
        "rule_detection_result": int(detection.detected),
        "rule_confidence_score": detection.confidence_score,
        "ml_detection_result": None if ml_detection is None or ml_detection.ml_detection_result is None else int(ml_detection.ml_detection_result),
        "ml_confidence_score": ml_detection.ml_confidence_score if ml_detection else None,
        "model_version": ml_detection.model_version if ml_detection else None,
    }
    init_db(db_path)
    result_id = insert_result(db_path, row)

    pr.result_id = result_id
    pr.detected = detection.detected
    pr.confidence_score = detection.confidence_score
    pr.ml_status = ml_detection.status if ml_detection else "SKIPPED"
    pr.summary_path = str(summary_path)
    pr.spectrogram_image = str(image_path) if image_path else None
    return pr


def capture_and_detect(
    *,
    satellite_name: str,
    frequency_hz,
    sample_rate,
    gain,
    duration,
    norad_id: Optional[int] = None,
    recorder_config: Optional[RecorderConfig] = None,
    output_dir: Optional[str] = None,
    db_path: Optional[Path] = None,
    detection_output_dir: Optional[Path] = None,
    no_ml: bool = False,
    save_image: bool = False,
) -> PipelineResult:
    """Record one pass, then run it through detection and store the result.

    This is the "auto capture" entry point: capture -> detect -> database,
    in one call. ``recorder_config`` lets a caller pass e.g.
    ``RecorderConfig(simulate=True)`` for testing without hardware.
    """
    recorder = RTLSDRRecorder(recorder_config or RecorderConfig())
    result = recorder.record(
        satellite_name=satellite_name,
        frequency_hz=frequency_hz,
        sample_rate=sample_rate,
        gain=gain,
        duration=duration,
        norad_id=norad_id,
        output_dir=output_dir,
    )
    return process_recording(
        result,
        frequency_hz=frequency_hz,
        sample_rate_hz=sample_rate,
        db_path=db_path,
        output_dir=detection_output_dir,
        no_ml=no_ml,
        save_image=save_image,
    )


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Record IQ samples with iq-recorder and run the result through the "
        "sdr-doppler-prototype detection pipeline into its SQLite database."
    )
    parser.add_argument("--satellite", required=True, help="Satellite/target name")
    parser.add_argument("--frequency", required=True, type=float, help="Center frequency in Hz")
    parser.add_argument("--sample-rate", required=True, type=float, help="Sample rate in Hz")
    parser.add_argument("--gain", default="auto", help="Tuner gain in dB, or 'auto'")
    parser.add_argument("--duration", required=True, type=float, help="Recording duration in seconds")
    parser.add_argument("--norad-id", type=int, default=None)
    parser.add_argument("--output-dir", default=None, help="Directory for the .iq/.json recording")
    parser.add_argument("--db", type=Path, default=None, help="SQLite database path (default: sdr-doppler-prototype's)")
    parser.add_argument("--detection-output-dir", type=Path, default=None, help="Directory for summaries/images")
    parser.add_argument("--simulate", action="store_true", help="Simulate the recording (no hardware required)")
    parser.add_argument("--no-ml", action="store_true", help="Skip ML detection")
    parser.add_argument("--save-image", action="store_true", help="Save a PNG spectrogram image")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    pr = capture_and_detect(
        satellite_name=args.satellite,
        frequency_hz=args.frequency,
        sample_rate=args.sample_rate,
        gain=args.gain,
        duration=args.duration,
        norad_id=args.norad_id,
        recorder_config=RecorderConfig(simulate=args.simulate),
        output_dir=args.output_dir,
        db_path=args.db,
        detection_output_dir=args.detection_output_dir,
        no_ml=args.no_ml,
        save_image=args.save_image,
    )

    print(f"recording_status={pr.recording.status.value}")
    if not pr.success:
        print(f"skipped={pr.skipped_reason or pr.recording.error_message}")
        return 1

    print(f"output_file={pr.recording.output_file}")
    print(f"result_id={pr.result_id}")
    print(f"detected={pr.detected}")
    print(f"confidence_score={pr.confidence_score:.3f}")
    print(f"ml_status={pr.ml_status}")
    print(f"summary={pr.summary_path}")
    if pr.spectrogram_image:
        print(f"spectrogram_image={pr.spectrogram_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
