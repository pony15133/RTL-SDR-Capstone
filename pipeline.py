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
from database import init_db, insert_result, update_result  # noqa: E402
from detect import detect_candidate  # noqa: E402
from detection.ml_detector import run_ml_detection  # noqa: E402
from features.extractor import extract_features  # noqa: E402
from iq_io import IQReader  # noqa: E402
from detection.waterfall_detector import predict_waterfall  # noqa: E402
from doppler import STANDARD_SPAN_HZ, DopplerCurve, standard_waterfall  # noqa: E402
from retention import DEFAULT_KEEP_THRESHOLD, DEFAULT_POLICY, POLICIES  # noqa: E402
from retention import apply as retention_apply  # noqa: E402
from retention import decide as retention_decide  # noqa: E402
from spectrogram import iq_to_spectrogram, save_spectrogram_image  # noqa: E402
from storage import safe_stem, save_summary, utc_timestamp  # noqa: E402

#: (kept for reference) rtl_sdr's native raw output: interleaved unsigned 8-bit I/Q samples,
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
    ml_confidence_score: Optional[float] = None
    summary_path: Optional[str] = None
    spectrogram_image: Optional[str] = None
    skipped_reason: Optional[str] = None
    retention_action: Optional[str] = None
    final_iq_path: Optional[str] = None
    wf_status: Optional[str] = None
    wf_confidence_score: Optional[float] = None
    waterfall_image: Optional[str] = None
    doppler_corrected: bool = False

    @property
    def success(self) -> bool:
        return self.recording.success and self.result_id is not None and self.skipped_reason is None


def load_raw_iq_as_complex(iq_path: Path) -> "np.ndarray":  # noqa: F821
    """Load an rtl_sdr raw ``.iq`` capture (interleaved uint8, offset 127.5)
    as complex64. Kept for callers/tests; the work is done by src/iq_io.py."""
    with IQReader(iq_path, "cu8") as reader:
        return reader.read_all()


def save_waterfall_image(matrix, path: Path, target_hz: float, doppler: Optional[DopplerCurve]) -> Path:
    """PNG of the standard waterfall (what a person checks, and what the dashboard shows)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    half_khz = STANDARD_SPAN_HZ / 2e3
    fig, ax = plt.subplots(figsize=(5, 7))
    vmin, vmax = np.percentile(matrix, [5, 99.7])
    ax.imshow(matrix, aspect="auto", origin="upper", cmap="viridis", vmin=vmin, vmax=vmax,
              extent=[-half_khz, half_khz, 1.0, 0.0])
    ax.set_xlabel(f"kHz from {target_hz / 1e6:.4f} MHz")
    ax.set_ylabel("fraction of recording")
    title = "Doppler-corrected" if doppler is not None else "not Doppler-corrected (no pass prediction)"
    if doppler is not None:
        title += f" (max {doppler.max_abs_hz / 1e3:.1f} kHz)"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def recording_metadata_columns(result: RecordingResult, *, frequency_hz=None, sample_rate_hz=None) -> dict:
    """Andre's capture_results metadata columns, filled from the recorder's
    result + JSON sidecar metadata, so every row says what was recorded,
    when, and how."""
    meta = result.metadata
    cols = {
        "recording_status": result.status.value,
        "actual_recording_start": result.actual_recording_start,
        "actual_recording_stop": result.actual_recording_stop,
        "recording_duration_seconds": result.recording_duration_seconds,
        "output_file_size": result.output_file_size,
        "metadata_file_path": result.metadata_file,
        "frequency_hz": int(frequency_hz) if frequency_hz else None,
        "sample_rate": int(sample_rate_hz) if sample_rate_hz else None,
    }
    if meta is not None:
        cols.update({
            "satellite_name": meta.satellite_name,
            "norad_id": meta.norad_id,
            "frequency_hz": meta.frequency_hz,
            "sample_rate": meta.sample_rate,
            "gain": meta.gain,
            "scheduled_aos": meta.scheduled_aos,
            "scheduled_los": meta.scheduled_los,
            "expected_file_size": meta.expected_file_size,
            "simulated": int(bool(meta.simulated)),
            "device_index": meta.device_index,
        })
    return cols


def log_failed_recording(result: RecordingResult, *, db_path: Optional[Path] = None,
                         frequency_hz=None, sample_rate_hz=None, satellite_name: Optional[str] = None) -> int:
    """Insert a row for a recording that did not succeed (busy device,
    cancelled, crash...) so the database is a complete log of recorder runs."""
    db_path = Path(db_path) if db_path else DB_PATH
    row = {
        "input_file": result.output_file or "",
        "timestamp_utc": utc_timestamp(),
        "detection_result": 0,
        "confidence_score": 0.0,
        "valid_signal_ratio": 0.0,
        "frequency_drift_hz": 0.0,
        "smoothness_score": 0.0,
        "notes": f"recording {result.status.value}: {result.error_message or 'no details'}",
        "processing_status": "NOT_PROCESSED_RECORDING_" + result.status.value,
        "iq_retention": "none",
        **recording_metadata_columns(result, frequency_hz=frequency_hz, sample_rate_hz=sample_rate_hz),
    }
    if satellite_name and not row.get("satellite_name"):
        row["satellite_name"] = satellite_name
    init_db(db_path)
    return insert_result(db_path, row)


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
    retention_policy: str = DEFAULT_POLICY,
    keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
    archive_dir: Optional[Path] = None,
    log_failures: bool = False,
    max_detection_seconds: Optional[float] = 120.0,
    target_frequency_hz: Optional[float] = None,
    doppler: Optional[DopplerCurve] = None,
    waterfall_model_path: Optional[Path] = None,
) -> PipelineResult:
    """Run detection on a completed recording, store the result in the DB,
    then keep/archive/delete the IQ file according to ``retention_policy``.

    A recording that didn't succeed has nothing to detect: it is skipped,
    and logged as its own row when ``log_failures`` is True (the automatic
    capture loop does this so every recorder run is in the database).

    ``max_detection_seconds`` bounds memory on very long passes: detection
    runs on the first N seconds (the whole file is still kept/moved as one).

    ``target_frequency_hz`` is the satellite's downlink when the recorder
    was tuned off-frequency (offset tuning keeps the RTL-SDR's DC spike away
    from the signal); ``doppler`` is the predicted Doppler curve for the
    recording. With them, a Doppler-corrected standard waterfall centred on
    the downlink is built and scored by the SatNOGS-trained waterfall model.
    """
    pr = PipelineResult(recording=result)
    if not result.success or not result.output_file:
        pr.skipped_reason = f"recording did not succeed (status={result.status.value}); nothing to detect"
        if log_failures:
            pr.result_id = log_failed_recording(result, db_path=db_path, frequency_hz=frequency_hz,
                                                sample_rate_hz=sample_rate_hz)
        return pr

    db_path = Path(db_path) if db_path else DB_PATH
    output_dir = Path(output_dir) if output_dir else db_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    ml_model_path = Path(ml_model_path) if ml_model_path else DEFAULT_ML_MODEL_PATH
    input_path = Path(result.output_file)

    limit = None if max_detection_seconds is None else int(max_detection_seconds * sample_rate_hz)
    with IQReader(input_path, "cu8") as reader:  # closed before retention may move/delete the file
        iq_samples = reader.read_all(limit)
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

    # Doppler-corrected, SatNOGS-style waterfall around the downlink + the waterfall model.
    wf_detection = None
    wf_image = None
    target = float(target_frequency_hz or frequency_hz)
    try:
        with IQReader(input_path, "cu8") as wf_reader:
            wf_matrix = standard_waterfall(wf_reader, sample_rate_hz, offset_hz=target - frequency_hz, doppler=doppler)
        wf_detection = None if no_ml else predict_waterfall(waterfall_model_path, wf_matrix)
        wf_image = save_waterfall_image(wf_matrix, output_dir / f"{safe_stem(input_path)}_{timestamp.replace(':', '')}"
                                        f"_waterfall.png", target, doppler)
    except ValueError as exc:  # e.g. a very short or narrow recording
        wf_matrix = None
        wf_status_note = f"no standard waterfall: {exc}"
    else:
        wf_status_note = None

    # Prefer the waterfall model (trained on many real SatNOGS passes), then the IQ model, then the rule.
    preferred = wf_detection if (wf_detection is not None and wf_detection.ml_confidence_score is not None) else ml_detection
    decision = retention_decide(ml_detection=preferred, rule_detection=detection, threshold=keep_threshold)
    if preferred is wf_detection and wf_detection is not None:
        decision.source = "waterfall-ml"

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
        "processing_status": "DETECTED",
        "target_frequency_hz": int(target),
        "doppler_corrected": int(doppler is not None),
        "doppler_max_hz": doppler.max_abs_hz if doppler is not None else None,
        "wf_ml_detection_result": None if wf_detection is None or wf_detection.ml_detection_result is None
        else int(wf_detection.ml_detection_result),
        "wf_ml_confidence_score": wf_detection.ml_confidence_score if wf_detection else None,
        "wf_model_version": wf_detection.model_version if wf_detection else None,
        "waterfall_image_path": str(wf_image) if wf_image else None,
        "decision_source": decision.source,
        "decision_score": decision.score,
        **recording_metadata_columns(result, frequency_hz=frequency_hz, sample_rate_hz=sample_rate_hz),
    }
    init_db(db_path)
    result_id = insert_result(db_path, row)

    # Only touch the file after its row is safely in the database.
    outcome = retention_apply(decision, input_path, policy=retention_policy, archive_dir=archive_dir)
    update_result(db_path, result_id, {
        "iq_retention": outcome.action,
        "retention_reason": decision.reason,
        "raw_iq_file_path": outcome.final_path,
    })

    pr.result_id = result_id
    pr.detected = detection.detected
    pr.confidence_score = detection.confidence_score
    pr.ml_status = ml_detection.status if ml_detection else "SKIPPED"
    pr.ml_confidence_score = ml_detection.ml_confidence_score if ml_detection else None
    pr.summary_path = str(summary_path)
    pr.spectrogram_image = str(image_path) if image_path else None
    pr.retention_action = outcome.action
    pr.wf_status = wf_detection.status if wf_detection else (wf_status_note or "SKIPPED")
    pr.wf_confidence_score = wf_detection.ml_confidence_score if wf_detection else None
    pr.waterfall_image = str(wf_image) if wf_image else None
    pr.doppler_corrected = doppler is not None
    pr.final_iq_path = outcome.final_path
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
    ml_model_path: Optional[Path] = None,
    retention_policy: str = DEFAULT_POLICY,
    keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
    log_failures: bool = True,
) -> PipelineResult:
    """Record now for ``duration`` seconds, then detect -> database -> retention.

    ``recorder_config`` lets a caller pass e.g. ``RecorderConfig(simulate=True)``
    for testing without hardware.
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
        ml_model_path=ml_model_path,
        retention_policy=retention_policy,
        keep_threshold=keep_threshold,
        log_failures=log_failures,
    )


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Record IQ samples with iq-recorder and run the result through the "
        "sdr-doppler-prototype detection pipeline into its SQLite database."
    )
    parser.add_argument("--doctor", action="store_true", help="Only check this computer's RTL-SDR setup (tools, device, packages) and exit")
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
    parser.add_argument("--ml-model", type=Path, default=None, help="Trained model (.joblib); default models/random_forest.joblib")
    parser.add_argument("--retention", choices=POLICIES, default=DEFAULT_POLICY,
                        help="What to do with the IQ file of a recording judged 'no satellite' (default keep-all)")
    parser.add_argument("--keep-threshold", type=float, default=DEFAULT_KEEP_THRESHOLD,
                        help="ML confidence needed to count as 'satellite' for retention (default 0.5)")
    return parser


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--doctor" in argv:
        from rtl_recorder.doctor import main as doctor_main

        return doctor_main([a for a in argv if a != "--doctor"])
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
        ml_model_path=args.ml_model,
        retention_policy=args.retention,
        keep_threshold=args.keep_threshold,
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
    if pr.ml_confidence_score is not None:
        print(f"ml_confidence={pr.ml_confidence_score:.3f}")
    print(f"iq_retention={pr.retention_action} -> {pr.final_iq_path}")
    print(f"summary={pr.summary_path}")
    if pr.spectrogram_image:
        print(f"spectrogram_image={pr.spectrogram_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
