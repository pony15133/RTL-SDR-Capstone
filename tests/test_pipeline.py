"""Tests for the pipeline.py connector between iq-recorder and
sdr-doppler-prototype's detection + database.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline  # noqa: E402
from rtl_recorder.config import RecorderConfig  # noqa: E402
from rtl_recorder.metadata import RecordingResult  # noqa: E402
from rtl_recorder.states import RecordingStatus  # noqa: E402


def _write_synthetic_iq(path: Path, sample_rate_hz: float, duration_s: float) -> None:
    """Write a uint8 interleaved I/Q file with an obvious drifting tone.

    Mimics real rtl_sdr output (offset-binary uint8 pairs) so the pipeline's
    ``load_raw_iq_as_complex`` conversion is exercised end to end.
    """
    n = int(sample_rate_hz * duration_s)
    t = np.arange(n) / sample_rate_hz
    freq_hz = np.linspace(-20_000, 20_000, n)  # drifting tone
    phase = 2 * np.pi * np.cumsum(freq_hz) / sample_rate_hz
    tone = 0.5 * np.exp(1j * phase)
    i = np.clip(np.real(tone) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    q = np.clip(np.imag(tone) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    interleaved = np.empty(2 * n, dtype=np.uint8)
    interleaved[0::2] = i
    interleaved[1::2] = q
    interleaved.tofile(path)


def test_load_raw_iq_as_complex_roundtrips(tmp_path):
    path = tmp_path / "sample.iq"
    _write_synthetic_iq(path, sample_rate_hz=240_000, duration_s=0.05)
    iq = pipeline.load_raw_iq_as_complex(path)
    assert iq.dtype == np.complex64
    assert iq.size == 12_000
    assert np.max(np.abs(iq)) <= 1.0 + 1e-6


def test_process_recording_skips_failed_capture():
    result = RecordingResult(status=RecordingStatus.FAILED, error_message="device busy")
    pr = pipeline.process_recording(result, frequency_hz=137_000_000, sample_rate_hz=240_000)
    assert pr.success is False
    assert pr.result_id is None
    assert "did not succeed" in pr.skipped_reason


def test_process_recording_inserts_into_database(tmp_path):
    iq_path = tmp_path / "capture.iq"
    _write_synthetic_iq(iq_path, sample_rate_hz=240_000, duration_s=0.5)
    result = RecordingResult(status=RecordingStatus.SUCCESS, output_file=str(iq_path))
    db_path = tmp_path / "captures.sqlite3"

    pr = pipeline.process_recording(
        result,
        frequency_hz=137_000_000,
        sample_rate_hz=240_000,
        db_path=db_path,
        output_dir=tmp_path,
        no_ml=True,
    )

    assert pr.success
    assert pr.result_id is not None
    assert pr.detected is not None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM capture_results WHERE id = ?", (pr.result_id,)).fetchone()
    assert row is not None
    assert row["raw_iq_file_path"] == str(iq_path)
    assert row["input_file"] == str(iq_path)


def test_capture_and_detect_simulated_end_to_end(tmp_path):
    """Full auto-capture -> detect -> database path, using the recorder's
    simulate mode so no RTL-SDR hardware is required."""
    db_path = tmp_path / "captures.sqlite3"
    pr = pipeline.capture_and_detect(
        satellite_name="TEST-SAT",
        frequency_hz=137_000_000,
        sample_rate=240_000,
        gain="auto",
        duration=1,
        recorder_config=RecorderConfig(simulate=True, output_dir=str(tmp_path / "recordings")),
        db_path=db_path,
        detection_output_dir=tmp_path,
        no_ml=True,
    )

    assert pr.recording.status == RecordingStatus.SUCCESS
    assert pr.success
    assert Path(pr.recording.output_file).exists()

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM capture_results").fetchone()[0]
    assert count == 1
