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


# --------------------------------------------------------------------------- #
# Recorder metadata in the database + IQ retention
# --------------------------------------------------------------------------- #

from types import SimpleNamespace  # noqa: E402

import retention  # noqa: E402
from database import get_result  # noqa: E402


def _simulated(tmp_path, **kwargs):
    return pipeline.capture_and_detect(
        satellite_name="TEST-SAT", frequency_hz=137_000_000, sample_rate=240_000, gain=20, duration=1,
        norad_id=12345,
        recorder_config=RecorderConfig(simulate=True, output_dir=str(tmp_path / "recordings")),
        db_path=tmp_path / "captures.sqlite3", detection_output_dir=tmp_path / "results", no_ml=True, **kwargs,
    )


def test_recorder_metadata_lands_in_database(tmp_path):
    pr = _simulated(tmp_path)
    row = get_result(tmp_path / "captures.sqlite3", pr.result_id)

    assert row["satellite_name"] == "TEST-SAT"
    assert row["norad_id"] == 12345
    assert row["frequency_hz"] == 137_000_000
    assert row["sample_rate"] == 240_000
    assert row["gain"] == 20
    assert row["recording_status"] == "SUCCESS"
    assert row["simulated"] == 1
    assert row["actual_recording_start"] and row["actual_recording_stop"]
    assert row["output_file_size"] > 0
    assert row["metadata_file_path"].endswith(".json")
    assert row["processing_status"] == "DETECTED"
    assert row["decision_source"] == "rule"  # no ML model in this test
    assert row["iq_retention"] == "kept (policy keep-all)"
    assert Path(row["raw_iq_file_path"]).exists()


def test_failed_recording_is_logged_as_a_row(tmp_path):
    result = RecordingResult(status=RecordingStatus.DEVICE_BUSY, error_message="usb_claim_interface error")
    db = tmp_path / "captures.sqlite3"
    pr = pipeline.process_recording(result, frequency_hz=137e6, sample_rate_hz=240e3, db_path=db, log_failures=True)

    assert pr.success is False and pr.result_id is not None
    row = get_result(db, pr.result_id)
    assert row["recording_status"] == "DEVICE_BUSY"
    assert row["processing_status"] == "NOT_PROCESSED_RECORDING_DEVICE_BUSY"
    assert "usb_claim_interface" in row["notes"]


def test_archive_policy_moves_negative_recording_and_updates_row(tmp_path):
    pr = _simulated(tmp_path, retention_policy="archive-negatives")  # random bytes -> no candidate
    row = get_result(tmp_path / "captures.sqlite3", pr.result_id)

    assert pr.detected is False
    assert row["iq_retention"] == "archived"
    assert Path(row["raw_iq_file_path"]).parent.name == "rejected"
    assert Path(row["raw_iq_file_path"]).exists()
    assert not Path(pr.recording.output_file).exists()
    # The JSON sidecar moves with it, so the archived recording stays self-describing.
    assert Path(row["raw_iq_file_path"]).with_suffix(".json").exists()


def test_delete_policy_removes_negative_recording(tmp_path):
    pr = _simulated(tmp_path, retention_policy="delete-negatives")
    row = get_result(tmp_path / "captures.sqlite3", pr.result_id)
    assert row["iq_retention"] == "deleted"
    assert row["raw_iq_file_path"] is None
    assert not Path(pr.recording.output_file).exists()


def test_ml_confidence_drives_the_decision_when_available():
    rule = SimpleNamespace(detected=False, confidence_score=0.1)
    ml_hi = SimpleNamespace(ml_confidence_score=0.9)
    ml_lo = SimpleNamespace(ml_confidence_score=0.2)
    assert retention.decide(ml_detection=ml_hi, rule_detection=rule).keep is True
    assert retention.decide(ml_detection=ml_hi, rule_detection=rule).source == "ml"
    assert retention.decide(ml_detection=ml_lo, rule_detection=rule, threshold=0.5).keep is False
    assert retention.decide(ml_detection=ml_lo, rule_detection=rule, threshold=0.1).keep is True
    # No ML prediction (MODEL_NOT_AVAILABLE) -> fall back to the rule detector
    no_pred = SimpleNamespace(ml_confidence_score=None)
    d = retention.decide(ml_detection=no_pred, rule_detection=SimpleNamespace(detected=True, confidence_score=0.8))
    assert (d.keep, d.source) == (True, "rule")


def test_positive_recording_is_kept_under_any_policy(tmp_path):
    iq = tmp_path / "rec.iq"
    iq.write_bytes(bytes([128]) * 1000)
    keep = retention.RetentionDecision(True, "ml", 0.9, "confident")
    for policy in retention.POLICIES:
        out = retention.apply(keep, iq, policy=policy)
        assert out.final_path == str(iq) and iq.exists()


def test_unknown_policy_rejected(tmp_path):
    with pytest.raises(ValueError):
        retention.apply(retention.RetentionDecision(False, "rule", 0.0, ""), tmp_path / "x.iq", policy="shred")


def _open_handles_to(path: Path) -> int:
    """How many file descriptors this process holds on ``path`` (Linux only)."""
    fd_dir = Path("/proc/self/fd")
    target = str(path.resolve())
    count = 0
    for fd in fd_dir.iterdir():
        try:
            if os.readlink(fd) == target:
                count += 1
        except OSError:
            pass
    return count


import os  # noqa: E402


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc to inspect open files")
def test_recording_is_not_held_open_after_detection(tmp_path):
    """Regression for Windows WinError 32: the memory map must be released
    before retention tries to move/delete the recording."""
    iq_path = tmp_path / "capture.iq"
    _write_synthetic_iq(iq_path, sample_rate_hz=240_000, duration_s=0.5)
    result = RecordingResult(status=RecordingStatus.SUCCESS, output_file=str(iq_path))
    pipeline.process_recording(result, frequency_hz=137e6, sample_rate_hz=240e3,
                               db_path=tmp_path / "c.sqlite3", output_dir=tmp_path, no_ml=True)
    import gc

    gc.collect()
    assert _open_handles_to(iq_path) == 0


def test_iq_reader_close_releases_the_map(tmp_path):
    from iq_io import IQReader

    path = tmp_path / "x.iq"
    path.write_bytes(bytes(range(256)) * 8)
    with IQReader(path, "cu8") as reader:
        samples = reader[0:100]
    assert samples.size == 100 and reader._raw is None
    path.unlink()  # would raise PermissionError on Windows if the map were still open
