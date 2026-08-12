"""Tests for RTLSDRRecorder: state machine, simulation, real process control
(via a fake rtl_sdr executable), cancellation, and error handling.

The fake rtl_sdr fixture (tests/fixtures/fake_rtl_sdr.py) lets the "real"
(non-simulated) code path - process launch, stderr parsing, graceful
stop via SIGINT, crash detection - be exercised without real hardware.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from rtl_recorder.config import RecorderConfig
from rtl_recorder.recorder import RTLSDRRecorder
from rtl_recorder.states import RecorderState, RecordingStatus

FAKE_RTL_SDR = str(Path(__file__).parent / "fixtures" / "fake_rtl_sdr.py")


def _fast_config(tmp_path, rtl_sdr_path=None, simulate=False, **overrides) -> RecorderConfig:
    kwargs = dict(
        output_dir=str(tmp_path),
        rtl_sdr_path=rtl_sdr_path,
        simulate=simulate,
        startup_grace_seconds=0.2,
        monitor_interval_seconds=0.1,
        stop_grace_seconds=2.0,
    )
    kwargs.update(overrides)
    return RecorderConfig(**kwargs)


@pytest.fixture(autouse=True)
def _clean_fake_rtlsdr_mode(monkeypatch):
    monkeypatch.delenv("FAKE_RTLSDR_MODE", raising=False)
    yield


def _wait_for_state(recorder, state, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if recorder.check_recording_status() == state:
            return True
        time.sleep(0.02)
    return False


# --------------------------------------------------------------------- #
# Simulation mode
# --------------------------------------------------------------------- #

def test_simulated_recording_succeeds(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, simulate=True))
    assert recorder.check_recording_status() == RecorderState.IDLE

    result = recorder.record(
        satellite_name="KISS92-TEST", frequency_hz=92_000_000, sample_rate=2_400_000, gain=30, duration=1,
    )

    assert result.status == RecordingStatus.SUCCESS
    assert recorder.check_recording_status() == RecorderState.COMPLETED
    assert Path(result.output_file).exists()
    assert Path(result.output_file).stat().st_size > 0
    assert Path(result.metadata_file).exists()

    meta = json.loads(Path(result.metadata_file).read_text())
    assert meta["simulated"] is True
    assert meta["recording_status"] == "SUCCESS"
    assert meta["satellite_name"] == "KISS92-TEST"
    assert meta["frequency_hz"] == 92_000_000


def test_simulated_recording_marks_metadata_as_simulated(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, simulate=True))
    result = recorder.record(satellite_name="X", frequency_hz=100_000_000, sample_rate=2_400_000, gain=None, duration=1)
    assert result.metadata.simulated is True


def test_simulate_mode_skips_device_check(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, simulate=True))
    check = recorder.check_device()
    assert check.available is True


# --------------------------------------------------------------------- #
# State transitions
# --------------------------------------------------------------------- #

def test_state_transition_sequence_on_success(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, simulate=True))
    transitions = []
    original = recorder._set_state

    def spy(new_state, reason=""):
        transitions.append(new_state)
        original(new_state, reason)

    recorder._set_state = spy
    result = recorder.record(satellite_name="X", frequency_hz=100_000_000, sample_rate=2_400_000, gain=None, duration=1)

    assert result.status == RecordingStatus.SUCCESS
    assert transitions == [
        RecorderState.STARTING,
        RecorderState.RECORDING,
        RecorderState.STOPPING,
        RecorderState.COMPLETED,
    ]


def test_idle_state_before_any_recording(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, simulate=True))
    assert recorder.check_recording_status() == RecorderState.IDLE


# --------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------- #

def test_cancel_recording_from_another_thread(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, simulate=True))
    results = {}

    def run():
        results["record"] = recorder.record(
            satellite_name="CANCEL-TEST", frequency_hz=100_000_000, sample_rate=2_400_000, gain=None, duration=10,
        )

    thread = threading.Thread(target=run)
    thread.start()
    assert _wait_for_state(recorder, RecorderState.RECORDING), "Recorder never reached RECORDING"

    cancel_result = recorder.cancel_recording()
    thread.join(timeout=10)

    assert cancel_result.status == RecordingStatus.CANCELLED
    assert results["record"].status == RecordingStatus.CANCELLED
    assert recorder.check_recording_status() == RecorderState.CANCELLED
    # A cancelled recording still leaves diagnostic metadata behind.
    assert cancel_result.metadata_file is not None
    assert json.loads(Path(cancel_result.metadata_file).read_text())["recording_status"] == "CANCELLED"


def test_cancel_with_no_active_recording_is_a_no_op(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, simulate=True))
    result = recorder.cancel_recording()
    assert result.status == RecordingStatus.FAILED
    assert recorder.check_recording_status() == RecorderState.IDLE


# --------------------------------------------------------------------- #
# Real (subprocess-driven) recording, via the fake rtl_sdr
# --------------------------------------------------------------------- #

def test_real_recording_success_via_fake_rtl_sdr(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, rtl_sdr_path=FAKE_RTL_SDR))

    result = recorder.record(
        satellite_name="KISS92", frequency_hz=92_000_000, sample_rate=250_000, gain=30, duration=1.5,
    )

    assert result.status == RecordingStatus.SUCCESS
    assert result.output_file_size > 0
    assert recorder.check_recording_status() == RecorderState.COMPLETED


def test_device_busy_reported_distinctly(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_RTLSDR_MODE", "busy")
    recorder = RTLSDRRecorder(_fast_config(tmp_path, rtl_sdr_path=FAKE_RTL_SDR))

    result = recorder.record(
        satellite_name="X", frequency_hz=100_000_000, sample_rate=2_400_000, gain=30, duration=5,
    )

    assert result.status == RecordingStatus.DEVICE_BUSY
    assert "busy" in result.error_message.lower()


def test_device_not_found_reported_as_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_RTLSDR_MODE", "not_found")
    recorder = RTLSDRRecorder(_fast_config(tmp_path, rtl_sdr_path=FAKE_RTL_SDR))

    result = recorder.record(
        satellite_name="X", frequency_hz=100_000_000, sample_rate=2_400_000, gain=30, duration=5,
    )

    assert result.status == RecordingStatus.FAILED
    assert "not found" in result.error_message.lower()


def test_crash_mid_recording_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_RTLSDR_MODE", "crash")
    recorder = RTLSDRRecorder(_fast_config(tmp_path, rtl_sdr_path=FAKE_RTL_SDR))

    result = recorder.record(
        satellite_name="X", frequency_hz=100_000_000, sample_rate=2_400_000, gain=30, duration=5,
    )

    assert result.status == RecordingStatus.FAILED
    assert "unexpectedly" in result.error_message.lower()


def test_second_recording_after_completion_succeeds(tmp_path):
    """Mirrors hardware Test 5: the device must be released for a repeat recording."""
    recorder = RTLSDRRecorder(_fast_config(tmp_path, rtl_sdr_path=FAKE_RTL_SDR))

    first = recorder.record(satellite_name="KISS92", frequency_hz=92_000_000, sample_rate=250_000, gain=30, duration=1)
    assert first.status == RecordingStatus.SUCCESS

    second = recorder.record(satellite_name="KISS92", frequency_hz=92_000_000, sample_rate=250_000, gain=30, duration=1)
    assert second.status == RecordingStatus.SUCCESS
    assert second.output_file != first.output_file  # unique filename, no overwrite


# --------------------------------------------------------------------- #
# Missing executable
# --------------------------------------------------------------------- #

def test_missing_rtl_sdr_executable_reported_as_failed(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, rtl_sdr_path="/nonexistent/rtl_sdr_binary_xyz"))

    result = recorder.record(
        satellite_name="X", frequency_hz=100_000_000, sample_rate=2_400_000, gain=30, duration=5,
    )

    assert result.status == RecordingStatus.FAILED
    assert "not found" in result.error_message.lower()
    assert recorder.check_recording_status() == RecorderState.FAILED


# --------------------------------------------------------------------- #
# Validation failures never report SUCCESS
# --------------------------------------------------------------------- #

def test_invalid_frequency_fails_cleanly(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, simulate=True))
    result = recorder.record(satellite_name="X", frequency_hz=1, sample_rate=2_400_000, gain=30, duration=5)
    assert result.status == RecordingStatus.FAILED
    assert result.error_message is not None


def test_invalid_duration_fails_cleanly(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, simulate=True))
    result = recorder.record(satellite_name="X", frequency_hz=100_000_000, sample_rate=2_400_000, gain=30, duration=-5)
    assert result.status == RecordingStatus.FAILED


# --------------------------------------------------------------------- #
# Device busy at the recorder-instance level (not the OS level)
# --------------------------------------------------------------------- #

def test_concurrent_record_calls_report_device_busy(tmp_path):
    recorder = RTLSDRRecorder(_fast_config(tmp_path, simulate=True))
    results = {}

    def run():
        results["first"] = recorder.record(
            satellite_name="FIRST", frequency_hz=100_000_000, sample_rate=2_400_000, gain=None, duration=2,
        )

    thread = threading.Thread(target=run)
    thread.start()
    assert _wait_for_state(recorder, RecorderState.RECORDING)

    second_result = recorder.record(
        satellite_name="SECOND", frequency_hz=100_000_000, sample_rate=2_400_000, gain=None, duration=1,
    )
    assert second_result.status == RecordingStatus.DEVICE_BUSY

    recorder.cancel_recording()
    thread.join(timeout=5)
