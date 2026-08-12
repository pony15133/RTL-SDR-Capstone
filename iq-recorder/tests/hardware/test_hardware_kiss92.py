"""Phase 1 hardware validation: RTL-SDR Blog V3 + KISS92 Singapore FM (92.0 MHz).

These tests talk to REAL RTL-SDR hardware and are excluded from the default
`pytest` run (see pytest.ini: `addopts = -m "not hardware"`). Run them
explicitly, with an RTL-SDR V3 plugged in, from the ``iq-recorder`` directory:

    pytest -m hardware -v tests/hardware/test_hardware_kiss92.py

Optional environment variables:
    RTL_SDR_PATH   - explicit path to rtl_sdr(.exe) if not on PATH
    RTL_TEST_PATH  - explicit path to rtl_test(.exe) if not on PATH
    HARDWARE_TEST_OUTPUT_DIR - where to write the captures (default: ./recordings/hardware_test)

KISS92 (92.0 MHz) is only used here as a strong, convenient real-world FM
signal to validate the recording pipeline end-to-end (device detection ->
tune -> capture -> auto-stop -> file/metadata verification -> device
release -> repeat). Nothing in the recorder is FM-specific; the same
pipeline is exercised later against the 137 MHz weather-satellite band.
"""

import os
import time
from pathlib import Path

import pytest

from rtl_recorder.config import RecorderConfig
from rtl_recorder.recorder import RTLSDRRecorder
from rtl_recorder.states import RecordingStatus
from rtl_recorder.utils import expected_iq_file_size

pytestmark = pytest.mark.hardware

KISS92_FREQUENCY_HZ = 92_000_000
TEST_SAMPLE_RATE = 2_400_000
# Not permanently hard-coded: override via the TEST_GAIN env var if 30 dB
# clips/underdrives your particular dongle + antenna combination.
TEST_GAIN_DB = float(os.environ.get("TEST_GAIN", "30"))
TEST_DURATION_SECONDS = 30


@pytest.fixture(scope="module")
def output_dir():
    path = Path(os.environ.get("HARDWARE_TEST_OUTPUT_DIR", "recordings/hardware_test"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def recorder(output_dir):
    config = RecorderConfig(
        output_dir=str(output_dir),
        rtl_sdr_path=os.environ.get("RTL_SDR_PATH"),
        rtl_test_path=os.environ.get("RTL_TEST_PATH"),
        simulate=False,
    )
    return RTLSDRRecorder(config)


def test_01_device_detected(recorder):
    """Test 1 - device detection. Stops the hardware suite cleanly if no device is found."""
    check = recorder.check_device()
    if not check.available:
        pytest.fail(
            f"RTL-SDR not detected - aborting hardware tests. rtl_test output:\n{check.raw_output}"
        )
    assert check.available is True


def test_02_to_05_kiss92_capture_stop_and_size_check(recorder, output_dir):
    """Tests 2-4: capture KISS92, confirm auto-stop, verify file + metadata + size sanity."""
    result = recorder.record(
        satellite_name="KISS92-SINGAPORE",
        norad_id=None,
        frequency_hz=KISS92_FREQUENCY_HZ,
        sample_rate=TEST_SAMPLE_RATE,
        gain=TEST_GAIN_DB,
        duration=TEST_DURATION_SECONDS,
    )

    # Test 3: automatic stop + SUCCESS status (never reported as success if it wasn't).
    assert result.status == RecordingStatus.SUCCESS, f"Recording did not succeed: {result.error_message}"

    # Output file exists and is non-empty.
    output_path = Path(result.output_file)
    assert output_path.exists(), "IQ output file was not created"
    actual_size = output_path.stat().st_size
    assert actual_size > 0, "IQ output file is empty"

    # Test 4: file-size sanity check against sample_rate * bytes_per_sample * duration.
    expected_size = expected_iq_file_size(TEST_SAMPLE_RATE, result.recording_duration_seconds or TEST_DURATION_SECONDS)
    print(f"Expected size: ~{expected_size} bytes, actual size: {actual_size} bytes")
    ratio = actual_size / expected_size
    assert ratio > 0.5, (
        f"Captured file is far smaller than expected ({ratio*100:.1f}% of ~{expected_size} bytes) "
        "- capture likely failed partway through"
    )

    # Metadata sidecar exists and records actual timestamps + status.
    assert result.metadata_file is not None
    meta = result.metadata
    assert meta.recording_status == "SUCCESS"
    assert meta.actual_recording_start is not None
    assert meta.actual_recording_stop is not None
    assert meta.simulated is False
    print(f"Recording written to: {output_path}")
    print(f"Metadata written to: {result.metadata_file}")


def test_06_second_recording_succeeds_after_device_release(recorder):
    """Test 5 - the device must be released so a second, independent recording can start."""
    time.sleep(5)  # brief pause, as specified

    result = recorder.record(
        satellite_name="KISS92-SINGAPORE",
        norad_id=None,
        frequency_hz=KISS92_FREQUENCY_HZ,
        sample_rate=TEST_SAMPLE_RATE,
        gain=TEST_GAIN_DB,
        duration=10,
    )

    assert result.status == RecordingStatus.SUCCESS, (
        f"Second recording failed - device may not have been released: {result.error_message}"
    )
    assert Path(result.output_file).stat().st_size > 0


# --------------------------------------------------------------------- #
# Test 6 (optional playback/verification) is manual - not automatable
# without adding an audio-demod dependency to the recorder itself, which
# would defeat the point of a modulation-agnostic raw-IQ recorder. See
# README.md "Verifying a capture manually" for suggested tools
# (e.g. inspectrum, GQRX, or `rtl_fm`/SDR++ opening the same .iq file).
# --------------------------------------------------------------------- #
