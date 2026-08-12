"""Recorder configuration and parameter validation.

Validation ranges here are intentionally documented with their rationale
(see README "Known limitations") rather than treated as arbitrary magic
numbers, since they come from real RTL2832U/R820T2 hardware constraints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .exceptions import OutputDirectoryError, ValidationError

#: RTL-SDR Blog V3 (R820T2 tuner) practical tuning range. Below this, direct
#: sampling mode would be needed (not implemented here - see README).
FREQUENCY_MIN_HZ = 24_000_000
FREQUENCY_MAX_HZ = 1_766_000_000

#: The RTL2832U cannot sample at arbitrary rates - there is a documented gap
#: between ~300 kS/s and ~900 kS/s where the demodulator is unreliable.
SAMPLE_RATE_RANGES = ((225_001, 300_000), (900_001, 3_200_000))

#: Tuner gain in dB. rtl_sdr snaps this to the nearest value the tuner
#: actually supports (a discrete, tuner-specific list) - we only validate
#: a sane outer range here and record whatever rtl_sdr reports it applied.
GAIN_MIN_DB = 0.0
GAIN_MAX_DB = 50.0

#: Sane outer bounds for a recording duration, mostly to catch obvious
#: mistakes (e.g. a duration accidentally given in milliseconds).
DURATION_MIN_SECONDS = 1
DURATION_MAX_SECONDS = 24 * 3600


@dataclass
class RecorderConfig:
    """Configuration shared across recordings made by one RTLSDRRecorder."""

    #: Directory recordings and metadata sidecars are written to.
    output_dir: str = "recordings"
    #: Explicit path to the rtl_sdr executable; None = discover via PATH.
    rtl_sdr_path: Optional[str] = None
    #: Explicit path to the rtl_test executable; None = discover via PATH.
    rtl_test_path: Optional[str] = None
    #: When True, no real hardware is touched - see rtl_recorder/recorder.py.
    simulate: bool = False
    #: RTL-SDR device index, passed to rtl_sdr -d (for multi-dongle setups).
    device_index: int = 0
    #: How long to wait for a graceful stop before escalating to terminate/kill.
    stop_grace_seconds: float = 5.0
    #: How often the recording wait loop polls for stop/cancel/crash conditions.
    monitor_interval_seconds: float = 0.5
    #: How long to wait after launching rtl_sdr before confirming it's alive.
    startup_grace_seconds: float = 1.5
    #: Fraction below the expected file size still considered a valid recording.
    file_size_tolerance: float = 0.10
    #: Safety margin applied on top of the expected size during the disk-space check.
    disk_space_margin: float = 1.05
    #: Cap on the dummy file size written in simulation mode (avoids writing
    #: multi-GB placeholder files for long simulated durations).
    simulate_max_bytes: int = 65536

    def resolve_output_dir(self) -> Path:
        """Ensure the configured output directory exists and is writable."""
        path = Path(self.output_dir)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OutputDirectoryError(
                f"Cannot create/access output directory '{path}': {exc}"
            ) from exc
        if not os.access(path, os.W_OK):
            raise OutputDirectoryError(f"Output directory '{path}' is not writable")
        return path


def validate_frequency(frequency_hz) -> int:
    try:
        freq = int(frequency_hz)
    except (TypeError, ValueError):
        raise ValidationError("frequency_hz", f"frequency_hz must be an integer, got {frequency_hz!r}")
    if not (FREQUENCY_MIN_HZ <= freq <= FREQUENCY_MAX_HZ):
        raise ValidationError(
            "frequency_hz",
            f"frequency_hz {freq} Hz is outside the supported range "
            f"[{FREQUENCY_MIN_HZ}, {FREQUENCY_MAX_HZ}] Hz for an RTL-SDR Blog V3",
        )
    return freq


def validate_sample_rate(sample_rate) -> int:
    try:
        rate = int(sample_rate)
    except (TypeError, ValueError):
        raise ValidationError("sample_rate", f"sample_rate must be an integer, got {sample_rate!r}")
    if not any(lo <= rate <= hi for lo, hi in SAMPLE_RATE_RANGES):
        ranges = " or ".join(f"[{lo}, {hi}]" for lo, hi in SAMPLE_RATE_RANGES)
        raise ValidationError(
            "sample_rate",
            f"sample_rate {rate} samples/sec is not in a supported RTL2832U range: {ranges}",
        )
    return rate


def validate_gain(gain) -> Optional[float]:
    """None (or the string "auto") means AGC / automatic gain - no -g flag is passed."""
    if gain is None:
        return None
    if isinstance(gain, str) and gain.strip().lower() == "auto":
        return None
    try:
        value = float(gain)
    except (TypeError, ValueError):
        raise ValidationError("gain", f"gain must be numeric or 'auto', got {gain!r}")
    if not (GAIN_MIN_DB <= value <= GAIN_MAX_DB):
        raise ValidationError(
            "gain", f"gain {value} dB is outside the supported range [{GAIN_MIN_DB}, {GAIN_MAX_DB}] dB"
        )
    return value


def validate_duration(duration) -> float:
    try:
        value = float(duration)
    except (TypeError, ValueError):
        raise ValidationError("duration", f"duration must be numeric, got {duration!r}")
    if not (DURATION_MIN_SECONDS <= value <= DURATION_MAX_SECONDS):
        raise ValidationError(
            "duration",
            f"duration {value}s is outside the sane range "
            f"[{DURATION_MIN_SECONDS}, {DURATION_MAX_SECONDS}] seconds",
        )
    return value
