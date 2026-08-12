"""rtl_recorder: the RTL-SDR Automatic IQ Recording Manager.

Public interface:

    from rtl_recorder import RTLSDRRecorder, RecorderConfig

    recorder = RTLSDRRecorder(RecorderConfig(output_dir="recordings"))
    result = recorder.record(
        satellite_name="METEOR-M2-4",
        norad_id=40069,
        frequency_hz=137_900_000,
        sample_rate=2_400_000,
        gain=30,
        duration=60,
    )

See README.md for the full interface, CLI usage, and integration notes.
"""

from .config import RecorderConfig
from .device import DeviceCheckResult
from .exceptions import (
    DeviceBusyError,
    DeviceNotFoundError,
    ExecutableNotFoundError,
    OutputDirectoryError,
    RecordingCancelledError,
    RecordingProcessError,
    RTLSDRError,
    ValidationError,
)
from .metadata import RecordingMetadata, RecordingResult
from .recorder import RTLSDRRecorder
from .states import RecorderState, RecordingStatus

__all__ = [
    "RTLSDRRecorder",
    "RecorderConfig",
    "RecorderState",
    "RecordingStatus",
    "RecordingMetadata",
    "RecordingResult",
    "DeviceCheckResult",
    "RTLSDRError",
    "ValidationError",
    "ExecutableNotFoundError",
    "DeviceNotFoundError",
    "DeviceBusyError",
    "OutputDirectoryError",
    "RecordingProcessError",
    "RecordingCancelledError",
]

__version__ = "0.1.0"
