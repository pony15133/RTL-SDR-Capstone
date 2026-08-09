"""Explicit state and result enums for the RTL-SDR recorder.

Two separate enums are used on purpose:

- ``RecorderState`` describes where the recorder is *right now* in its
  lifecycle (useful for ``check_recording_status()`` and for logging state
  transitions).
- ``RecordingStatus`` describes the *outcome* of a finished (or aborted)
  recording attempt, returned to the caller as part of ``RecordingResult``.

Keeping them separate avoids overloading a single enum with both "what is
happening" and "what happened", which tends to turn into a pile of loosely
related boolean flags.
"""

from enum import Enum


class RecorderState(Enum):
    """Lifecycle state of an :class:`rtl_recorder.recorder.RTLSDRRecorder`."""

    IDLE = "IDLE"
    WAITING = "WAITING"
    STARTING = "STARTING"
    RECORDING = "RECORDING"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RecordingStatus(Enum):
    """Outcome of a recording attempt, returned in ``RecordingResult``."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DEVICE_BUSY = "DEVICE_BUSY"
