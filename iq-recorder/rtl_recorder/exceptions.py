"""Exception hierarchy for the RTL-SDR recorder.

These exceptions are used internally by the lower-level modules
(``config``, ``utils``, ``device``, ``process``, ``recorder``). The public
entry points on :class:`rtl_recorder.recorder.RTLSDRRecorder`
(``record()``/``record_pass()``) catch these and translate them into a
``RecordingResult`` rather than letting them propagate, so calling code
(eventually a scheduler) never needs a try/except around every call.

Only genuine programming errors (wrong argument types passed by a caller,
bugs) should surface as raw exceptions from the public API.
"""


class RTLSDRError(Exception):
    """Base class for all recorder-related errors."""


class ExecutableNotFoundError(RTLSDRError):
    """``rtl_sdr`` / ``rtl_test`` could not be located on PATH or at the configured path."""


class DeviceNotFoundError(RTLSDRError):
    """No RTL-SDR device is connected/detected."""


class DeviceBusyError(RTLSDRError):
    """The RTL-SDR device is already claimed by another process (or this recorder)."""


class ValidationError(RTLSDRError):
    """A recording parameter (frequency, sample rate, gain, duration, ...) is invalid."""

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(message)


class OutputDirectoryError(RTLSDRError):
    """The output directory is unavailable, not writable, or has insufficient disk space."""


class RecordingProcessError(RTLSDRError):
    """The ``rtl_sdr`` process failed to start, crashed, or otherwise misbehaved."""


class RecordingCancelledError(RTLSDRError):
    """Raised internally to unwind a recording that was cancelled mid-flight."""
