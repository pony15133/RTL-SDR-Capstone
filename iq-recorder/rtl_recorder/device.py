"""RTL-SDR device presence/availability check, using ``rtl_test``.

``rtl_test`` without arguments streams samples and tests for dropped bytes
forever until interrupted. Run with ``-t`` it instead probes the device,
tuner and supported gain values, then exits - which is what we want for a
quick "is a device present and free" check. A subprocess timeout is still
applied as a safety net in case a particular build/driver combination
doesn't exit promptly.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

from .exceptions import ExecutableNotFoundError
from .utils import find_executable

logger = logging.getLogger(__name__)

_BUSY_MARKERS = ("usb_claim_interface error", "usb_open error", "error claiming", "resource busy")
_NOT_FOUND_MARKERS = ("no supported devices found",)
_FOUND_MARKERS = ("found 1 device", "found device", "detached kernel driver")


@dataclass
class DeviceCheckResult:
    available: bool
    message: str
    raw_output: str = ""


def check_device(rtl_test_path=None, timeout: float = 10.0) -> DeviceCheckResult:
    """Probe for a connected, available RTL-SDR device via ``rtl_test -t``."""
    try:
        exe = find_executable("rtl_test", rtl_test_path)
    except ExecutableNotFoundError as exc:
        return DeviceCheckResult(False, str(exc), "")

    try:
        proc = subprocess.run([exe, "-t"], capture_output=True, text=True, timeout=timeout)
        output = (proc.stdout or "") + (proc.stderr or "")
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        return DeviceCheckResult(
            False, f"'{exe} -t' did not exit within {timeout}s - device may be unresponsive", ""
        )
    except OSError as exc:
        return DeviceCheckResult(False, f"Failed to run rtl_test: {exc}", "")

    lowered = output.lower()
    if any(marker in lowered for marker in _NOT_FOUND_MARKERS):
        return DeviceCheckResult(False, "No RTL-SDR device found", output)
    if any(marker in lowered for marker in _BUSY_MARKERS):
        return DeviceCheckResult(False, "RTL-SDR device found but busy/in use by another process", output)
    if any(marker in lowered for marker in _FOUND_MARKERS) or returncode == 0:
        return DeviceCheckResult(True, "RTL-SDR device detected", output)
    return DeviceCheckResult(False, f"rtl_test exited with code {returncode} and an unrecognised output", output)
