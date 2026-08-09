"""Small, dependency-free helper functions shared across the package."""

from __future__ import annotations

import platform
import shutil
from pathlib import Path
from typing import Optional

from .exceptions import ExecutableNotFoundError

#: rtl_sdr's default output format is interleaved 8-bit unsigned I/Q samples
#: (one byte for I, one byte for Q), with no file header. This is used both
#: to pre-flight-check disk space and to sanity-check finished recordings.
BYTES_PER_IQ_SAMPLE = 2


def is_windows() -> bool:
    """Return True when running on Windows."""
    return platform.system() == "Windows"


def find_executable(name: str, explicit_path: Optional[str] = None) -> str:
    """Locate an RTL-SDR command-line tool (e.g. ``rtl_sdr``, ``rtl_test``).

    Resolution order:
      1. ``explicit_path`` if given (must exist).
      2. ``name`` on PATH via :func:`shutil.which`.
      3. ``name.exe`` on PATH (Windows convenience, in case the caller
         passed the bare name without the extension).

    Raises :class:`ExecutableNotFoundError` if none of the above resolve.
    """
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return str(path)
        raise ExecutableNotFoundError(
            f"'{name}' not found at configured path '{explicit_path}'. "
            "Check RecorderConfig.rtl_sdr_path / rtl_test_path."
        )

    found = shutil.which(name)
    if found:
        return found

    if is_windows() and not name.lower().endswith(".exe"):
        found = shutil.which(name + ".exe")
        if found:
            return found

    raise ExecutableNotFoundError(
        f"'{name}' executable not found on PATH. Install the RTL-SDR command-line "
        "tools (see README) and/or set an explicit path via RecorderConfig."
    )


def expected_iq_file_size(sample_rate: float, duration_seconds: float,
                           bytes_per_sample: int = BYTES_PER_IQ_SAMPLE) -> int:
    """Approximate expected raw IQ file size in bytes for a given capture."""
    return int(sample_rate * bytes_per_sample * duration_seconds)


def check_disk_space(directory, required_bytes: int, margin: float = 1.05) -> bool:
    """Return True if ``directory``'s filesystem has enough free space.

    ``margin`` adds a safety buffer on top of ``required_bytes`` (default 5%).
    """
    usage = shutil.disk_usage(directory)
    return usage.free >= required_bytes * margin
