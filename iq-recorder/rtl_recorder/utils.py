"""Small, dependency-free helper functions shared across the package."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import List, Optional

from .exceptions import ExecutableNotFoundError

#: rtl_sdr's default output format is interleaved 8-bit unsigned I/Q samples
#: (one byte for I, one byte for Q), with no file header. This is used both
#: to pre-flight-check disk space and to sanity-check finished recordings.
BYTES_PER_IQ_SAMPLE = 2


def is_windows() -> bool:
    """Return True when running on Windows."""
    return platform.system() == "Windows"


def platform_name() -> str:
    """'Windows', 'macOS' or 'Linux' (anything else is reported as-is)."""
    system = platform.system()
    return {"Darwin": "macOS"}.get(system, system)


def candidate_install_dirs() -> List[Path]:
    """Folders where the RTL-SDR tools are usually installed on this OS,
    searched after PATH. ``RTL_SDR_HOME`` (a folder) is always tried first.

    Windows has no package manager default, so people unzip the osmocom
    release somewhere - the common spots are covered, including a
    ``rtl-sdr*`` folder in Downloads or on C:\\.
    """
    dirs: List[Path] = []
    home_override = os.environ.get("RTL_SDR_HOME")
    if home_override:
        dirs += [Path(home_override), Path(home_override) / "bin", Path(home_override) / "x64"]
    system = platform_name()
    if system == "Windows":
        user = Path(os.environ.get("USERPROFILE", str(Path.home())))
        roots = [Path("C:/"), Path(os.environ.get("ProgramFiles", "C:/Program Files")),
                 Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")), user, user / "Downloads"]
        for root in roots:
            for match in sorted(root.glob("rtl-sdr*")) if root.exists() else []:
                dirs += [match, match / "x64", match / "bin"]
        dirs += [Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "PothosSDR" / "bin"]
    elif system == "macOS":
        dirs += [Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path("/opt/local/bin")]
    else:
        dirs += [Path("/usr/bin"), Path("/usr/local/bin"), Path("/snap/bin"), Path.home() / ".local" / "bin"]
    return dirs


def install_hint() -> str:
    """How to install the RTL-SDR command-line tools on the current OS."""
    system = platform_name()
    if system == "Windows":
        return ("Windows: download the rtl-sdr release zip (https://ftp.osmocom.org/binaries/windows/rtl-sdr/), "
                "unzip it (e.g. to C:\\rtl-sdr), install the WinUSB driver for the dongle with Zadig, then either add "
                "that folder to PATH or set RTL_SDR_HOME to it.")
    if system == "macOS":
        return "macOS: brew install librtlsdr   (provides rtl_sdr and rtl_test)"
    return ("Linux: sudo apt install rtl-sdr   (Debian/Ubuntu; 'dnf install rtl-sdr' on Fedora). If the device "
            "is claimed by the DVB driver, blacklist dvb_usb_rtl28xxu and replug.")


def find_executable(name: str, explicit_path: Optional[str] = None) -> str:
    """Locate an RTL-SDR command-line tool (e.g. ``rtl_sdr``, ``rtl_test``)
    on Windows, macOS or Linux - never a hard-coded ``.exe`` path.

    Resolution order:
      1. ``explicit_path`` if given (must exist).
      2. ``name`` on PATH via :func:`shutil.which` (which also tries
         ``name.exe`` on Windows via PATHEXT).
      3. The usual install folders for this OS (:func:`candidate_install_dirs`),
         including ``$RTL_SDR_HOME``.

    Raises :class:`ExecutableNotFoundError` with OS-specific install
    instructions if none of the above resolve.
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

    names = [name]
    if is_windows() and not name.lower().endswith(".exe"):
        names.append(name + ".exe")
    for folder in candidate_install_dirs():
        for candidate in names:
            path = folder / candidate
            if path.is_file() and (is_windows() or os.access(path, os.X_OK)):
                return str(path)

    raise ExecutableNotFoundError(
        f"'{name}' was not found on PATH or in the usual {platform_name()} install folders. "
        f"{install_hint()} Or pass an explicit path (--rtl-sdr-path / --rtl-test-path)."
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
