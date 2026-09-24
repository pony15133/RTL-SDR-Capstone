"""Safety for long unattended runs of the station (used by auto_capture.py).

* Disk space: before each pass, check there is room for the recording
  (sample rate x 2 bytes x window) plus a safety margin. If not, free space
  by deleting the OLDEST files in <output_dir>/rejected/ (recordings the
  model already judged "no satellite"); never touches kept or uncertain
  recordings. Still not enough -> the pass is skipped and logged.
* Keep awake: stops Windows from sleeping while the station runs (no
  settings changed; ends with the program). macOS/Linux: run_station.sh
  wraps the station in caffeinate / systemd-inhibit.
* Daily log files: logs/station_YYYY-MM-DD.log, the last 30 kept.
* Retry: a recording that fails because the dongle was busy or unplugged is
  retried (after a pause) for whatever is left of the pass.
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2  # rtl_sdr cu8: one byte I + one byte Q
DEFAULT_MIN_FREE_GB = 2.0
DEFAULT_MARGIN = 1.2


@dataclass
class DiskCheck:
    ok: bool
    needed_bytes: int
    free_bytes: int
    freed_bytes: int = 0
    deleted: Optional[List[str]] = None
    message: str = ""


def recording_bytes(sample_rate: float, seconds: float) -> int:
    return int(sample_rate * BYTES_PER_SAMPLE * max(0.0, seconds))


def free_bytes(path) -> int:
    path = Path(path)
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def prune_rejected(output_dir, bytes_to_free: int) -> tuple:
    """Delete the oldest recordings in <output_dir>/rejected/ until
    ``bytes_to_free`` is reached. Returns (freed_bytes, [deleted paths])."""
    rejected = Path(output_dir) / "rejected"
    if not rejected.is_dir():
        return 0, []
    files = sorted((p for p in rejected.iterdir() if p.is_file() and p.suffix.lower() != ".json"),
                   key=lambda p: p.stat().st_mtime)
    freed, deleted = 0, []
    for f in files:
        if freed >= bytes_to_free:
            break
        size = f.stat().st_size
        try:
            f.unlink()
        except OSError as exc:
            logger.warning("Could not delete %s: %s", f, exc)
            continue
        freed += size
        deleted.append(str(f))
    return freed, deleted


def ensure_space(output_dir, sample_rate: float, seconds: float, *, min_free_gb: float = DEFAULT_MIN_FREE_GB,
                 margin: float = DEFAULT_MARGIN, prune: bool = True, free_fn=free_bytes) -> DiskCheck:
    """Is there room for this recording? Frees rejected recordings if needed."""
    needed = int(recording_bytes(sample_rate, seconds) * margin + min_free_gb * 1e9)
    free = free_fn(output_dir)
    if free >= needed:
        return DiskCheck(True, needed, free, message=f"{_gb(free)} free, {_gb(needed)} needed")
    freed, deleted = prune_rejected(output_dir, needed - free) if prune else (0, [])
    free = free_fn(output_dir)
    ok = free >= needed
    msg = (f"{_gb(free)} free, {_gb(needed)} needed"
           + (f"; deleted {len(deleted)} rejected recording(s) ({_gb(freed)})" if deleted else "")
           + ("" if ok else " - NOT ENOUGH SPACE: pass skipped. Free disk space or lower sample_rate."))
    return DiskCheck(ok, needed, free, freed, deleted, msg)


def keep_awake(enable: bool = True) -> bool:
    """Windows only: ask the OS not to sleep while this process runs."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:  # never fatal
        return False


def setup_daily_log(log_dir="logs", *, keep_days: int = 30, level=logging.INFO) -> Path:
    """Also write the station's log to logs/station.log, rotated at midnight
    into station.log.YYYY-MM-DD (the last ``keep_days`` kept)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "station.log"
    root = logging.getLogger()
    if not any(isinstance(h, TimedRotatingFileHandler) and Path(h.baseFilename) == path.resolve()
               for h in root.handlers):
        handler = TimedRotatingFileHandler(path, when="midnight", backupCount=keep_days, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.setLevel(level)
        root.addHandler(handler)
    return path


#: Recorder outcomes worth another try while the pass is still overhead
#: (dongle busy / unplugged / rtl_sdr crashed). CANCELLED is a deliberate stop.
RETRYABLE = {"DEVICE_BUSY", "FAILED"}
