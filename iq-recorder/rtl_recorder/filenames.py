"""Human-readable, collision-safe filename generation for IQ recordings."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Tuple

_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def sanitize_satellite_name(name: str) -> str:
    """Make a satellite/target name safe to use as (part of) a filename.

    Anything that isn't alphanumeric, ``_`` or ``-`` is replaced with ``_``;
    repeated underscores are collapsed. Falls back to "UNKNOWN" for an
    empty/entirely-invalid name so callers never end up with a blank
    filename component.
    """
    name = (name or "").strip()
    name = _SAFE_CHARS_RE.sub("_", name)
    name = _MULTI_UNDERSCORE_RE.sub("_", name).strip("_")
    return name or "UNKNOWN"


def build_base_filename(satellite_name: str, timestamp: datetime, frequency_hz: int) -> str:
    """Build the base filename (no extension), e.g. ``METEOR-M2-4_20260809_193430_137900000Hz``."""
    safe_name = sanitize_satellite_name(satellite_name)
    ts = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{safe_name}_{ts}_{int(frequency_hz)}Hz"


def unique_output_paths(output_dir: Path, base_filename: str,
                         iq_ext: str = ".iq", meta_ext: str = ".json") -> Tuple[Path, Path]:
    """Return a (iq_path, metadata_path) pair guaranteed not to already exist.

    On collision (same satellite/timestamp/frequency already recorded, or a
    metadata file already present), an incrementing numeric suffix is
    appended until a free pair of filenames is found. An existing recording
    is never overwritten.
    """
    output_dir = Path(output_dir)
    candidate = base_filename
    suffix = 1
    while True:
        iq_path = output_dir / f"{candidate}{iq_ext}"
        meta_path = output_dir / f"{candidate}{meta_ext}"
        if not iq_path.exists() and not meta_path.exists():
            return iq_path, meta_path
        suffix += 1
        candidate = f"{base_filename}_{suffix}"
