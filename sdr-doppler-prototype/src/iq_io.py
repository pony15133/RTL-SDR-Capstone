"""Reading raw IQ recordings in every format this project meets, plus
best-effort discovery of their sample rate / centre frequency.

Formats (``iq_format``):

    cu8        interleaved unsigned 8-bit I/Q, offset 127.5 - rtl_sdr's native
               output (what iq-recorder writes as ``.iq``)
    ci16       interleaved little-endian int16 I/Q - SigMF ``ci16_le``,
               CAMRAS ``.raw`` snapshots
    complex64  interleaved float32 I/Q - GNU Radio, LoRadar ``.bin``, numpy
    wav        SDR#/HDSDR IQ WAV: stereo int16 (or float) with I=left, Q=right

Everything is returned as complex64 scaled to roughly [-1, 1], and large
files are memory-mapped so a multi-GB capture is never loaded whole unless
the caller slices it.

Metadata is looked for, in order: a SigMF ``.sigmf-meta`` next to the data,
the JSON sidecar iq-recorder writes next to every ``.iq``, the WAV header
(sample rate only), and finally the file name
(e.g. ``..._436.950MHz_1.00Msps_ci16_le...``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

IQ_FORMATS = ("cu8", "ci16", "complex64", "wav")

#: Accepted spellings -> canonical format name.
FORMAT_ALIASES = {
    "cu8": "cu8", "uint8": "cu8", "u8": "cu8", "rtl": "cu8",
    "ci16": "ci16", "ci16_le": "ci16", "int16": "ci16", "i16": "ci16", "sc16": "ci16",
    "complex64": "complex64", "cf32": "complex64", "cf32_le": "complex64", "fc32": "complex64", "float32": "complex64",
    "wav": "wav",
}

#: Default format by file extension when nothing better is known.
SUFFIX_DEFAULT_FORMAT = {
    ".iq": "cu8",        # iq-recorder / rtl_sdr output
    ".cu8": "cu8",
    ".raw": "ci16",      # CAMRAS snapshots
    ".cs16": "ci16",
    ".ci16": "ci16",
    ".bin": "complex64",
    ".dat": "complex64",
    ".cf32": "complex64",
    ".fc32": "complex64",
    ".wav": "wav",
}

RAW_IQ_SUFFIXES = tuple(SUFFIX_DEFAULT_FORMAT) + (".sigmf-data",)


@dataclass
class IQInfo:
    """What we could find out about a recording before reading it."""

    path: Path
    iq_format: str
    sample_rate_hz: Optional[float] = None
    center_freq_hz: Optional[float] = None
    source: str = "defaults"  # where the metadata came from

    @property
    def bytes_per_sample(self) -> int:
        return {"cu8": 2, "ci16": 4, "complex64": 8}.get(self.iq_format, 4)


def normalise_format(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in ("", "auto"):
        return None
    if key not in FORMAT_ALIASES:
        raise ValueError(f"Unknown IQ format {name!r}; use one of {', '.join(IQ_FORMATS)}")
    return FORMAT_ALIASES[key]


def is_raw_iq_file(path: Path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(".sigmf-data") or Path(path).suffix.lower() in SUFFIX_DEFAULT_FORMAT


# --------------------------------------------------------------------------- #
# Metadata discovery
# --------------------------------------------------------------------------- #

_FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(GHz|MHz|kHz|Hz)(?![a-z])", re.IGNORECASE)
_RATE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(Msps|ksps|sps)", re.IGNORECASE)
_UNIT = {"ghz": 1e9, "mhz": 1e6, "khz": 1e3, "hz": 1.0, "msps": 1e6, "ksps": 1e3, "sps": 1.0}


def parse_filename_metadata(name: str) -> dict:
    """Pull frequency / sample rate / format hints out of a file name like
    ``rsp03_2026_02_13_09_39_30_436.950MHz_1.00Msps_ci16_le.chan1.sigmf-data``
    or ``SDRSharp_20200728_093524Z_1544500000Hz_NOAA-15.wav``."""
    out: dict = {}
    rate = _RATE_RE.search(name)
    if rate:
        out["sample_rate_hz"] = float(rate.group(1)) * _UNIT[rate.group(2).lower()]
    freq = _FREQ_RE.search(name)
    if freq:
        out["center_freq_hz"] = float(freq.group(1)) * _UNIT[freq.group(2).lower()]
    lowered = name.lower()
    for token, fmt in (("ci16", "ci16"), ("cf32", "complex64"), ("fc32", "complex64"), ("cu8", "cu8")):
        if token in lowered:
            out["iq_format"] = fmt
            break
    return out


def _sigmf_meta_path(path: Path) -> Optional[Path]:
    name = path.name
    if name.endswith(".sigmf-data"):
        candidate = path.with_name(name[: -len(".sigmf-data")] + ".sigmf-meta")
        return candidate if candidate.exists() else None
    candidate = path.with_suffix(".sigmf-meta")
    return candidate if candidate.exists() else None


def _read_sigmf(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    glob = meta.get("global", {})
    captures = meta.get("captures") or [{}]
    out = {}
    if glob.get("core:sample_rate"):
        out["sample_rate_hz"] = float(glob["core:sample_rate"])
    if captures[0].get("core:frequency"):
        out["center_freq_hz"] = float(captures[0]["core:frequency"])
    datatype = str(glob.get("core:datatype", "")).lower()
    if datatype.startswith("ci16"):
        out["iq_format"] = "ci16"
    elif datatype.startswith("cf32"):
        out["iq_format"] = "complex64"
    elif datatype.startswith("cu8"):
        out["iq_format"] = "cu8"
    return out


def _read_recorder_sidecar(path: Path) -> dict:
    """iq-recorder writes <name>.json next to <name>.iq."""
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    if data.get("sample_rate"):
        out["sample_rate_hz"] = float(data["sample_rate"])
    if data.get("frequency_hz"):
        out["center_freq_hz"] = float(data["frequency_hz"])
    if "sample_rate" in data and "frequency_hz" in data:
        out["iq_format"] = "cu8"  # rtl_sdr native output
    return out


def _read_wav_rate(path: Path) -> dict:
    try:
        from scipy.io import wavfile

        rate, _ = wavfile.read(path, mmap=True)
        return {"sample_rate_hz": float(rate), "iq_format": "wav"}
    except Exception:
        return {"iq_format": "wav"}


def probe(path, iq_format: Optional[str] = None, sample_rate_hz: Optional[float] = None,
          center_freq_hz: Optional[float] = None) -> IQInfo:
    """Work out format / sample rate / centre frequency for a recording.

    Explicit arguments always win; otherwise metadata files, then the file
    name, then the extension default are used.
    """
    path = Path(path)
    found: dict = {}
    source = "defaults"
    meta = _sigmf_meta_path(path)
    if meta is not None:
        found, source = _read_sigmf(meta), f"sigmf:{meta.name}"
    elif path.suffix.lower() == ".wav":
        found, source = _read_wav_rate(path), "wav header"
    else:
        sidecar = _read_recorder_sidecar(path)
        if sidecar:
            found, source = sidecar, f"sidecar:{path.with_suffix('.json').name}"
    from_name = parse_filename_metadata(path.name)
    for key, value in from_name.items():
        if key not in found:
            found[key] = value
            if source == "defaults":
                source = "filename"

    if sample_rate_hz or center_freq_hz or normalise_format(iq_format):
        source = "arguments" if source == "defaults" else f"arguments+{source}"
    fmt = normalise_format(iq_format) or found.get("iq_format") or SUFFIX_DEFAULT_FORMAT.get(path.suffix.lower(), "complex64")
    return IQInfo(
        path=path,
        iq_format=fmt,
        sample_rate_hz=float(sample_rate_hz) if sample_rate_hz else found.get("sample_rate_hz"),
        center_freq_hz=float(center_freq_hz) if center_freq_hz else found.get("center_freq_hz"),
        source=source,
    )


# --------------------------------------------------------------------------- #
# Reading samples
# --------------------------------------------------------------------------- #

class IQReader:
    """Random access to a recording as complex64 without loading it all.

    ``reader[a:b]`` returns samples a..b as a normal complex64 array;
    ``len(reader)`` is the number of complex samples.
    """

    def __init__(self, path, iq_format: str):
        self.path = Path(path)
        self.iq_format = normalise_format(iq_format) or "complex64"
        if self.iq_format == "wav":
            from scipy.io import wavfile

            _, data = wavfile.read(self.path, mmap=True)
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError(f"{self.path.name}: IQ WAV must have 2 channels (I, Q), got shape {data.shape}")
            self._raw = data
            self._scale = 32768.0 if data.dtype == np.int16 else (128.0 if data.dtype == np.uint8 else 1.0)
            self._offset = 128.0 if data.dtype == np.uint8 else 0.0
            self._n = data.shape[0]
        elif self.iq_format == "complex64":
            self._raw = np.memmap(self.path, dtype=np.complex64, mode="r")
            self._n = self._raw.size
        elif self.iq_format == "ci16":
            self._raw = np.memmap(self.path, dtype="<i2", mode="r")
            self._n = self._raw.size // 2
        elif self.iq_format == "cu8":
            self._raw = np.memmap(self.path, dtype=np.uint8, mode="r")
            self._n = self._raw.size // 2
        else:  # pragma: no cover - normalise_format guards this
            raise ValueError(self.iq_format)

    def __len__(self) -> int:
        return self._n

    def close(self) -> None:
        """Release the memory map. On Windows a mapped file cannot be moved or
        deleted until this is done (WinError 32)."""
        raw = getattr(self, "_raw", None)
        mm = getattr(raw, "_mmap", None)
        self._raw = None
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):
                pass  # a view is still alive somewhere; the GC will release it

    def __enter__(self) -> "IQReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __getitem__(self, item) -> np.ndarray:
        if not isinstance(item, slice):
            raise TypeError("IQReader supports slicing only, e.g. reader[0:1024]")
        start, stop, step = item.indices(self._n)
        if step != 1:
            raise ValueError("IQReader slices must have step 1")
        if self.iq_format == "complex64":
            # np.array copies, so returned samples never pin the memory map open
            return np.array(self._raw[start:stop], dtype=np.complex64)
        if self.iq_format == "wav":
            block = np.asarray(self._raw[start:stop, :2], dtype=np.float32)
            block = (block - self._offset) / self._scale
            return (block[:, 0] + 1j * block[:, 1]).astype(np.complex64)
        raw = np.asarray(self._raw[2 * start:2 * stop], dtype=np.float32)
        if self.iq_format == "ci16":
            raw /= 32768.0
        else:  # cu8
            raw = (raw - 127.5) / 127.5
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    def read_all(self, max_samples: Optional[int] = None) -> np.ndarray:
        stop = self._n if max_samples is None else min(self._n, int(max_samples))
        return self[0:stop]


def open_iq(path, iq_format: Optional[str] = None) -> IQReader:
    info = probe(path, iq_format=iq_format)
    return IQReader(path, info.iq_format)
