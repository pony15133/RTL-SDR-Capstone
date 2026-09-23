#!/usr/bin/env python3
"""Convert a SigMF capture (.sigmf-meta + .sigmf-data) into a project-ready
.npy complex64 IQ array that load_data.py / main.py / the other scripts
in this project can consume directly.

SigMF (https://github.com/sigmf/SigMF) stores raw samples in a
`.sigmf-data` file with no header, and describes their format in a
sibling `.sigmf-meta` JSON file's `global."core:datatype"` field, e.g.
`ci16_le` (complex signed 16-bit little-endian - the GUI's primary
case), `cu8` (complex unsigned 8-bit - RTL-SDR's own native raw format),
or `cf32_le` (complex float32, i.e. already what this project calls
complex64). Only complex ("c...") datatypes are supported - a real-valued
("r...") SigMF file isn't an IQ capture and is rejected with a clear
error rather than silently misread.

Integer sample types are scaled to roughly [-1, 1] before being combined
into complex64:
    - signed N-bit:   value / 2**(N-1)
    - unsigned N-bit:  (value - midpoint) / midpoint, midpoint = 2**(N-1) - 0.5
      (matches the standard RTL-SDR cu8-style DC-centred conversion)
Floating-point sample types are cast to complex64 unscaled - the source
format is assumed to already be normalised.

Example (matches what the GUI's "Convert SigMF" button runs):
    python scripts/convert_sigmf_to_iq.py \\
        --meta capture.sigmf-meta --data capture.sigmf-data --output data/raw/capture.npy
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

#: SigMF core:datatype grammar: [c|r]{i|u|f}{8|16|32|64}[_le|_be]
#: (the endianness suffix is only present for 16/32/64-bit types).
_DATATYPE_RE = re.compile(r"^(?P<complex>[cr])(?P<kind>[iuf])(?P<bits>8|16|32|64)(?:_(?P<endian>le|be))?$")


class SigMFFormatError(ValueError):
    """The .sigmf-meta file is missing required fields or names an
    unsupported/unrecognised core:datatype."""


def parse_sigmf_datatype(datatype: str) -> Tuple[bool, np.dtype]:
    """Return (is_complex, numpy_dtype) for a SigMF core:datatype string."""
    match = _DATATYPE_RE.match(datatype.strip())
    if not match:
        raise SigMFFormatError(
            f"Unrecognised SigMF core:datatype {datatype!r} - expected the SigMF grammar "
            "[c|r]{i|u|f}{8|16|32|64}[_le|_be], e.g. 'ci16_le', 'cu8', 'cf32_le'"
        )
    is_complex = match.group("complex") == "c"
    kind = match.group("kind")
    bits = int(match.group("bits"))
    endian = match.group("endian") or "le"  # irrelevant for 8-bit, harmless default
    byte_order = "<" if endian == "le" else ">"
    dtype = np.dtype(f"{byte_order}{kind}{bits // 8}")
    return is_complex, dtype


def load_sigmf_meta(meta_path: Path) -> dict:
    try:
        return json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SigMFFormatError(f"{meta_path} is not valid JSON: {exc}") from exc


def _scale_to_unit_range(samples: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return samples
    bits = dtype.itemsize * 8
    if np.issubdtype(dtype, np.signedinteger):
        full_scale = float(2 ** (bits - 1))
        return samples / full_scale
    if np.issubdtype(dtype, np.unsignedinteger):
        midpoint = float(2 ** (bits - 1)) - 0.5
        return (samples - midpoint) / midpoint
    raise SigMFFormatError(f"Unsupported base numpy dtype {dtype} - not int, uint, or float")  # pragma: no cover - regex already restricts this


def convert_sigmf(meta_path: Path, data_path: Path, output_path: Path) -> dict:
    """Convert one SigMF capture to a complex64 .npy file. Returns a summary dict."""
    meta = load_sigmf_meta(meta_path)
    global_meta = meta.get("global", {})
    datatype = global_meta.get("core:datatype")
    if not datatype:
        raise SigMFFormatError(f"{meta_path} has no 'global.core:datatype' field - cannot determine the raw sample format")

    is_complex, base_dtype = parse_sigmf_datatype(datatype)
    if not is_complex:
        raise SigMFFormatError(
            f"core:datatype {datatype!r} is real-valued, not complex - this converter only handles "
            "complex IQ SigMF captures ('c...' datatypes). A real-valued capture is not an IQ recording."
        )

    if Path(data_path).stat().st_size == 0:
        raise SigMFFormatError(f"{data_path} is empty")

    raw = np.memmap(data_path, dtype=base_dtype, mode="r")
    if raw.size % 2 != 0:
        raise SigMFFormatError(
            f"{data_path} has {raw.size} {base_dtype} values - an odd count, so it cannot be "
            "interleaved I/Q pairs. Check --data points at the right file and core:datatype is correct."
        )

    i_samples = _scale_to_unit_range(raw[0::2].astype(np.float32), base_dtype)
    q_samples = _scale_to_unit_range(raw[1::2].astype(np.float32), base_dtype)
    iq = (i_samples + 1j * q_samples).astype(np.complex64)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, iq)

    captures = meta.get("captures") or []
    center_freq = captures[0].get("core:frequency") if captures else None

    return {
        "datatype": datatype,
        "n_samples": int(iq.size),
        "sample_rate_hz": global_meta.get("core:sample_rate"),
        "center_freq_hz": center_freq,
        "output_path": str(output_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a SigMF (.sigmf-meta/.sigmf-data) capture into a project-ready .npy complex64 IQ array.")
    parser.add_argument("--meta", required=True, type=Path, help="Path to the .sigmf-meta JSON file")
    parser.add_argument("--data", required=True, type=Path, help="Path to the .sigmf-data raw binary file")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the converted .npy file")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.meta.exists():
        print(f"FAILED: metadata file not found: {args.meta}", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"FAILED: data file not found: {args.data}", file=sys.stderr)
        return 1

    try:
        info = convert_sigmf(args.meta, args.data, args.output)
    except (SigMFFormatError, OSError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"datatype={info['datatype']}")
    print(f"n_samples={info['n_samples']}")
    if info["sample_rate_hz"] is not None:
        print(f"sample_rate_hz={info['sample_rate_hz']} (from SigMF metadata - use this for --sample-rate downstream)")
    if info["center_freq_hz"] is not None:
        print(f"center_freq_hz={info['center_freq_hz']} (from SigMF metadata - use this for --center-freq downstream)")
    print(f"output={info['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
