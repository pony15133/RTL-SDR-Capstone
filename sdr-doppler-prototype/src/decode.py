"""Decode METEOR-M LRPT images from kept recordings with SatDump.

A decoded image is the strongest possible proof that a pass really captured
the satellite, so this is also used as a check on the detector: an
"uncertain" capture that decodes is automatically confirmed as a signal.

Steps for one capture:

1. Prepare a baseband SatDump can read: the station records 150 kHz below
   the downlink (offset tuning), so the recording is shifted to put the
   downlink at 0 Hz, the predicted Doppler (from the stored pass track) is
   removed, and it is low-pass filtered and decimated to ~256 ksps cs16.
   Streaming, so a 1 GB recording needs little memory.
2. Run SatDump's METEOR pipelines (72k first, then 80k):
       satdump pipeline meteor_m2-x_lrpt baseband <file> <out> --samplerate 256000 --baseband_format cs16
3. Any PNG produced = decoded. The images land in
   <results>/decoded/<capture id>_<satellite>/ and the capture row gets
   decode_status + decoded_image_dir.

SatDump is free (https://www.satdump.org): install it once; it is found on
PATH, in the usual install folders, or via --satdump / "satdump_path" in
capture_config.json. Without it, decoding is skipped with a clear status.

    python src/decode.py --capture 12            # one capture
    python src/decode.py --pending               # every kept METEOR capture not decoded yet
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy import signal

from config import DB_PATH, RESULTS_DIR
from database import get_pass_positions, get_result, list_results, update_result
from doppler import DopplerCurve
from iq_io import IQReader

logger = logging.getLogger(__name__)

METEOR_PIPELINES = ("meteor_m2-x_lrpt", "meteor_m2-x_lrpt_80k")
TARGET_RATE_HZ = 256_000.0
BLOCK_SAMPLES = 1 << 20
DEFAULT_TIMEOUT_S = 1800

STATUS_NO_SATDUMP = "skipped: SatDump not installed"
STATUS_NOT_METEOR = "skipped: not a METEOR satellite"
STATUS_NO_IQ = "skipped: raw IQ not available"


def is_meteor(satellite_name: Optional[str]) -> bool:
    return bool(satellite_name) and "METEOR" in satellite_name.upper()


def candidate_satdump_paths() -> List[Path]:
    paths = []
    if sys.platform.startswith("win"):
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
            if base:
                paths += [Path(base) / "SatDump" / "satdump.exe", Path(base) / "SatDump" / "bin" / "satdump.exe"]
        paths.append(Path("C:/SatDump/satdump.exe"))
    elif sys.platform == "darwin":
        paths += [Path("/Applications/SatDump.app/Contents/MacOS/satdump"), Path("/opt/homebrew/bin/satdump")]
    paths += [Path("/usr/bin/satdump"), Path("/usr/local/bin/satdump")]
    return paths


def find_satdump(explicit: Optional[str] = None) -> Optional[str]:
    """Path to the satdump command line program, or None."""
    for cand in ([explicit] if explicit else []) + [os.environ.get("SATDUMP_PATH")]:
        if cand and Path(cand).is_file():
            return str(cand)
    found = shutil.which("satdump")
    if found:
        return found
    for p in candidate_satdump_paths():
        if p.is_file():
            return str(p)
    return None


def doppler_from_track(track: Sequence[dict], recording_start: Optional[str]) -> Optional[DopplerCurve]:
    """Doppler curve (seconds since recording start -> Hz) from stored pass_positions."""
    pts = [(p["timestamp_utc"], p.get("doppler_hz")) for p in track if p.get("doppler_hz") is not None]
    if len(pts) < 2 or not recording_start:
        return None
    t0 = datetime.fromisoformat(recording_start.replace("Z", "+00:00"))
    times = np.array([(datetime.fromisoformat(t.replace("Z", "+00:00")) - t0).total_seconds() for t, _ in pts])
    return DopplerCurve(times, np.array([d for _, d in pts], dtype=float))


def prepare_baseband(iq_path, sample_rate_hz: float, out_path, *, offset_hz: float = 0.0,
                     doppler: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                     target_rate_hz: float = TARGET_RATE_HZ, fmt: str = "cu8") -> float:
    """Shift (offset + Doppler) to 0 Hz, low-pass, decimate and write cs16.
    Streams block by block with continuous phase and filter state.
    Returns the output sample rate."""
    decim = max(1, int(round(sample_rate_hz / target_rate_hz)))
    out_rate = sample_rate_hz / decim
    taps = signal.firwin(8 * decim + 1, 0.8 / decim) if decim > 1 else np.array([1.0])
    zi = np.zeros(taps.size - 1, dtype=np.complex128)
    phase = 0.0
    keep_from = 0          # decimation phase carried across blocks
    n_done = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with IQReader(iq_path, fmt) as reader, out_path.open("wb") as out:
        total = len(reader)
        for start in range(0, total, BLOCK_SAMPLES):
            block = np.asarray(reader[start:start + BLOCK_SAMPLES], dtype=np.complex64).astype(np.complex128)
            t = (n_done + np.arange(block.size)) / sample_rate_hz
            shift = offset_hz + (doppler(t) if doppler is not None else 0.0)
            ph = phase + 2 * np.pi * np.cumsum(np.broadcast_to(shift, t.shape)) / sample_rate_hz
            phase = float(ph[-1]) % (2 * np.pi)
            mixed = block * np.exp(-1j * ph)
            filtered, zi = signal.lfilter(taps, 1.0, mixed, zi=zi)
            dec = filtered[keep_from::decim]
            keep_from = (keep_from - block.size) % decim
            n_done += block.size
            scale = 32767.0 / 1.2   # cu8 full scale is ~1.0; leave headroom
            pairs = np.empty(dec.size * 2, dtype=np.int16)
            pairs[0::2] = np.clip(dec.real * scale, -32768, 32767)
            pairs[1::2] = np.clip(dec.imag * scale, -32768, 32767)
            out.write(pairs.tobytes())
    return out_rate


def run_satdump(satdump: str, pipeline: str, baseband: Path, out_dir: Path, samplerate: float, *,
                timeout_s: int = DEFAULT_TIMEOUT_S, runner=subprocess.run) -> tuple:
    """(returncode, log text). Never raises for a failed decode."""
    cmd = [satdump, "pipeline", pipeline, "baseband", str(baseband), str(out_dir),
           "--samplerate", str(int(samplerate)), "--baseband_format", "cs16", "--dc_block"]
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = runner(cmd, capture_output=True, text=True, timeout=timeout_s)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, f"SatDump timed out after {timeout_s} s"
    except OSError as exc:
        return -1, f"Could not run SatDump: {exc}"


def images_in(folder: Path) -> List[Path]:
    return sorted(p for p in Path(folder).rglob("*.png")) if Path(folder).exists() else []


def decode_capture(capture_id: int, *, db_path: Optional[Path] = None, results_dir: Optional[Path] = None,
                   satdump_path: Optional[str] = None, pipelines: Sequence[str] = METEOR_PIPELINES,
                   force: bool = False, runner=subprocess.run, confirm_review: bool = True) -> dict:
    """Decode one capture; updates its row. Returns {"status", "images", "image_dir"}."""
    db_path = Path(db_path) if db_path else DB_PATH
    row = get_result(db_path, int(capture_id))
    if row is None:
        raise KeyError(f"No capture with id {capture_id}")

    def finish(status, images=(), image_dir=None):
        update_result(db_path, int(capture_id), {"decode_status": status,
                                                 "decoded_image_dir": str(image_dir) if image_dir else None})
        return {"status": status, "images": [str(p) for p in images], "image_dir": str(image_dir) if image_dir else None}

    if not is_meteor(row.get("satellite_name")):
        return finish(STATUS_NOT_METEOR)
    iq = row.get("raw_iq_file_path")
    if not iq or not Path(iq).exists():
        return finish(STATUS_NO_IQ)
    satdump = find_satdump(satdump_path)
    if satdump is None:
        return finish(STATUS_NO_SATDUMP)
    if not force and str(row.get("decode_status") or "").startswith("decoded"):
        return {"status": row["decode_status"], "images": [str(p) for p in images_in(row["decoded_image_dir"])],
                "image_dir": row.get("decoded_image_dir")}

    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in row["satellite_name"])
    out_dir = results_dir / "decoded" / f"{capture_id}_{safe_name}"
    baseband = out_dir / "baseband.cs16"
    sample_rate = float(row.get("sample_rate") or 1_024_000)
    tuned = row.get("frequency_hz")
    target = row.get("target_frequency_hz") or tuned
    offset = float(target - tuned) if tuned and target else 0.0
    doppler = doppler_from_track(get_pass_positions(db_path, int(capture_id)), row.get("actual_recording_start"))
    update_result(db_path, int(capture_id), {"decode_status": "decoding"})
    try:
        rate = prepare_baseband(iq, sample_rate, baseband, offset_hz=offset, doppler=doppler)
        log_lines = []
        images: List[Path] = []
        for pipeline in pipelines:
            code, log = run_satdump(satdump, pipeline, baseband, out_dir / pipeline, rate, runner=runner)
            log_lines.append(f"$ satdump pipeline {pipeline} (exit {code})\n{log}")
            images = images_in(out_dir / pipeline)
            if images:
                break
        (out_dir / "decode.log").write_text("\n".join(log_lines), encoding="utf-8")
    except Exception as exc:  # decoding is a bonus - never break the station
        logger.warning("Decoding capture %s failed: %s", capture_id, exc)
        return finish(f"error: {exc}")
    finally:
        if baseband.exists():
            baseband.unlink()   # large temporary file

    if not images:
        return finish("no images (signal too weak or not LRPT)", image_dir=out_dir)
    result = finish(f"decoded: {len(images)} image(s)", images, image_dir=out_dir)
    # An uncertain capture that decodes is certainly a signal.
    if confirm_review and row.get("review_status") == "pending":
        from review import resolve

        resolve(int(capture_id), 1, db_path=db_path, reviewer="satdump-decode",
                notes=f"confirmed by METEOR decode ({len(images)} images)")
    return result


def pending_captures(db_path: Path, limit: int = 500) -> List[dict]:
    return [r for r in list_results(db_path, limit)
            if is_meteor(r.get("satellite_name")) and r.get("raw_iq_file_path") and Path(r["raw_iq_file_path"]).exists()
            and not str(r.get("decode_status") or "").startswith("decoded")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Decode METEOR LRPT images from kept recordings with SatDump.")
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--capture", type=int, help="Capture id (see history.py)")
    p.add_argument("--pending", action="store_true", help="Every kept METEOR capture not decoded yet")
    p.add_argument("--satdump", help="Path to satdump(.exe) if it isn't found automatically")
    p.add_argument("--force", action="store_true", help="Decode again even if already decoded")
    args = p.parse_args(argv)
    if find_satdump(args.satdump) is None:
        print("SatDump not found. Install it from https://www.satdump.org (free), or pass --satdump <path>.")
        return 1
    ids = [args.capture] if args.capture is not None else [r["id"] for r in pending_captures(args.db)] if args.pending else []
    if not ids:
        print("Nothing to decode (use --capture N or --pending).")
        return 0
    for cid in ids:
        r = decode_capture(cid, db_path=args.db, results_dir=args.results_dir, satdump_path=args.satdump,
                           force=args.force)
        print(f"capture {cid}: {r['status']}" + (f" -> {r['image_dir']}" if r["images"] else ""))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
