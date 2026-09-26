#!/usr/bin/env python3
"""Turn the client's labelled pass recordings into training data.

Layout (one folder per pass, named <SATELLITE>_<passN>_<outcome>; one sub-folder
per part of the pass, named by the client after the elevation):

    client_pass_recordings/SARAL_pass1_detected/notinsky/spectrogram0_465988000.txt ... (+ Freqtest_*.png)
    client_pass_recordings/SARAL_pass1_detected/justrisen/...
    client_pass_recordings/NORAD-16_marginal/12to41/...
    client_pass_recordings/LILACSAT-2_not_detected/8tomax/spectrogram*_437200000.txt

Outcome = detected | marginal | not_detected (the client's own verdict). The
client's original folder names (DetectedSatellite(SARAL), Nondetection(LILACSAT2),
...) are still understood.

Each spectrogram*.txt is one ~1 s snapshot: 250 time rows x 1024 frequency bins
in dB, centred on the frequency in the file name (240 kHz wide). Snapshots are
~1.6 s apart and ordered by their file time, so each pass folder is a complete
pass from before rise to after set.

What this script does, per pass:
  1. Removes fixed local spurs (bins that stand out while the satellite is not
     in the sky, bins that stand out in >40% of the whole pass, and the DC bin)
     so they can't be mistaken for a satellite.
  2. Finds the satellite's Doppler track from the data: in each snapshot the
     strongest peak within +/-15 kHz of the centre, kept only when strong, then
     fitted with a decreasing curve (Doppler always falls through a pass). A
     "track" that sweeps less than 3 kHz is a fixed line, not a satellite.
  3. Labels each snapshot from the folder AND the data:
       - not in the sky / below the horizon, or a "not_detected" pass  -> 0
       - in the sky and >= 6 dB on the track                             -> 1
       - a part that rises/sets mid-way, >30 s outside the track, < 3 dB -> 0
       - anything else (in the sky but faint, or unclear)                -> dropped
     Faint in-sky snapshots are dropped, not called noise, so the model never
     learns to throw away weak real passes.
  4. Writes one IQ-model row per snapshot (the 9 spectrogram features) and one
     waterfall-model row per ~100 s window (Doppler-corrected, +/-24 kHz,
     128 x 128 - the same picture the station builds).
  5. Saves a preview image per pass with the track and labels drawn on it.
     LOOK AT THESE before trusting the numbers.

Passes are split by folder so nothing from a test pass is ever trained on:
  --external "SARAL_pass2_detected" "NORAD-16_marginal"   (default)

    python scripts/build_client_spectrogram_set.py --root E:\\CAPSTONE\\CAP2\\client_pass_recordings

Outputs (in data/training/): client_passes_iq_train.csv, client_passes_iq_external.csv,
client_passes_waterfall_train.csv, client_passes_waterfall_external.csv; previews and
per-pass caches in data/client_passes/. Re-running reuses the caches (fast).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doppler import STANDARD_COLS, STANDARD_ROWS, STANDARD_SPAN_HZ, resize  # noqa: E402
from features.extractor import FEATURE_NAMES, extract_features, feature_vector_to_array  # noqa: E402
from features.waterfall_features import WATERFALL_FEATURE_NAMES, waterfall_features  # noqa: E402
from spectrogram import SpectrogramData  # noqa: E402

DEFAULT_SPAN_HZ = 240_000.0        # width of each snapshot (from the client's PNG axes)
DEFAULT_SNAPSHOT_S = 1.07          # length of each snapshot (250 rows)
SEARCH_HZ = 15_000.0               # LEO Doppler at 470 MHz stays within about +/-11 kHz
TRACK_POINT_DB = 8.0               # a snapshot peak must be this strong to shape the track
TRACK_HALF_BINS = 3                # "on the track" = within +/-3 bins (~700 Hz)
SIGNAL_DB = 6.0                    # label 1 at or above this on the track
FAINT_DB = 3.0                     # below this counts as "nothing there"
MIN_TRACK_SWEEP_HZ = 3_000.0       # a real pass sweeps many kHz (SARAL: ~20 kHz); a fixed line sweeps 0
PERSISTENT_SPUR_FRACTION = 0.4     # a bin standing out in >40% of all snapshots is a local spur
HORIZON_MARGIN_S = 30.0            # mixed parts: this far outside the track's time span = below horizon
WINDOW_ROWS, WINDOW_STEP = 64, 32  # waterfall windows: 64 snapshots (~100 s), half overlap
DEFAULT_EXTERNAL = ("SARAL_pass2_detected", "NORAD-16_marginal")
FILE_RE = re.compile(r"spectrogram(\d+)_(\d+)\.txt$", re.IGNORECASE)


# --------------------------------------------------------------------------- discovery

@dataclass
class Segment:
    name: str
    state: str                     # "below" | "mixed" | "above"
    files: list = field(default_factory=list)   # [(index, path, mtime)]


@dataclass
class Pass:
    name: str
    kind: str                      # detected | marginal | nondetection
    satellite: str
    centre_hz: float
    segments: list


def segment_state(name: str) -> str:
    """Where the satellite is during this part of the pass, from the folder name."""
    n = name.lower().replace(" ", "")
    if n == "notinsky":
        return "below"
    if n.startswith("notinsky"):
        return "mixed"             # rises during this part
    nums = [int(x) for x in re.findall(r"-?\d+", n)]
    if len(nums) == 2:
        if max(nums) <= 0:
            return "below"
        if min(nums) < 0:
            return "mixed"         # sets during this part
    return "above"                 # justrisen, max35, 8tomax, 15toset, ...


def pass_kind(name: str) -> str:
    """The client's verdict for the pass, from the folder name (new or original naming)."""
    n = name.lower().replace("-", "_")
    if "not_detected" in n or "nondetection" in n:
        return "nondetection"
    if "marginal" in n:
        return "marginal"
    return "detected"


def pass_satellite(name: str) -> str:
    """SARAL_pass1_detected -> SARAL; DetectedSatellite(SARAL) -> SARAL."""
    m = re.search(r"\(([^)]+)\)", name)
    return m.group(1) if m else name.split("_")[0]


def discover(root: Path) -> list:
    passes = []
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        segs = []
        centre = None
        for sdir in sorted(p for p in pdir.iterdir() if p.is_dir()):
            files = []
            for f in sdir.iterdir():
                m = FILE_RE.search(f.name)
                if m:
                    files.append((int(m.group(1)), f, f.stat().st_mtime))
                    centre = float(m.group(2))
            if files:
                files.sort()
                segs.append(Segment(sdir.name, segment_state(sdir.name), files))
        if not segs:
            continue
        segs.sort(key=lambda s: min(t for _, _, t in s.files))   # time order of the pass
        passes.append(Pass(pdir.name, pass_kind(pdir.name), pass_satellite(pdir.name), centre, segs))
    return passes


# --------------------------------------------------------------------------- signal processing

def load_snapshot(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float32, engine="c").to_numpy()


def mean_spectrum_db(m: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.mean(10 ** (m.astype(np.float64) / 10), axis=0))


def running_median(x: np.ndarray, width: int = 31) -> np.ndarray:
    pad = width // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + width]) for i in range(len(x))])


def find_spurs(reference_db: np.ndarray, excess_db: float = 3.0) -> np.ndarray:
    """Bins that stand out while no satellite is present (local interference), plus DC."""
    spur = (reference_db - running_median(reference_db)) > excess_db
    spur[len(spur) // 2] = True
    return spur | np.roll(spur, 1) | np.roll(spur, -1)


def notch(row_or_matrix: np.ndarray, spurs: np.ndarray, floor: np.ndarray) -> np.ndarray:
    out = np.array(row_or_matrix, dtype=np.float64, copy=True)
    out[..., spurs] = floor[spurs]
    return out


def peak_near_centre(spec_db: np.ndarray, bin_hz: float, spurs: np.ndarray):
    n = len(spec_db)
    c = n // 2
    half = int(SEARCH_HZ / bin_hz)
    seg = spec_db[c - half:c + half + 1].copy()
    seg[spurs[c - half:c + half + 1]] = -np.inf
    k = int(np.argmax(seg))
    noise = np.median(spec_db[c - 4 * half:c + 4 * half + 1])
    return (k - half) * bin_hz, float(seg[k] - noise)


def fit_track(times: np.ndarray, offsets: np.ndarray, snr: np.ndarray, usable: np.ndarray):
    """Doppler falls monotonically through a pass: robust decreasing fit on strong peaks."""
    from sklearn.isotonic import IsotonicRegression
    keep = usable & (snr >= TRACK_POINT_DB)
    if keep.sum() < 10:
        return None
    t, f = times[keep], offsets[keep]
    for _ in range(2):   # drop points far from a first fit (interference bursts)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(t, f)
        ok = np.abs(f - iso.predict(t)) < 1500.0
        if ok.sum() < 10:
            return None
        t, f = t[ok], f[ok]
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(t, f)
    fitted = iso.predict(t)
    if fitted.max() - fitted.min() < MIN_TRACK_SWEEP_HZ:
        return None      # a line that doesn't move in frequency is interference, not a satellite
    return iso, float(t.min()), float(t.max())


def snr_on_track(spec_db: np.ndarray, offset_hz: float, bin_hz: float) -> float:
    c = len(spec_db) // 2 + int(round(offset_hz / bin_hz))
    on = spec_db[c - TRACK_HALF_BINS:c + TRACK_HALF_BINS + 1].max()
    around = np.r_[spec_db[c - 60:c - 8], spec_db[c + 9:c + 61]]
    return float(on - np.median(around))


# --------------------------------------------------------------------------- per pass

def process_pass(p: Pass, cache_dir: Path, span_hz: float, snapshot_s: float, limit: Optional[int] = None) -> dict:
    cache = cache_dir / f"{safe(p.name)}.npz"
    rows_meta = [(s.name, s.state, i, f, t) for s in p.segments for i, f, t in (s.files[:limit] if limit else s.files)]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if len(z["times"]) == len(rows_meta):
            return dict(z)
    states = np.array([r[1] for r in rows_meta])
    # Spurs come from the part of the pass with no satellite in the sky (read first, it's small).
    ref_idx = np.flatnonzero(states == "below")
    if ref_idx.size == 0:
        ref_idx = np.arange(min(30, len(rows_meta)))
    ref = np.median([mean_spectrum_db(load_snapshot(rows_meta[k][3])) for k in ref_idx], axis=0)
    spurs = find_spurs(ref)
    floor = running_median(ref)
    nbins = len(ref)
    bin_hz = span_hz / nbins
    freqs = p.centre_hz + (np.arange(nbins) - nbins // 2) * bin_hz
    spectra, feats = [], []
    for k, (seg, state, idx, f, t) in enumerate(rows_meta):
        m = notch(load_snapshot(f), spurs, floor)          # one snapshot in memory at a time
        spectra.append(mean_spectrum_db(m))
        spec = SpectrogramData(power_db=m, frequencies_hz=freqs, times_s=np.linspace(0, snapshot_s, m.shape[0]))
        feats.append(feature_vector_to_array(extract_features(spec, snr_threshold_db=SIGNAL_DB)))
        print(f"\r  {p.name}: {k + 1}/{len(rows_meta)} snapshots", end="", flush=True)
    print()
    clean = np.array(spectra)
    times = np.array([r[4] for r in rows_meta], dtype=float)
    times -= times.min()
    out = dict(spectra=clean.astype(np.float32), iq_features=np.array(feats), times=times, states=states,
               segments=np.array([r[0] for r in rows_meta]), indices=np.array([r[2] for r in rows_meta]),
               files=np.array([str(r[3]) for r in rows_meta]), spurs=spurs, bin_hz=bin_hz)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def persistent_spurs(spectra: np.ndarray, excess_db: float = 3.0) -> np.ndarray:
    """Bins that stand out in a large share of ALL snapshots. A satellite moves through
    the band (Doppler), so it never sits in one bin that long; local interference does -
    including lines that only appear after the pass has started."""
    base = running_median(np.median(spectra, axis=0))
    frac = ((spectra - base) > excess_db).mean(axis=0)
    spur = frac > PERSISTENT_SPUR_FRACTION
    return spur | np.roll(spur, 1) | np.roll(spur, -1)


def label_pass(p: Pass, d: dict) -> dict:
    times, states, bin_hz = d["times"], d["states"], float(d["bin_hz"])
    late = persistent_spurs(d["spectra"])
    spurs = d["spurs"] | late
    spectra = notch(d["spectra"], late, running_median(np.median(d["spectra"], axis=0)))
    peaks = np.array([peak_near_centre(s, bin_hz, spurs) for s in spectra])
    offsets, peak_snr = peaks[:, 0], peaks[:, 1]
    track = None if p.kind == "nondetection" else fit_track(times, offsets, peak_snr, states != "below")
    if track is not None:
        iso, t0, t1 = track
        track_hz = iso.predict(times)
    else:
        track_hz = np.zeros_like(times)
    snr = np.array([snr_on_track(s, f, bin_hz) for s, f in zip(spectra, track_hz)])
    label = np.full(len(times), -1)                      # -1 = dropped
    if p.kind == "nondetection":
        label[:] = 0
    else:
        label[states == "below"] = 0
        if track is not None:
            on = (states != "below") & (snr >= SIGNAL_DB)
            label[on] = 1
            # Parts that rise or set mid-way ("notinskyto3", "5to-5"): well before the first /
            # after the last strong track point the satellite is below the horizon.
            _, t0, t1 = track
            outside = (states == "mixed") & ((times < t0 - HORIZON_MARGIN_S) | (times > t1 + HORIZON_MARGIN_S))
            label[outside & (snr < FAINT_DB)] = 0
    return dict(track_hz=track_hz, track_found=track is not None, snr=snr, label=label, spectra=spectra)


def waterfall_windows(d: dict, lab: dict, span_hz: float = STANDARD_SPAN_HZ):
    spectra, bin_hz = lab["spectra"], float(d["bin_hz"])
    n, nbins = spectra.shape
    c = nbins // 2
    half = int(round(span_hz / 2 / bin_hz))
    shift = np.round(lab["track_hz"] / bin_hz).astype(int)     # Doppler correction (0 when no track)
    corrected = np.array([row[c + s - half:c + s + half + 1] for row, s in zip(spectra, shift)])
    out = []
    for start in range(0, max(1, n - WINDOW_ROWS + 1), WINDOW_STEP):
        stop = min(n, start + WINDOW_ROWS)
        labels = lab["label"][start:stop]
        if (labels == 1).mean() >= 0.5:
            y = 1
        elif (labels == 0).all():
            y = 0
        else:
            continue                                          # mixed or mostly dropped
        wf = resize(corrected[start:stop], STANDARD_ROWS, STANDARD_COLS)
        out.append((start, stop, y, wf))
    return out


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def preview(p: Pass, d: dict, lab: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spectra, bin_hz = lab["spectra"], float(d["bin_hz"])
    nb = spectra.shape[1]
    half = int(25_000 / bin_hz)
    view = spectra[:, nb // 2 - half:nb // 2 + half + 1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [4, 1]}, sharey=True)
    lo, hi = np.percentile(view, [5, 99.7])
    a.imshow(view, aspect="auto", origin="lower", cmap="viridis", vmin=lo, vmax=hi,
             extent=[-25, 25, 0, len(view)])
    if lab["track_found"]:
        a.plot(lab["track_hz"] / 1e3, np.arange(len(view)) + 0.5, color="white", lw=1, ls="--")
    seg = d["segments"]
    for i in range(1, len(seg)):
        if seg[i] != seg[i - 1]:
            a.axhline(i, color="white", lw=0.6)
            a.text(-24.5, i + 1, seg[i], color="white", fontsize=8, va="bottom")
    a.text(-24.5, 1, seg[0], color="white", fontsize=8, va="bottom")
    a.set_xlabel("offset from centre (kHz)")
    a.set_ylabel("snapshot (time runs up)")
    a.set_title(f"{p.name} - {p.centre_hz / 1e6:.3f} MHz - dashed = fitted Doppler track")
    colors = np.where(lab["label"] == 1, "tab:green", np.where(lab["label"] == 0, "tab:gray", "tab:orange"))
    b.scatter(lab["snr"], np.arange(len(view)) + 0.5, c=colors, s=6)
    b.axvline(SIGNAL_DB, color="k", lw=0.6)
    b.set_xlabel("dB on track\ngreen=1 grey=0 orange=dropped")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


# --------------------------------------------------------------------------- main

def build(root: Path, out_dir: Path, work_dir: Path, external=DEFAULT_EXTERNAL, span_hz=DEFAULT_SPAN_HZ,
          snapshot_s=DEFAULT_SNAPSHOT_S, limit=None) -> dict:
    passes = discover(root)
    if not passes:
        raise SystemExit(f"FAILED: no pass folders with spectrogram*.txt under {root}")
    iq_rows, wf_rows, summary = [], [], []
    ext = {e.lower() for e in external}
    for p in passes:
        d = process_pass(p, work_dir / "cache", span_hz, snapshot_s, limit)
        lab = label_pass(p, d)
        preview(p, d, lab, work_dir / "preview" / f"{safe(p.name)}.png")
        split = "external" if p.name.lower() in ext else "train"
        tag = safe(p.name)
        for k in range(len(d["times"])):
            if lab["label"][k] < 0:
                continue
            iq_rows.append({
                "capture_id": f"client_{tag}_{d['segments'][k]}_{int(d['indices'][k])}",
                **dict(zip(FEATURE_NAMES, d["iq_features"][k])), "label": int(lab["label"][k]), "is_synthetic": 0,
                "source_file": str(Path(str(d["files"][k])).relative_to(root)), "sample_rate_hz": span_hz,
                "nperseg": int(d["spectra"].shape[1]), "noverlap": 0,
                "notes": f"{p.kind}; {d['states'][k]}; {lab['snr'][k]:.1f} dB on track",
                "recording_id": f"client_{tag}_{d['segments'][k]}", "pass": p.name, "satellite": p.satellite,
                "split": split, "snr_on_track_db": round(float(lab["snr"][k]), 2),
            })
        for start, stop, y, wf in waterfall_windows(d, lab):
            (work_dir / "waterfalls").mkdir(parents=True, exist_ok=True)
            np.save(work_dir / "waterfalls" / f"{tag}_{start:04d}.npy", wf.astype(np.float16))
            wf_rows.append({
                "capture_id": f"client_{tag}_w{start:04d}", **waterfall_features(wf), "label": y, "is_synthetic": 0,
                "recording_id": f"client_{tag}", "source_file": f"{p.name} snapshots {start}-{stop - 1}",
                "pass": p.name, "satellite": p.satellite, "split": split,
            })
        n = lab["label"]
        summary.append({"pass": p.name, "kind": p.kind, "split": split, "snapshots": len(n),
                        "signal": int((n == 1).sum()), "noise": int((n == 0).sum()), "dropped": int((n < 0).sum()),
                        "track_found": lab["track_found"]})
    iq = pd.DataFrame(iq_rows)
    wf = pd.DataFrame(wf_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in (("iq", iq), ("waterfall", wf)):
        for split in ("train", "external"):
            part = df[df["split"] == split] if len(df) else df
            path = out_dir / f"client_passes_{name}_{split}.csv"
            part.to_csv(path, index=False)
            paths[f"{name}_{split}"] = (path, len(part), int(part["label"].sum()) if len(part) else 0)
    return {"summary": pd.DataFrame(summary), "paths": paths}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="Folder with one sub-folder per pass (client_pass_recordings)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "training")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "data" / "client_passes")
    ap.add_argument("--external", nargs="*", default=list(DEFAULT_EXTERNAL),
                    help="Pass folders kept out of training (tested on only)")
    ap.add_argument("--span-hz", type=float, default=DEFAULT_SPAN_HZ)
    ap.add_argument("--snapshot-seconds", type=float, default=DEFAULT_SNAPSHOT_S)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N snapshots per part (quick test)")
    args = ap.parse_args(argv)
    r = build(args.root, args.out_dir, args.work_dir, args.external, args.span_hz, args.snapshot_seconds, args.limit)
    print("\nPasses:")
    print(r["summary"].to_string(index=False))
    print("\nWritten:")
    for key, (path, n, pos) in r["paths"].items():
        print(f"  {path}  {n} rows ({pos} signal)")
    print(f"\nCheck the previews in {args.work_dir / 'preview'} before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
