#!/usr/bin/env python3
"""Build a labelled training CSV from a recording where the satellite's
carrier frequency is known (the dish/antenna was tracking it and the
frequency was predicted from TLE), without labelling every window by hand.

How labels are produced (reference labels, independent of the model):

    1. The capture is cut into fixed, NON-overlapping windows
       (--window-seconds). No overlap means neighbouring rows don't share
       samples.
    2. For each FFT frame, the carrier SNR is measured: power in a narrow
       band around the known carrier (--carrier-bw-hz) versus the median
       power across the whole band. The carrier is found automatically
       within +-(--search-khz) of the expected offset, because Doppler and
       oscillator error shift it by a few kHz.
    3. A frame is "on" when carrier SNR > --on-snr-db. A window's on-fraction
       is the share of its frames that are on.
    4. on-fraction >= --pos-fraction  -> label 1 (satellite signal present)
       on-fraction <= --neg-fraction  -> label 0 (noise only)
       anything in between            -> AMBIGUOUS, left out of the dataset
       so borderline windows can't poison the labels.

The features written for each window are the pipeline's normal features
(src/features/extractor.py, computed on the whole band with no knowledge of
where the carrier is), so the model can't just read back the labelling rule.
The rule-based detector's opinion is stored in `notes` for comparison.

Every row gets a `recording_id`. Use the SAME id for files that are the same
moment in time (e.g. two polarisation channels) so ml/train.py never puts
them in different splits.

A PNG waterfall with the window labels overlaid is written next to the CSV
(--review-dir) so a human can check the labels at a glance.

Example (CAMRAS RSP-03 snapshot, ci16, 1 Msps, 436.95 MHz):

    python scripts/label_known_carrier.py \\
        --input ../Data/Satellite_Data_snapshots/RSP-03_3.raw \\
        --dtype ci16 --sample-rate 1000000 --center-freq 436950000 \\
        --recording-id rsp03_snap3 --dataset data/training/rsp03_features.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (  # noqa: E402
    DEFAULT_MAX_SMOOTHNESS_HZ,
    DEFAULT_MIN_DRIFT_HZ,
    DEFAULT_MIN_VALID_RATIO,
    DEFAULT_NOVERLAP,
    DEFAULT_NPERSEG,
    DEFAULT_SNR_THRESHOLD_DB,
)
from detect import detect_candidate  # noqa: E402
from features.dataset_io import CSV_COLUMNS, feature_row_values  # noqa: E402
from features.extractor import extract_features  # noqa: E402
from iq_io import IQ_FORMATS, IQReader  # noqa: E402
from spectrogram import iq_to_spectrogram  # noqa: E402

SUPPORTED_DTYPES = IQ_FORMATS


def load_iq(path: Path, dtype: str) -> np.ndarray:
    """All samples as complex64 scaled to about +-1 (see src/iq_io.py for the formats)."""
    return IQReader(path, dtype).read_all()


def frame_power_db(iq: np.ndarray, nfft: int) -> np.ndarray:
    """(n_frames, nfft) power in dB, fftshifted, one row per non-overlapping frame."""
    n_frames = iq.size // nfft
    frames = np.asarray(iq[: n_frames * nfft]).reshape(n_frames, nfft)
    spectrum = np.fft.fftshift(np.fft.fft(frames * np.hanning(nfft), axis=1), axes=1)
    return 10.0 * np.log10(np.abs(spectrum) ** 2 + 1e-12)


def locate_carrier_bin(power_db: np.ndarray, sample_rate: float, expected_offset_hz: float, search_hz: float) -> int:
    """Bin of the strongest persistent narrowband line near the expected carrier."""
    nfft = power_db.shape[1]
    bin_hz = sample_rate / nfft
    centre = nfft // 2 + int(round(expected_offset_hz / bin_hz))
    half = max(1, int(round(search_hz / bin_hz)))
    lo, hi = max(0, centre - half), min(nfft, centre + half + 1)
    relative = power_db - np.median(power_db, axis=1, keepdims=True)
    return lo + int(np.argmax(np.percentile(relative[:, lo:hi], 95, axis=0)))


def carrier_snr_per_frame(power_db: np.ndarray, carrier_bin: int, half_width_bins: int) -> np.ndarray:
    relative = power_db - np.median(power_db, axis=1, keepdims=True)
    lo = max(0, carrier_bin - half_width_bins)
    hi = min(power_db.shape[1], carrier_bin + half_width_bins + 1)
    return relative[:, lo:hi].max(axis=1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Label windows of a tracked-satellite recording by known-carrier SNR.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--dtype", required=True, choices=SUPPORTED_DTYPES)
    p.add_argument("--sample-rate", required=True, type=float)
    p.add_argument("--center-freq", required=True, type=float, help="Frequency the capture was tuned to (Hz)")
    p.add_argument("--recording-id", required=True, help="Group id; same value for files recorded at the same moment")
    p.add_argument("--dataset", required=True, type=Path, help="CSV to append rows to (created if missing)")
    p.add_argument("--carrier-offset-hz", type=float, default=0.0, help="Expected carrier offset from centre (Hz)")
    p.add_argument("--search-khz", type=float, default=10.0, help="Search +-this many kHz for the carrier")
    p.add_argument("--carrier-bw-hz", type=float, default=5000.0, help="Width of the carrier band used for SNR")
    p.add_argument("--window-seconds", type=float, default=1.0)
    p.add_argument("--label-nfft", type=int, default=1024, help="FFT size for the labelling measurement")
    p.add_argument("--on-snr-db", type=float, default=10.0)
    p.add_argument("--pos-fraction", type=float, default=0.30)
    p.add_argument("--neg-fraction", type=float, default=0.10)
    p.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG)
    p.add_argument("--noverlap", type=int, default=DEFAULT_NOVERLAP)
    p.add_argument("--snr-threshold-db", type=float, default=DEFAULT_SNR_THRESHOLD_DB)
    p.add_argument("--review-dir", type=Path, default=None, help="Where to save the label-overlay PNG (default: next to --dataset)")
    return p


def save_review_png(power_db, sample_rate, windows, carrier_bin, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames, nfft = power_db.shape
    step = max(1, n_frames // 600)
    img = power_db[: n_frames // step * step].reshape(-1, step, nfft).mean(axis=1)
    duration = n_frames * nfft / sample_rate
    vmin, vmax = np.percentile(img, [5, 99.5])
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(img, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
              extent=[-sample_rate / 2e3, sample_rate / 2e3, duration, 0])
    carrier_khz = (carrier_bin - nfft // 2) * sample_rate / nfft / 1e3
    ax.axvline(carrier_khz, color="white", lw=0.5, ls=":")
    colours = {1: "#e8453c", 0: "#3c8de8", None: "#999999"}
    for w in windows:
        ax.add_patch(plt.Rectangle((sample_rate / 2e3 * 0.86, w["start_s"]), sample_rate / 2e3 * 0.12,
                                   w["end_s"] - w["start_s"], color=colours[w["label"]], alpha=0.85))
    ax.set_xlabel("offset from centre (kHz)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"{title}\nright bar: red=1 signal, blue=0 noise, grey=ambiguous (dropped)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def run(args) -> int:
    iq = load_iq(args.input, args.dtype)
    nfft = args.label_nfft
    power_db = frame_power_db(iq, nfft)
    bin_hz = args.sample_rate / nfft
    carrier_bin = locate_carrier_bin(power_db, args.sample_rate, args.carrier_offset_hz, args.search_khz * 1e3)
    half_width = max(1, int(round(args.carrier_bw_hz / 2 / bin_hz)))
    snr = carrier_snr_per_frame(power_db, carrier_bin, half_width)
    on = snr > args.on_snr_db
    carrier_offset_khz = (carrier_bin - nfft // 2) * bin_hz / 1e3

    win = int(round(args.window_seconds * args.sample_rate))
    frames_per_win = win // nfft
    n_windows = iq.size // win
    rows, windows = [], []
    for w in range(n_windows):
        fraction = float(on[w * frames_per_win:(w + 1) * frames_per_win].mean())
        label = 1 if fraction >= args.pos_fraction else 0 if fraction <= args.neg_fraction else None
        windows.append({"start_s": w * args.window_seconds, "end_s": (w + 1) * args.window_seconds, "label": label})
        if label is None:
            continue
        samples = np.asarray(iq[w * win:(w + 1) * win])
        spec = iq_to_spectrogram(samples, args.sample_rate, args.center_freq, args.nperseg, args.noverlap)
        features = extract_features(spec, snr_threshold_db=args.snr_threshold_db)
        rule = detect_candidate(spec, DEFAULT_MIN_VALID_RATIO, DEFAULT_MIN_DRIFT_HZ, DEFAULT_MAX_SMOOTHNESS_HZ,
                                args.snr_threshold_db, features=features)
        start, end = w * win, (w + 1) * win
        rows.append({
            "capture_id": f"{args.recording_id}_{args.input.stem}_w{w:04d}",
            **feature_row_values(features),
            "label": label,
            "is_synthetic": 0,
            "source_file": f"{args.input}#samples={start}-{end}",
            "sample_rate_hz": args.sample_rate,
            "nperseg": args.nperseg,
            "noverlap": args.noverlap,
            "notes": (f"known-carrier label: on_fraction={fraction:.2f} at {carrier_offset_khz:+.1f} kHz; "
                      f"rule_detector={int(rule.detected)}"),
            "recording_id": args.recording_id,
        })

    columns = [*CSV_COLUMNS, "recording_id"]
    new = pd.DataFrame(rows, columns=columns)
    if args.dataset.exists() and args.dataset.stat().st_size > 0:
        existing = pd.read_csv(args.dataset)
        new = new[~new["capture_id"].isin(existing["capture_id"])]
        pd.concat([existing, new], ignore_index=True).reindex(columns=columns).to_csv(args.dataset, index=False)
    else:
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(args.dataset, index=False)

    review_dir = args.review_dir or args.dataset.parent / "label_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    png = review_dir / f"{args.recording_id}_{args.input.stem}_labels.png"
    save_review_png(power_db, args.sample_rate, windows, carrier_bin, png, f"{args.input.name} ({args.recording_id})")

    labels = [w["label"] for w in windows]
    print(f"{args.input.name}: carrier at {carrier_offset_khz:+.1f} kHz, {n_windows} windows -> "
          f"positive={labels.count(1)} negative={labels.count(0)} ambiguous(dropped)={labels.count(None)}; "
          f"appended {len(new)} rows; review image {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
