"""Waterfall plot of a raw IQ recording - no detection, no ML.

This is the standalone visualisation Bijaya asked for (24 Aug): point it at
any capture and get the picture you would look at to judge what signal is
there. It streams through the file, so a multi-GB recording renders in
bounded memory: the time axis is split into ``--rows`` rows and each row
averages up to ``--ffts-per-row`` FFTs taken evenly across that slice.

    python src/visualize.py --input recordings/METEOR_20260901_137900000Hz.iq
    python src/visualize.py --input ../Data/Satellite_Data_snapshots/RSP-03_4.raw \\
        --start-seconds 5 --duration-seconds 10 --output data/results/rsp03_zoom.png

Format, sample rate and centre frequency are auto-detected from SigMF /
iq-recorder metadata or the file name when possible; pass --format,
--sample-rate or --center-freq to override.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iq_io import IQ_FORMATS, IQReader, probe  # noqa: E402

DEFAULT_NFFT = 1024
DEFAULT_ROWS = 600
DEFAULT_FFTS_PER_ROW = 16


@dataclass
class Waterfall:
    power_db: np.ndarray       # (rows, nfft), row 0 = earliest
    freqs_hz: np.ndarray       # absolute if centre known, else offset
    start_s: float
    end_s: float
    sample_rate_hz: float
    center_freq_hz: Optional[float]


def compute_waterfall(reader: IQReader, sample_rate_hz: float, *, center_freq_hz: Optional[float] = None,
                      nfft: int = DEFAULT_NFFT, rows: int = DEFAULT_ROWS, ffts_per_row: int = DEFAULT_FFTS_PER_ROW,
                      start_seconds: float = 0.0, duration_seconds: Optional[float] = None) -> Waterfall:
    total = len(reader)
    first = max(0, int(round(start_seconds * sample_rate_hz)))
    last = total if duration_seconds is None else min(total, first + int(round(duration_seconds * sample_rate_hz)))
    n_frames = (last - first) // nfft
    if n_frames < 1:
        raise ValueError(
            f"Selected span has fewer than {nfft} samples "
            f"(file has {total / sample_rate_hz:.2f} s at {sample_rate_hz:g} Hz)."
        )
    rows = max(1, min(rows, n_frames))
    # Row boundaries spread every frame across the rows, so no tail is dropped.
    bounds = np.linspace(0, n_frames, rows + 1).astype(int)
    window = np.hanning(nfft).astype(np.float32)

    power = np.empty((rows, nfft), dtype=np.float32)
    for r in range(rows):
        lo, hi = bounds[r], max(bounds[r] + 1, bounds[r + 1])
        picks = np.unique(np.linspace(lo, hi - 1, min(ffts_per_row, hi - lo)).astype(int))
        acc = np.zeros(nfft, dtype=np.float64)
        for f in picks:
            s = first + f * nfft
            block = reader[s:s + nfft]
            acc += np.abs(np.fft.fft(block * window)) ** 2
        power[r] = np.fft.fftshift(10.0 * np.log10(acc / picks.size + 1e-20))

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate_hz))
    if center_freq_hz:
        freqs = freqs + center_freq_hz
    used = n_frames * nfft
    return Waterfall(power, freqs, first / sample_rate_hz, (first + used) / sample_rate_hz,
                     sample_rate_hz, center_freq_hz)


def save_waterfall_png(wf: Waterfall, output_path: Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if wf.center_freq_hz:
        x = wf.freqs_hz / 1e6
        xlabel = "Frequency (MHz)"
    else:
        x = wf.freqs_hz / 1e3
        xlabel = "Offset from centre (kHz) - centre frequency unknown"
    vmin, vmax = np.percentile(wf.power_db, [5, 99.7])
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(wf.power_db, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                   extent=[x[0], x[-1], wf.end_s, wf.start_s], interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (s)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="Power (dB, relative)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render a waterfall (spectrogram) PNG of a raw IQ recording. No ML involved.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None,
                   help="PNG path (default: data/results/waterfall_<name>.png)")
    p.add_argument("--format", default="auto", help=f"IQ format: auto, {', '.join(IQ_FORMATS)}")
    p.add_argument("--sample-rate", type=float, default=None, help="Hz (default: from metadata/file name)")
    p.add_argument("--center-freq", type=float, default=None, help="Hz (default: from metadata/file name)")
    p.add_argument("--nfft", type=int, default=DEFAULT_NFFT, help="FFT size = frequency resolution")
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Time rows in the image")
    p.add_argument("--ffts-per-row", type=int, default=DEFAULT_FFTS_PER_ROW)
    p.add_argument("--start-seconds", type=float, default=0.0)
    p.add_argument("--duration-seconds", type=float, default=None)
    return p


def run(args) -> Path:
    info = probe(args.input, iq_format=args.format, sample_rate_hz=args.sample_rate, center_freq_hz=args.center_freq)
    if not info.sample_rate_hz:
        raise SystemExit(f"FAILED: sample rate unknown for {args.input.name} - pass --sample-rate.")
    with IQReader(args.input, info.iq_format) as reader:
        total_samples = len(reader)
        wf = compute_waterfall(reader, info.sample_rate_hz, center_freq_hz=info.center_freq_hz, nfft=args.nfft,
                               rows=args.rows, ffts_per_row=args.ffts_per_row,
                               start_seconds=args.start_seconds, duration_seconds=args.duration_seconds)
    output = args.output or Path(__file__).resolve().parents[1] / "data" / "results" / f"waterfall_{args.input.stem}.png"
    centre = f"{info.center_freq_hz / 1e6:.4f} MHz" if info.center_freq_hz else "centre unknown"
    title = (f"{args.input.name}\n{info.iq_format}, {info.sample_rate_hz / 1e6:g} Msps, {centre}, "
             f"{wf.start_s:.1f}-{wf.end_s:.1f} s  (metadata: {info.source})")
    save_waterfall_png(wf, output, title)
    print(f"format={info.iq_format} sample_rate={info.sample_rate_hz:g} center_freq={info.center_freq_hz} source={info.source}")
    print(f"duration_s={total_samples / info.sample_rate_hz:.2f} rendered={wf.start_s:.2f}-{wf.end_s:.2f}s")
    print(f"waterfall_image={output}")
    return output


def main(argv=None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
