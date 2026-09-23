"""Doppler correction and the standard "SatNOGS-style" waterfall.

A LEO satellite's signal slides by up to about +/-10 kHz at 437 MHz during a pass
(it approaches, then recedes). An RTL-SDR can't retune smoothly while
rtl_sdr is writing a file, so the correction is done afterwards: the
pass prediction (TLE + ground station, rtl_recorder.passes) gives the
expected Doppler shift f_d(t) for every moment of the recording, and each
short FFT block is shifted back by f_d(t). After correction a real
satellite signal sits still at the expected frequency, whereas noise and
terrestrial interference don't line up - that's what makes it easy to
spot, for a human looking at the waterfall and for the ML model.

``standard_waterfall()`` produces the fixed-size, Doppler-corrected,
+/-24 kHz waterfall centred on the downlink frequency - the same view
SatNOGS stations publish - so a model trained on SatNOGS observations can
score our own recordings (see features/waterfall_features.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

#: Width of the standard waterfall around the downlink frequency (SatNOGS FM view).
STANDARD_SPAN_HZ = 48_000.0
STANDARD_ROWS = 128
STANDARD_COLS = 128

DopplerFn = Callable[[np.ndarray], np.ndarray]  # seconds since recording start -> Hz


@dataclass
class DopplerCurve:
    """Predicted Doppler shift sampled over the recording (for storage/plots)."""
    times_s: np.ndarray
    doppler_hz: np.ndarray

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return np.interp(t, self.times_s, self.doppler_hz)

    @property
    def max_abs_hz(self) -> float:
        return float(np.max(np.abs(self.doppler_hz))) if self.doppler_hz.size else 0.0


def doppler_curve_from_pass(propagator, station, frequency_hz: float, start, duration_s: float,
                            step_s: float = 1.0) -> DopplerCurve:
    """Sample rtl_recorder.passes.doppler_shift_hz over [start, start+duration]."""
    from datetime import timedelta

    from rtl_recorder.passes import doppler_shift_hz

    times = np.arange(0.0, max(duration_s, step_s) + step_s, step_s)
    values = np.array([doppler_shift_hz(propagator, station, frequency_hz, start + timedelta(seconds=float(t)))
                       for t in times])
    return DopplerCurve(times, values)


def correct_block(block: np.ndarray, sample_rate_hz: float, shift_hz: float) -> np.ndarray:
    """Shift a short block of IQ down by ``shift_hz`` (a constant over the block)."""
    n = np.arange(block.size, dtype=np.float64)
    return (block * np.exp(-2j * np.pi * shift_hz * n / sample_rate_hz)).astype(np.complex64)


def correct_iq(iq: np.ndarray, sample_rate_hz: float, doppler: DopplerFn, *, offset_hz: float = 0.0,
               start_s: float = 0.0) -> np.ndarray:
    """Remove offset + Doppler from a whole IQ array with continuous phase
    (phase = 2*pi * integral of the shift), e.g. before demodulation."""
    t = start_s + np.arange(iq.size) / sample_rate_hz
    shift = offset_hz + doppler(t)
    phase = 2 * np.pi * np.cumsum(shift) / sample_rate_hz
    return (iq * np.exp(-1j * phase)).astype(np.complex64)


def _block_spectrum(block: np.ndarray, window: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.fftshift(np.fft.fft(block * window))) ** 2


def standard_waterfall(reader, sample_rate_hz: float, *, offset_hz: float = 0.0, doppler: Optional[DopplerFn] = None,
                       span_hz: float = STANDARD_SPAN_HZ, rows: int = STANDARD_ROWS, cols: int = STANDARD_COLS,
                       ffts_per_row: int = 8, max_seconds: Optional[float] = None) -> np.ndarray:
    """Doppler-corrected waterfall of +/-span/2 around (centre + offset_hz).

    ``reader`` is anything sliceable with len() (an iq_io.IQReader or a
    numpy array). Streams through the recording, so length doesn't matter.
    Returns a (rows, cols) float32 array of dB values, row 0 = start.
    """
    total = len(reader)
    if max_seconds is not None:
        total = min(total, int(max_seconds * sample_rate_hz))
    # FFT size giving at least `cols` bins across the span.
    nfft = 1 << int(np.ceil(np.log2(max(64, cols * sample_rate_hz / span_hz))))
    if nfft > total:
        raise ValueError(f"Recording too short ({total} samples) for a {nfft}-point FFT")
    bin_hz = sample_rate_hz / nfft
    half_bins = int(round(span_hz / 2 / bin_hz))
    if 2 * half_bins + 1 > nfft:
        raise ValueError(f"Sample rate {sample_rate_hz:g} Hz is too low for a {span_hz:g} Hz span")
    window = np.hanning(nfft).astype(np.float32)
    n_frames = total // nfft
    requested_rows = rows
    rows = max(1, min(rows, n_frames))
    bounds = np.linspace(0, n_frames, rows + 1).astype(int)
    centre_bin = nfft // 2
    out = np.empty((rows, 2 * half_bins + 1), dtype=np.float32)
    for r in range(rows):
        lo, hi = bounds[r], max(bounds[r] + 1, bounds[r + 1])
        picks = np.unique(np.linspace(lo, hi - 1, min(ffts_per_row, hi - lo)).astype(int))
        acc = np.zeros(2 * half_bins + 1)
        for f in picks:
            s = f * nfft
            block = np.asarray(reader[s:s + nfft], dtype=np.complex64)
            t_mid = (s + nfft / 2) / sample_rate_hz
            shift = offset_hz + (float(doppler(np.array([t_mid]))[0]) if doppler is not None else 0.0)
            spec = _block_spectrum(correct_block(block, sample_rate_hz, shift), window)
            acc += spec[centre_bin - half_bins: centre_bin + half_bins + 1]
        out[r] = 10 * np.log10(acc / picks.size + 1e-20)
    return resize(out, rows_out=requested_rows, cols_out=cols)


def resize(matrix: np.ndarray, rows_out: int, cols_out: int) -> np.ndarray:
    """Area-average (down) / linear (up) resize without extra dependencies."""
    m = np.asarray(matrix, dtype=np.float64)

    def _axis(a: np.ndarray, n_out: int, axis: int) -> np.ndarray:
        n_in = a.shape[axis]
        if n_in == n_out:
            return a
        if n_in > n_out:
            edges = np.linspace(0, n_in, n_out + 1)
            idx = np.floor(edges[:-1]).astype(int)
            sums = np.add.reduceat(a, idx, axis=axis)
            counts = np.diff(np.append(idx, n_in))
            shape = [1] * a.ndim
            shape[axis] = -1
            return sums / counts.reshape(shape)
        x_old = np.linspace(0, 1, n_in)
        x_new = np.linspace(0, 1, n_out)
        return np.apply_along_axis(lambda v: np.interp(x_new, x_old, v), axis, a)

    return _axis(_axis(m, rows_out, 0), cols_out, 1).astype(np.float32)
