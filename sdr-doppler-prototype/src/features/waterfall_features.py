"""Features for a standard (Doppler-corrected, centred) waterfall.

These describe the 128 x 128 view produced by ``doppler.standard_waterfall``
(our own recordings) and ``waterfall_png.png_to_waterfall`` (SatNOGS
observations). Both are relative-power images whose absolute scale
differs, so every feature is scale-free: values are robust z-scores
(median / MAD of the whole image) and positions are fractions of the
image, never Hz or dB.

The idea behind them: after Doppler correction a real satellite signal
sits near the centre column and appears in many rows; noise is spread
evenly; terrestrial interference is usually off-centre or constant.
"""

from __future__ import annotations

import numpy as np

WATERFALL_FEATURE_NAMES = (
    "wf_peak_z",               # 99.5th percentile z-score anywhere (how strong is the strongest thing)
    "wf_centre_excess_z",      # mean z of the centre band minus the rest (signal where predicted?)
    "wf_centre_active_frac",   # fraction of rows with a centre-band pixel above 4 sigma
    "wf_any_active_frac",      # fraction of rows with any pixel above 4 sigma
    "wf_centre_to_any_ratio",  # how much of the activity is at the centre
    "wf_profile_peak_z",       # peak of the time-averaged spectrum, z-scored over frequency
    "wf_profile_peak_offset",  # |offset| of that peak from centre, fraction of half-span
    "wf_track_jitter",         # median |row-to-row change| of the peak column in active rows, fraction of span
    "wf_active_span_frac",     # first..last active row, fraction of the observation
    "wf_row_burstiness",       # std/mean of centre-band power over time (bursty FM/packets vs steady)
)

#: Fraction of columns trimmed each side (receiver filter roll-off at the edges).
EDGE_TRIM = 0.10
#: Half-width of the "centre" band as a fraction of the (trimmed) width.
CENTRE_HALF_WIDTH = 0.08
ACTIVE_Z = 4.0


def _robust_z(m: np.ndarray) -> np.ndarray:
    med = np.median(m)
    mad = np.median(np.abs(m - med)) * 1.4826
    return (m - med) / (mad if mad > 1e-9 else (np.std(m) + 1e-9))


def waterfall_features(matrix: np.ndarray) -> dict:
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] < 4 or m.shape[1] < 16:
        raise ValueError(f"waterfall must be 2-D and at least 4x16, got {m.shape}")
    rows, cols = m.shape
    trim = int(round(cols * EDGE_TRIM))
    m = m[:, trim:cols - trim]
    # Remove slow per-row level changes (AGC, horizon noise) before scoring.
    m = m - np.median(m, axis=1, keepdims=True)
    z = _robust_z(m)
    width = z.shape[1]
    c = width // 2
    half = max(1, int(round(width * CENTRE_HALF_WIDTH)))
    centre = z[:, c - half:c + half + 1]
    outside = np.concatenate([z[:, :c - half], z[:, c + half + 1:]], axis=1)

    centre_active = centre.max(axis=1) > ACTIVE_Z
    any_active = z.max(axis=1) > ACTIVE_Z

    profile = z.mean(axis=0)
    profile_z = _robust_z(profile)
    peak_col = int(np.argmax(profile_z))

    peak_track = np.argmax(z, axis=1)
    if any_active.sum() >= 3:
        track = peak_track[any_active]
        jitter = float(np.median(np.abs(np.diff(track)))) / width
        active_rows = np.where(any_active)[0]
        span = (active_rows[-1] - active_rows[0] + 1) / rows
    else:
        jitter, span = 0.5, 0.0

    centre_rows = centre.mean(axis=1)
    burst = float(np.std(centre_rows) / (np.mean(np.abs(centre_rows)) + 1e-9))

    return {
        "wf_peak_z": float(np.percentile(z, 99.5)),
        "wf_centre_excess_z": float(centre.mean() - outside.mean()),
        "wf_centre_active_frac": float(centre_active.mean()),
        "wf_any_active_frac": float(any_active.mean()),
        "wf_centre_to_any_ratio": float(centre_active.sum() / max(1, any_active.sum())),
        "wf_profile_peak_z": float(profile_z[peak_col]),
        "wf_profile_peak_offset": float(abs(peak_col - c) / (width / 2)),
        "wf_track_jitter": jitter,
        "wf_active_span_frac": float(span),
        "wf_row_burstiness": min(burst, 50.0),
    }


def waterfall_feature_array(matrix: np.ndarray) -> np.ndarray:
    f = waterfall_features(matrix)
    return np.array([f[name] for name in WATERFALL_FEATURE_NAMES], dtype=np.float64)
