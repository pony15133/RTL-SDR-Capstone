import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detect import detect_candidate
from spectrogram import SpectrogramData


def test_detects_smooth_drifting_trace():
    rows, cols = 80, 128
    rng = np.random.default_rng(123)
    power = rng.normal(-80, 2, size=(rows, cols))
    bins = np.linspace(30, 90, rows).astype(int)
    power[np.arange(rows), bins] = -45
    spec = SpectrogramData(
        power_db=power,
        frequencies_hz=np.linspace(136.7e6, 136.9e6, cols),
        times_s=np.arange(rows),
    )

    result = detect_candidate(
        spec,
        min_valid_ratio=0.5,
        min_drift_hz=10_000,
        max_smoothness_hz=5_000,
        snr_threshold_db=8,
    )

    assert result.detected
    assert result.valid_signal_ratio > 0.9
