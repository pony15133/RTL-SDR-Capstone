"""save_spectrogram_image(): long captures are averaged down to at most
MAX_IMAGE_ROWS time rows before plotting (memory), short ones untouched."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectrogram import MAX_IMAGE_ROWS, SpectrogramData, rows_for_display, save_spectrogram_image  # noqa: E402


def test_short_matrix_is_passed_through_unchanged():
    m = np.random.default_rng(0).normal(size=(MAX_IMAGE_ROWS, 16))
    assert rows_for_display(m) is m


def test_long_matrix_is_block_averaged():
    rows, cols = 5 * MAX_IMAGE_ROWS + 3, 8
    m = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    out = rows_for_display(m)
    assert out.shape[1] == cols and out.shape[0] <= MAX_IMAGE_ROWS
    factor = int(np.ceil(rows / MAX_IMAGE_ROWS))
    np.testing.assert_allclose(out[0], m[:factor].mean(axis=0))
    np.testing.assert_allclose(out[-1], m[(out.shape[0] - 1) * factor:].mean(axis=0))


def test_block_average_works_on_transposed_views():
    # iq_to_spectrogram() returns power_db as a transposed (non C-contiguous) view.
    m = np.random.default_rng(1).normal(size=(8, 3 * MAX_IMAGE_ROWS)).T
    out = rows_for_display(m)
    np.testing.assert_allclose(out[0], m[:3].mean(axis=0))


def test_long_spectrogram_image_is_written(tmp_path):
    rows, cols = 4 * MAX_IMAGE_ROWS, 64
    power = np.random.default_rng(2).normal(-80, 2, size=(rows, cols))
    spec = SpectrogramData(power_db=power, frequencies_hz=np.linspace(137.7e6, 137.8e6, cols),
                           times_s=np.arange(rows) * 5e-4)
    out = save_spectrogram_image(spec, tmp_path / "long.png")
    assert out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
