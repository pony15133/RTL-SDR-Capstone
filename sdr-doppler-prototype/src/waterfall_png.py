"""Turn a SatNOGS waterfall PNG back into a numeric waterfall.

SatNOGS publishes one waterfall image per observation: a viridis image of
power vs frequency (x) and time (y, increasing upwards), with a colour bar
on the right and axis labels around it. Layout details vary a little
between client versions, so nothing is hard-coded:

1. "Coloured" pixels (not white/grey/black text) are found.
2. The widest band of columns that is mostly coloured is the waterfall;
   a narrow coloured band to its right is the colour bar.
3. Every waterfall pixel is mapped to the nearest colour-bar colour, which
   gives its relative power (0 = bottom of the colour bar, 1 = top). If no
   colour bar is found, matplotlib's viridis table is used instead.
4. Rows are flipped so row 0 is the start of the observation, and the
   result is resized to the standard 128 x 128.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from doppler import STANDARD_COLS, STANDARD_ROWS, resize

COLOURED_SPREAD = 40 / 255  # max(R,G,B) - min(R,G,B) above this = coloured pixel


class WaterfallParseError(ValueError):
    pass


def _runs(mask_1d: np.ndarray):
    runs, start = [], None
    for i, on in enumerate(list(mask_1d) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def locate_plot(rgb: np.ndarray) -> Tuple[Tuple[int, int, int, int], Optional[Tuple[int, int]]]:
    """((top, bottom, left, right) of the waterfall, (left, right) of the colour bar or None)."""
    coloured = (rgb.max(axis=2) - rgb.min(axis=2)) > COLOURED_SPREAD
    h, w = coloured.shape
    col_runs = [r for r in _runs(coloured.mean(axis=0) > 0.3) if r[1] - r[0] >= 3]
    if not col_runs:
        raise WaterfallParseError("no waterfall found (no coloured column band)")
    plot = max(col_runs, key=lambda r: r[1] - r[0])
    left, right = plot
    rows_on = np.where(coloured[:, left:right + 1].mean(axis=1) > 0.5)[0]
    if rows_on.size < 10:
        raise WaterfallParseError("waterfall has too few coloured rows")
    top, bottom = int(rows_on[0]), int(rows_on[-1])
    bars = [r for r in col_runs if r[0] > right and (r[1] - r[0]) < (right - left) * 0.25]
    bar = bars[0] if bars else None
    return (top, bottom, left, right), bar


def _viridis_table(n: int = 256) -> np.ndarray:
    from matplotlib import colormaps

    return colormaps["viridis"](np.linspace(0, 1, n))[:, :3]


def png_to_waterfall(path_or_array, *, rows: int = STANDARD_ROWS, cols: int = STANDARD_COLS,
                     trim_edge_pixels: int = 2) -> np.ndarray:
    """Relative power (0..1) waterfall, shape (rows, cols), row 0 = start."""
    if isinstance(path_or_array, (str, Path)):
        from matplotlib import image as mpimg

        rgb = mpimg.imread(str(path_or_array))
    else:
        rgb = np.asarray(path_or_array)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    rgb = rgb[..., :3].astype(np.float32)

    (top, bottom, left, right), bar = locate_plot(rgb)
    t = trim_edge_pixels
    plot = rgb[top + t:bottom - t + 1, left + t:right - t + 1]

    if bar is not None:
        bar_px = rgb[top + t:bottom - t + 1, bar[0] + 1:bar[1]]
        table = np.median(bar_px, axis=1)[::-1]          # bottom of bar (low) -> top (high)
    else:
        table = _viridis_table()

    from scipy.spatial import cKDTree

    idx = cKDTree(table).query(plot.reshape(-1, 3), k=1)[1]
    values = (idx / max(1, len(table) - 1)).reshape(plot.shape[:2]).astype(np.float32)
    values = values[::-1]                                  # time increases upwards in the PNG
    return resize(values, rows, cols)
