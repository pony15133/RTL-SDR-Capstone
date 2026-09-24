"""REQ-1 ("load signal data from text files") regression tests.

Drives the original text-file workflow end to end through ``src/main.py``:
a whitespace ``.txt`` / comma ``.csv`` spectrogram matrix (time rows x
frequency columns, dB - the layout the original notebook material saved)
-> load_input() -> matrix_to_spectrogram() -> features -> rule detector
-> JSON summary + PNG + capture_results row.

The matrices are synthetic test fixtures (a clean drifting trace in
Gaussian noise, and noise alone), not real captures.
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main  # noqa: E402
from load_data import load_input  # noqa: E402

ROWS, COLS = 120, 256


def _drifting_matrix(seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    power = rng.normal(-80, 2, size=(ROWS, COLS))
    power[np.arange(ROWS), np.linspace(60, 200, ROWS).astype(int)] = -45
    return power


def _noise_matrix(seed=2) -> np.ndarray:
    return np.random.default_rng(seed).normal(-80, 2, size=(ROWS, COLS))


def _run_main(tmp_path, input_path, *extra) -> tuple[int, dict, dict]:
    db = tmp_path / "captures.sqlite3"
    args = main.build_parser().parse_args(
        ["--input", str(input_path), "--output", str(tmp_path / "out"), "--db", str(db), "--no-ml", *extra]
    )
    code = main.run(args)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM capture_results ORDER BY id DESC LIMIT 1").fetchone())
    summary = json.loads(next((tmp_path / "out").glob(f"session_*/{input_path.stem}_*_summary.json")).read_text())
    return code, row, summary


@pytest.mark.parametrize("suffix, delimiter", [(".txt", " "), (".txt", "\t"), (".csv", ",")])
def test_text_spectrogram_runs_through_detection_to_database(tmp_path, suffix, delimiter):
    path = tmp_path / f"spectrogram0_136800000{suffix}"
    np.savetxt(path, _drifting_matrix(), fmt="%.3f", delimiter=delimiter)

    code, row, summary = _run_main(tmp_path, path, "--save-image")

    assert code == 0
    assert row["input_file"] == str(path)
    assert row["detection_result"] == 1 and row["rule_detection_result"] == 1
    assert row["raw_iq_file_path"] is None          # a matrix is not raw IQ, so no IQ path is stored
    assert row["spectrogram_image_path"] and Path(row["spectrogram_image_path"]).exists()
    assert summary["rule_detection_result"] is True
    assert summary["frequency_drift_hz"] > 0


def test_text_noise_matrix_is_not_flagged(tmp_path):
    path = tmp_path / "noise.txt"
    np.savetxt(path, _noise_matrix(), fmt="%.3f")

    code, row, summary = _run_main(tmp_path, path)

    assert code == 0
    assert row["detection_result"] == 0
    assert summary["rule_detection_result"] is False


def test_txt_and_csv_give_identical_results(tmp_path):
    matrix = _drifting_matrix()
    txt, csv = tmp_path / "m.txt", tmp_path / "m.csv"
    np.savetxt(txt, matrix, fmt="%.3f")
    np.savetxt(csv, matrix, fmt="%.3f", delimiter=",")
    a, b = load_input(txt), load_input(csv)
    assert a.kind == b.kind == "spectrogram"
    assert a.values.shape == (ROWS, COLS)
    np.testing.assert_array_equal(a.values, b.values)


def test_one_dimensional_text_file_is_rejected(tmp_path):
    path = tmp_path / "row.txt"
    np.savetxt(path, _drifting_matrix()[0], fmt="%.3f")
    with pytest.raises(ValueError, match="2D spectrogram matrix"):
        load_input(path)
