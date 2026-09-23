"""Tests for scripts/chunk_bin_to_dataset.py - the windowed .bin -> CSV tool.

scripts/ isn't a package, so the module is loaded by file path. The .bin
fixture here is synthetic (background noise with one embedded chirp
"burst" standing in for a short real-world packet), written purely to
exercise windowing/dedup/labelling-path logic - not to represent a real
LoRadar capture or claim any detection accuracy.

Detection thresholds in these tests deliberately do NOT use the
pipeline's satellite-pass defaults (snr_threshold_db=6, min_valid_ratio=
0.45) - see the module docstring in chunk_bin_to_dataset.py for why those
under-detect a short burst diluted inside a longer window, and why
snr_threshold_db~15 / min_valid_ratio~0.2 was needed to correctly
separate the synthetic burst from surrounding noise.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "chunk_bin_to_dataset.py"
sys.path.insert(0, str(SRC_DIR))


def _load_script_module():
    spec = importlib.util.spec_from_file_location("chunk_bin_to_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cbd = _load_script_module()

from features.dataset_io import existing_source_files  # noqa: E402

SAMPLE_RATE = 240_000.0


def _write_noise_with_burst_bin(path: Path, *, duration_s: float = 6.0, burst_start_s: float = 3.0, burst_len_s: float = 0.3, seed: int = 5) -> None:
    """Background noise with one embedded drifting-tone burst - a synthetic
    stand-in for a short real packet inside a much longer, mostly-empty
    capture (same shape of problem as the LoRadar dataset's ~3% duty cycle,
    at a much smaller/faster scale for test speed)."""
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * duration_s)
    iq = (0.15 * (rng.normal(size=n) + 1j * rng.normal(size=n))).astype(np.complex64)

    burst_start = int(burst_start_s * SAMPLE_RATE)
    burst_len = int(burst_len_s * SAMPLE_RATE)
    t = np.arange(burst_len) / SAMPLE_RATE
    rate = (20_000.0 - (-20_000.0)) / t[-1]
    phase = 2 * np.pi * (-20_000.0 * t + 0.5 * rate * t * t)
    burst = np.exp(1j * phase).astype(np.complex64)
    iq[burst_start:burst_start + burst_len] += burst * 3.0

    iq.tofile(path)


def _write_pure_noise_bin(path: Path, *, duration_s: float = 2.0, seed: int = 9) -> None:
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * duration_s)
    noise = (0.15 * (rng.normal(size=n) + 1j * rng.normal(size=n))).astype(np.complex64)
    noise.tofile(path)


def _base_args(input_path, dataset, **overrides):
    argv = [
        "--input", str(input_path),
        "--dataset", str(dataset),
        "--sample-rate", str(SAMPLE_RATE),
        "--center-freq", "401300000",
        "--binary-dtype", "complex64",
        "--window-seconds", "1.0",
        "--stride-seconds", "0.5",
        "--nperseg", "1024",
        "--noverlap", "512",
        "--snr-threshold-db", "15",
        "--min-valid-ratio", "0.2",
    ]
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            argv.append(flag)
        elif value is not False:
            argv += [flag, str(value)]
    return cbd.build_arg_parser().parse_args(argv)


class TestIterWindows:
    def test_basic_windowing(self):
        windows = list(cbd.iter_windows(total_samples=1000, window_samples=400, stride_samples=200))
        assert windows == [(0, 400), (200, 600), (400, 800), (600, 1000)]

    def test_no_overlap_when_stride_equals_window(self):
        windows = list(cbd.iter_windows(total_samples=900, window_samples=300, stride_samples=300))
        assert windows == [(0, 300), (300, 600), (600, 900)]

    def test_shorter_than_one_window_yields_nothing(self):
        windows = list(cbd.iter_windows(total_samples=100, window_samples=400, stride_samples=200))
        assert windows == []

    def test_rejects_non_positive_sizes(self):
        with pytest.raises(ValueError):
            list(cbd.iter_windows(total_samples=1000, window_samples=0, stride_samples=200))
        with pytest.raises(ValueError):
            list(cbd.iter_windows(total_samples=1000, window_samples=400, stride_samples=0))


class TestWindowSourceKey:
    def test_deterministic_for_same_inputs(self, tmp_path):
        path = tmp_path / "a.bin"
        assert cbd.window_source_key(path, 0, 100) == cbd.window_source_key(path, 0, 100)

    def test_differs_for_different_windows(self, tmp_path):
        path = tmp_path / "a.bin"
        assert cbd.window_source_key(path, 0, 100) != cbd.window_source_key(path, 100, 200)


class TestRunEndToEnd:
    def test_burst_window_flagged_as_candidate_and_auto_accepted(self, tmp_path):
        bin_path = tmp_path / "capture.bin"
        _write_noise_with_burst_bin(bin_path, duration_s=6.0, burst_start_s=3.0, burst_len_s=0.3)
        dataset = tmp_path / "features.csv"

        args = _base_args(bin_path, dataset, auto_accept_candidates=True)
        exit_code = cbd.run(args)

        assert exit_code == 0
        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        labels = [int(row["label"]) for row in rows]
        assert 1 in labels, "expected at least one window covering the burst to be labelled positive"
        assert 0 in labels, "expected the surrounding noise-only windows to be auto-labelled negative"
        assert all(row["is_synthetic"] == "0" for row in rows)

    def test_pure_noise_file_produces_only_negatives(self, tmp_path):
        bin_path = tmp_path / "noise_only.bin"
        _write_pure_noise_bin(bin_path, duration_s=3.0)
        dataset = tmp_path / "features.csv"

        args = _base_args(bin_path, dataset, auto_accept_candidates=True)
        cbd.run(args)

        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows  # windows were produced
        assert all(int(row["label"]) == 0 for row in rows)
        assert all("auto-negative" in row["notes"] for row in rows)

    def test_auto_negative_rows_tagged_in_notes(self, tmp_path):
        bin_path = tmp_path / "noise_only.bin"
        _write_pure_noise_bin(bin_path, duration_s=2.0)
        dataset = tmp_path / "features.csv"

        cbd.run(_base_args(bin_path, dataset, auto_accept_candidates=True))

        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert all("auto-negative (rule detector: no signal)" in row["notes"] for row in rows)

    def test_auto_accepted_candidate_rows_tagged_in_notes(self, tmp_path):
        bin_path = tmp_path / "capture.bin"
        _write_noise_with_burst_bin(bin_path, duration_s=6.0, burst_start_s=3.0, burst_len_s=0.3)
        dataset = tmp_path / "features.csv"

        cbd.run(_base_args(bin_path, dataset, auto_accept_candidates=True))

        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        candidate_rows = [row for row in rows if int(row["label"]) == 1]
        assert candidate_rows
        assert all("auto-accepted candidate" in row["notes"] for row in candidate_rows)

    def test_interactive_review_path(self, tmp_path, monkeypatch):
        bin_path = tmp_path / "capture.bin"
        _write_noise_with_burst_bin(bin_path, duration_s=6.0, burst_start_s=3.0, burst_len_s=0.3)
        dataset = tmp_path / "features.csv"

        monkeypatch.setattr("builtins.input", lambda _: "1")  # confirm every candidate as positive
        args = _base_args(bin_path, dataset)  # no --auto-accept-candidates
        cbd.run(args)

        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        reviewed = [row for row in rows if "manually reviewed" in row["notes"]]
        assert reviewed
        assert all(int(row["label"]) == 1 for row in reviewed)

    def test_skip_at_review_does_not_append_row(self, tmp_path, monkeypatch):
        bin_path = tmp_path / "capture.bin"
        _write_noise_with_burst_bin(bin_path, duration_s=6.0, burst_start_s=3.0, burst_len_s=0.3)
        dataset = tmp_path / "features.csv"

        monkeypatch.setattr("builtins.input", lambda _: "s")  # skip every candidate
        cbd.run(_base_args(bin_path, dataset))

        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert all(int(row["label"]) == 0 for row in rows)  # only auto-negatives got appended

    def test_rerun_deduplicates_by_window(self, tmp_path):
        bin_path = tmp_path / "noise_only.bin"
        _write_pure_noise_bin(bin_path, duration_s=2.0)
        dataset = tmp_path / "features.csv"

        args = _base_args(bin_path, dataset, auto_accept_candidates=True)
        cbd.run(args)
        first_count = len(existing_source_files(dataset))

        cbd.run(args)
        second_count = len(existing_source_files(dataset))

        assert first_count > 0
        assert second_count == first_count

    def test_reprocess_flag_adds_duplicate_rows(self, tmp_path):
        bin_path = tmp_path / "noise_only.bin"
        _write_pure_noise_bin(bin_path, duration_s=1.5)
        dataset = tmp_path / "features.csv"

        args = _base_args(bin_path, dataset, auto_accept_candidates=True)
        cbd.run(args)
        with dataset.open(newline="") as handle:
            first_row_count = len(list(csv.DictReader(handle)))

        cbd.run(_base_args(bin_path, dataset, auto_accept_candidates=True, reprocess=True))
        with dataset.open(newline="") as handle:
            second_row_count = len(list(csv.DictReader(handle)))

        assert second_row_count == first_row_count * 2

    def test_max_windows_caps_processing(self, tmp_path):
        bin_path = tmp_path / "noise_only.bin"
        _write_pure_noise_bin(bin_path, duration_s=5.0)  # would otherwise produce several windows
        dataset = tmp_path / "features.csv"

        cbd.run(_base_args(bin_path, dataset, auto_accept_candidates=True, max_windows=2))

        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2

    def test_save_candidate_images_writes_a_png(self, tmp_path):
        bin_path = tmp_path / "capture.bin"
        _write_noise_with_burst_bin(bin_path, duration_s=6.0, burst_start_s=3.0, burst_len_s=0.3)
        dataset = tmp_path / "features.csv"
        image_dir = tmp_path / "images"

        cbd.run(_base_args(bin_path, dataset, auto_accept_candidates=True, save_candidate_images=True, output=image_dir))

        assert any(image_dir.glob("*.png"))

    def test_capture_shorter_than_one_window_fails_cleanly(self, tmp_path):
        bin_path = tmp_path / "short.bin"
        _write_pure_noise_bin(bin_path, duration_s=0.1)  # shorter than the 1.0s window
        dataset = tmp_path / "features.csv"

        exit_code = cbd.run(_base_args(bin_path, dataset))

        assert exit_code == 1
        assert not dataset.exists()

    def test_missing_input_file_fails_cleanly(self, tmp_path):
        exit_code = cbd.run(_base_args(tmp_path / "does_not_exist.bin", tmp_path / "features.csv"))
        assert exit_code == 1

    def test_spectrogram_matrix_input_rejected(self, tmp_path):
        matrix_path = tmp_path / "matrix.csv"
        np.savetxt(matrix_path, np.zeros((10, 10)), delimiter=",")
        exit_code = cbd.run(_base_args(matrix_path, tmp_path / "features.csv"))
        assert exit_code == 1

    def test_rows_use_deterministic_source_file_for_dedup(self, tmp_path):
        bin_path = tmp_path / "noise_only.bin"
        _write_pure_noise_bin(bin_path, duration_s=1.5)
        dataset = tmp_path / "features.csv"

        cbd.run(_base_args(bin_path, dataset, auto_accept_candidates=True))

        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            assert row["source_file"].startswith(str(bin_path) + "#samples=")
