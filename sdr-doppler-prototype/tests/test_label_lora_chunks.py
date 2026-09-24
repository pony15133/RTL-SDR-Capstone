import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.label_lora_chunks import existing_capture_ids, run  # noqa: E402


def _write_bin(path: Path, sample_rate: float, duration_s: float, *, burst_start_s=None, burst_len_s=0.0, noise_std=0.01, burst_amp=1.0):
    rng = np.random.default_rng(1234)
    n = int(sample_rate * duration_s)
    signal = (rng.normal(0, noise_std, n) + 1j * rng.normal(0, noise_std, n)).astype(np.complex64)
    if burst_start_s is not None and burst_len_s > 0:
        start = int(burst_start_s * sample_rate)
        length = int(burst_len_s * sample_rate)
        t = np.arange(length) / sample_rate
        tone = (burst_amp * np.exp(1j * 2 * np.pi * 15_000 * t)).astype(np.complex64)
        signal[start:start + length] += tone
    signal.tofile(path)
    return n


def _args(**overrides):
    defaults = dict(
        input=None,
        dataset=None,
        output=Path("/tmp"),
        sample_rate=240_000.0,
        center_freq=401_000_000.0,
        binary_dtype="complex64",
        window_seconds=0.5,
        step_seconds=None,
        nperseg=1024,
        noverlap=512,
        snr_threshold_db=20.0,
        activity_ratio=0.01,
        no_auto_negatives=False,
        save_image=False,
        min_drift_hz=0.0,
        max_smoothness_hz=float("inf"),
        min_valid_ratio=0.0,
        max_windows=None,
        start_seconds=0.0,
        notes="",
    )
    defaults.update(overrides)
    return _Namespace(**defaults)


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_pure_noise_file_auto_labels_all_negative_without_prompting(tmp_path, monkeypatch):
    bin_path = tmp_path / "noise_only.bin"
    _write_bin(bin_path, sample_rate=240_000, duration_s=1.5, noise_std=0.01)
    dataset = tmp_path / "features.csv"

    def _fail_if_prompted():
        raise AssertionError("should not prompt for a window with no activity")

    monkeypatch.setattr("scripts.label_lora_chunks.prompt_for_label", _fail_if_prompted)

    args = _args(input=bin_path, dataset=dataset, window_seconds=0.5)
    run(args)

    with dataset.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3  # 1.5s / 0.5s windows
    assert all(int(row["label"]) == 0 for row in rows)
    assert all("auto" in row["notes"] for row in rows)


def test_burst_window_is_shown_for_review_and_respects_given_label(tmp_path, monkeypatch):
    bin_path = tmp_path / "with_burst.bin"
    _write_bin(bin_path, sample_rate=240_000, duration_s=1.5, noise_std=0.01, burst_start_s=0.5, burst_len_s=0.4, burst_amp=1.0)
    dataset = tmp_path / "features.csv"

    answers = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    args = _args(input=bin_path, dataset=dataset, window_seconds=0.5, max_windows=2)
    run(args)

    with dataset.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # window 0 (0.0-0.5s): no activity -> auto label 0
    # window 1 (0.5-1.0s): contains the burst -> prompted, answered "1"
    assert len(rows) == 2
    assert int(rows[0]["label"]) == 0 and "auto" in rows[0]["notes"]
    assert int(rows[1]["label"]) == 1 and rows[1]["notes"].startswith("reviewed")
    assert rows[1]["snr_db"] and float(rows[1]["snr_db"]) > float(rows[0]["snr_db"])


def test_skip_answer_appends_no_row(tmp_path, monkeypatch):
    bin_path = tmp_path / "with_burst.bin"
    _write_bin(bin_path, sample_rate=240_000, duration_s=1.0, noise_std=0.01, burst_start_s=0.1, burst_len_s=0.3, burst_amp=1.0)
    dataset = tmp_path / "features.csv"

    monkeypatch.setattr("builtins.input", lambda _: "s")

    args = _args(input=bin_path, dataset=dataset, window_seconds=1.0)
    run(args)

    assert not dataset.exists() or existing_capture_ids(dataset) == set()


def test_rerun_skips_already_labeled_windows(tmp_path, monkeypatch):
    bin_path = tmp_path / "noise_only.bin"
    _write_bin(bin_path, sample_rate=240_000, duration_s=1.0, noise_std=0.01)
    dataset = tmp_path / "features.csv"

    args = _args(input=bin_path, dataset=dataset, window_seconds=0.5)
    run(args)
    first_ids = existing_capture_ids(dataset)
    assert len(first_ids) == 2

    # Second run over the same file/dataset should skip every window - no
    # duplicate rows.
    run(args)
    with dataset.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert existing_capture_ids(dataset) == first_ids


def test_max_windows_caps_how_much_of_the_file_is_scanned(tmp_path):
    bin_path = tmp_path / "noise_only.bin"
    _write_bin(bin_path, sample_rate=240_000, duration_s=2.0, noise_std=0.01)
    dataset = tmp_path / "features.csv"

    args = _args(input=bin_path, dataset=dataset, window_seconds=0.5, max_windows=2)
    run(args)

    assert len(existing_capture_ids(dataset)) == 2
