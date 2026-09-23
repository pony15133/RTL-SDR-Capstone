"""Tests for scripts/build_training_dataset.py - the batch .bin/.iq -> CSV tool.

scripts/ isn't a package, so the module is loaded by file path. Fixture
.bin files here are synthetic (a clean chirp for "positive-shaped", pure
noise for "negative-shaped"), written purely to exercise the batch tool's
file discovery/label-resolution/dedup/error-handling logic - not to
represent real captures or claim any model accuracy.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_training_dataset.py"
sys.path.insert(0, str(SRC_DIR))


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_training_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


btd = _load_script_module()

from features.dataset_io import existing_source_files  # noqa: E402


def _write_chirp_bin(path: Path, samples: int = 4096, seed: int = 1) -> None:
    """A synthetic drifting-tone .bin fixture (complex64), positive-shaped."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / 240_000.0
    rate = (30_000.0 - (-30_000.0)) / t[-1]
    phase = 2 * np.pi * (-30_000.0 * t + 0.5 * rate * t * t)
    noise = 0.1 * (rng.normal(size=samples) + 1j * rng.normal(size=samples))
    iq = (np.exp(1j * phase) + noise).astype(np.complex64)
    iq.tofile(path)


def _write_noise_bin(path: Path, samples: int = 4096, seed: int = 2) -> None:
    """A synthetic pure-noise .bin fixture (complex64), negative-shaped."""
    rng = np.random.default_rng(seed)
    noise = (0.2 * (rng.normal(size=samples) + 1j * rng.normal(size=samples))).astype(np.complex64)
    noise.tofile(path)


def _base_args(tmp_path, input_dir, dataset, **overrides):
    argv = [
        "--input-dir", str(input_dir),
        "--dataset", str(dataset),
        "--sample-rate", "240000",
        "--center-freq", "137900000",
        "--binary-dtype", "complex64",
        "--nperseg", "128",
        "--noverlap", "64",
    ]
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            argv.append(flag)
        elif value is not False:
            argv += [flag, str(value)]
    return btd.build_arg_parser().parse_args(argv)


class TestParseExtensions:
    def test_normalizes_missing_dots(self):
        assert btd.parse_extensions("bin,iq,.dat") == {".bin", ".iq", ".dat"}

    def test_strips_whitespace(self):
        assert btd.parse_extensions(" .bin , .npy ") == {".bin", ".npy"}


class TestLoadLabelsManifest:
    def test_none_path_returns_empty_dict(self):
        assert btd.load_labels_manifest(None) == {}

    def test_valid_manifest(self, tmp_path):
        path = tmp_path / "labels.csv"
        path.write_text("filename,label\na.bin,1\nb.bin,0\n")
        assert btd.load_labels_manifest(path) == {"a.bin": 1, "b.bin": 0}

    def test_missing_columns_raises(self, tmp_path):
        path = tmp_path / "labels.csv"
        path.write_text("file,tag\na.bin,1\n")
        with pytest.raises(SystemExit, match="filename.*label"):
            btd.load_labels_manifest(path)

    def test_invalid_label_value_raises(self, tmp_path):
        path = tmp_path / "labels.csv"
        path.write_text("filename,label\na.bin,maybe\n")
        with pytest.raises(SystemExit):
            btd.load_labels_manifest(path)

    def test_out_of_range_label_raises(self, tmp_path):
        path = tmp_path / "labels.csv"
        path.write_text("filename,label\na.bin,2\n")
        with pytest.raises(SystemExit):
            btd.load_labels_manifest(path)


class TestResolveLabel:
    def test_manifest_takes_priority_over_folder_convention(self, tmp_path):
        input_dir = tmp_path
        path = input_dir / "negative" / "a.bin"
        path.parent.mkdir()
        path.touch()
        labels_map = {"negative/a.bin": 1}  # manifest disagrees with the folder name on purpose

        assert btd.resolve_label(path, input_dir, labels_map, forced_label=None) == 1

    def test_manifest_matches_by_basename_too(self, tmp_path):
        input_dir = tmp_path
        path = input_dir / "sub" / "a.bin"
        path.parent.mkdir()
        path.touch()

        assert btd.resolve_label(path, input_dir, {"a.bin": 0}, forced_label=None) == 0

    def test_folder_convention_positive(self, tmp_path):
        path = tmp_path / "positive" / "a.bin"
        path.parent.mkdir()
        path.touch()
        assert btd.resolve_label(path, tmp_path, {}, forced_label=None) == 1

    def test_folder_convention_negative(self, tmp_path):
        path = tmp_path / "noise" / "a.bin"
        path.parent.mkdir()
        path.touch()
        assert btd.resolve_label(path, tmp_path, {}, forced_label=None) == 0

    def test_forced_label_used_when_no_manifest_or_folder_match(self, tmp_path):
        path = tmp_path / "unlabelled_dir" / "a.bin"
        path.parent.mkdir()
        path.touch()
        assert btd.resolve_label(path, tmp_path, {}, forced_label=1) == 1

    def test_returns_none_when_nothing_resolves(self, tmp_path):
        path = tmp_path / "unlabelled_dir" / "a.bin"
        path.parent.mkdir()
        path.touch()
        assert btd.resolve_label(path, tmp_path, {}, forced_label=None) is None


class TestIterCaptureFiles:
    def test_filters_by_extension(self, tmp_path):
        (tmp_path / "a.bin").touch()
        (tmp_path / "b.txt").touch()
        found = list(btd.iter_capture_files(tmp_path, {".bin"}, recursive=False))
        assert [p.name for p in found] == ["a.bin"]

    def test_non_recursive_ignores_subdirectories(self, tmp_path):
        (tmp_path / "a.bin").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.bin").touch()
        found = list(btd.iter_capture_files(tmp_path, {".bin"}, recursive=False))
        assert [p.name for p in found] == ["a.bin"]

    def test_recursive_includes_subdirectories(self, tmp_path):
        (tmp_path / "a.bin").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.bin").touch()
        found = list(btd.iter_capture_files(tmp_path, {".bin"}, recursive=True))
        assert {p.name for p in found} == {"a.bin", "b.bin"}


class TestRunEndToEnd:
    def test_folder_convention_appends_correct_labels(self, tmp_path):
        input_dir = tmp_path / "captures"
        (input_dir / "positive").mkdir(parents=True)
        (input_dir / "negative").mkdir(parents=True)
        _write_chirp_bin(input_dir / "positive" / "p1.bin")
        _write_noise_bin(input_dir / "negative" / "n1.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, recursive=True, skip_unlabeled=True)
        exit_code = btd.run(args)

        assert exit_code == 0
        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        labels = {Path(row["source_file"]).name: row["label"] for row in rows}
        assert labels["p1.bin"] == "1"
        assert labels["n1.bin"] == "0"
        assert all(row["is_synthetic"] == "0" for row in rows)

    def test_rerun_deduplicates_by_source_file(self, tmp_path):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "p1.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        btd.run(args)
        first_count = len(existing_source_files(dataset))

        btd.run(args)  # re-run against the same folder
        second_count = len(existing_source_files(dataset))

        assert first_count == 1
        assert second_count == 1  # not doubled

    def test_reprocess_flag_adds_duplicate_row(self, tmp_path):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "p1.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        btd.run(args)

        args_reprocess = _base_args(tmp_path, input_dir, dataset, label=1, reprocess=True)
        btd.run(args_reprocess)

        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2

    def test_skip_unlabeled_skips_files_with_no_resolvable_label(self, tmp_path):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "mystery.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, skip_unlabeled=True)  # no manifest, no folder convention, no --label
        exit_code = btd.run(args)

        assert exit_code == 0
        assert not dataset.exists() or existing_source_files(dataset) == set()

    def test_corrupt_file_does_not_abort_the_batch(self, tmp_path):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "good.bin")
        # A .bin whose size isn't a multiple of complex64's itemsize (8 bytes)
        # makes np.memmap raise - a realistic "corrupt/truncated capture".
        (input_dir / "corrupt.bin").write_bytes(b"\x00\x00\x00")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        exit_code = btd.run(args)

        assert exit_code == 0  # at least one file succeeded
        rows = list(existing_source_files(dataset))
        assert len(rows) == 1
        assert rows[0].endswith("good.bin")

    def test_manifest_csv_end_to_end(self, tmp_path):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "capA.bin")
        _write_noise_bin(input_dir / "capB.bin")
        manifest = tmp_path / "labels.csv"
        manifest.write_text("filename,label\ncapA.bin,1\ncapB.bin,0\n")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, labels_csv=manifest)
        exit_code = btd.run(args)

        assert exit_code == 0
        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        labels = {Path(row["source_file"]).name: row["label"] for row in rows}
        assert labels == {"capA.bin": "1", "capB.bin": "0"}

    def test_empty_input_dir_returns_success_with_no_rows(self, tmp_path):
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        exit_code = btd.run(args)

        assert exit_code == 0
        assert not dataset.exists()

    def test_missing_input_dir_fails_cleanly(self, tmp_path):
        args = _base_args(tmp_path, tmp_path / "does_not_exist", tmp_path / "features.csv", label=1)
        assert btd.run(args) == 1
