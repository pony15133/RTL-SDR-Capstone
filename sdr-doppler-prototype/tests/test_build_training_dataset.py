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

from features.dataset_io import CSV_COLUMNS, existing_source_files  # noqa: E402
from features.extractor import FEATURE_NAMES, SMOOTHNESS_INF_SENTINEL_HZ  # noqa: E402


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


def _chirp_iq(samples: int = 4096, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / 240_000.0
    rate = (30_000.0 - (-30_000.0)) / t[-1]
    phase = 2 * np.pi * (-30_000.0 * t + 0.5 * rate * t * t)
    noise = 0.1 * (rng.normal(size=samples) + 1j * rng.normal(size=samples))
    return (np.exp(1j * phase) + noise).astype(np.complex64)


def _write_chirp_npy(path: Path, samples: int = 4096, seed: int = 1) -> None:
    """A synthetic drifting-tone raw-IQ .npy fixture (1D complex64), positive-shaped."""
    np.save(path, _chirp_iq(samples, seed))


def _write_noise_npy(path: Path, samples: int = 4096, seed: int = 2) -> None:
    """A synthetic pure-noise raw-IQ .npy fixture (1D complex64), negative-shaped."""
    rng = np.random.default_rng(seed)
    noise = (0.2 * (rng.normal(size=samples) + 1j * rng.normal(size=samples))).astype(np.complex64)
    np.save(path, noise)


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


class TestParseClassMap:
    def test_default_includes_satellite_and_noise(self):
        class_map = btd.parse_class_map(None)
        assert class_map["satellite"] == 1
        assert class_map["noise"] == 0

    def test_empty_string_returns_defaults(self):
        assert btd.parse_class_map("") == btd.DEFAULT_CLASS_MAP

    def test_override_existing_entry(self):
        class_map = btd.parse_class_map("noise=1")
        assert class_map["noise"] == 1
        assert class_map["satellite"] == 1  # unrelated defaults untouched

    def test_add_new_entry(self):
        class_map = btd.parse_class_map("interference=0,unknown_sat=1")
        assert class_map["interference"] == 0
        assert class_map["unknown_sat"] == 1
        assert class_map["satellite"] == 1  # defaults still present

    def test_is_case_insensitive_on_folder_name(self):
        class_map = btd.parse_class_map("Interference=0")
        assert class_map["interference"] == 0

    def test_missing_equals_sign_raises(self):
        with pytest.raises(SystemExit):
            btd.parse_class_map("not_valid")

    def test_out_of_range_label_raises(self):
        with pytest.raises(SystemExit):
            btd.parse_class_map("foo=2")

    def test_non_integer_label_raises(self):
        with pytest.raises(SystemExit):
            btd.parse_class_map("foo=maybe")


class TestResolveLabelWithSource:
    def test_satellite_folder_resolves_to_one(self, tmp_path):
        path = tmp_path / "satellite" / "meteor_m2_4_001.npy"
        path.parent.mkdir()
        path.touch()
        label, source = btd.resolve_label_with_source(path, tmp_path, {}, forced_label=None)
        assert (label, source) == (1, "folder:satellite")

    def test_noise_folder_resolves_to_zero(self, tmp_path):
        path = tmp_path / "noise" / "empty_001.npy"
        path.parent.mkdir()
        path.touch()
        label, source = btd.resolve_label_with_source(path, tmp_path, {}, forced_label=None)
        assert (label, source) == (0, "folder:noise")

    def test_manifest_source_tag(self, tmp_path):
        path = tmp_path / "a.bin"
        path.touch()
        label, source = btd.resolve_label_with_source(path, tmp_path, {"a.bin": 1}, forced_label=None)
        assert (label, source) == (1, "manifest")

    def test_forced_label_source_tag(self, tmp_path):
        path = tmp_path / "unlabelled" / "a.bin"
        path.parent.mkdir()
        path.touch()
        label, source = btd.resolve_label_with_source(path, tmp_path, {}, forced_label=1)
        assert (label, source) == (1, "forced")

    def test_unresolved_returns_none_source(self, tmp_path):
        path = tmp_path / "unlabelled" / "a.bin"
        path.parent.mkdir()
        path.touch()
        label, source = btd.resolve_label_with_source(path, tmp_path, {}, forced_label=None)
        assert (label, source) == (None, None)

    def test_custom_class_map_extends_default(self, tmp_path):
        path = tmp_path / "interference" / "a.bin"
        path.parent.mkdir()
        path.touch()
        class_map = btd.parse_class_map("interference=0")
        label, source = btd.resolve_label_with_source(path, tmp_path, {}, forced_label=None, class_map=class_map)
        assert (label, source) == (0, "folder:interference")


class TestSchemaMatchesTrainingPipeline:
    """CSV column set/order must exactly match features.dataset_io.CSV_COLUMNS
    (which is itself built from FEATURE_NAMES) - the schema
    src/ml/train.py's REQUIRED_COLUMNS and split_features_labels() expect.
    """

    def test_header_matches_csv_columns_exactly(self, tmp_path):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "p1.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        btd.run(args)

        with dataset.open(newline="") as handle:
            header = next(csv.reader(handle))
        assert header == CSV_COLUMNS

    def test_every_feature_name_is_a_column(self, tmp_path):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "p1.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        btd.run(args)

        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        for name in FEATURE_NAMES:
            assert name in rows[0]
            float(rows[0][name])  # every feature value must parse as a finite number
            assert np.isfinite(float(rows[0][name]))

    def test_label_column_present_and_binary(self, tmp_path):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "p1.bin")
        _write_noise_bin(input_dir / "n1.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        btd.run(args)
        args2 = _base_args(tmp_path, input_dir, dataset, label=0, reprocess=True)
        btd.run(args2)

        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert all(row["label"] in ("0", "1") for row in rows)


class TestNpyRawIq:
    """The dataset builder must handle raw-IQ .npy the same way it handles .bin -
    both go through load_input() -> iq_to_spectrogram() -> extract_features()."""

    def test_npy_satellite_noise_folders_out_of_the_box(self, tmp_path):
        """The exact directory layout from the project's dataset-building workflow:
        training_data/satellite/*.npy -> label 1, training_data/noise/*.npy -> label 0,
        resolved purely from folder names with zero extra CLI flags."""
        input_dir = tmp_path / "training_data"
        (input_dir / "satellite").mkdir(parents=True)
        (input_dir / "noise").mkdir(parents=True)
        _write_chirp_npy(input_dir / "satellite" / "meteor_m2_4_001.npy")
        _write_noise_npy(input_dir / "noise" / "empty_001.npy")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, recursive=True, skip_unlabeled=True)
        exit_code = btd.run(args)

        assert exit_code == 0
        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        labels = {Path(row["source_file"]).name: row["label"] for row in rows}
        assert labels == {"meteor_m2_4_001.npy": "1", "empty_001.npy": "0"}


class TestSummaryOutput:
    def test_prints_processed_successful_failed_and_output(self, tmp_path, capsys):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "p1.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        btd.run(args)

        out = capsys.readouterr().out
        assert "Processed: 1" in out
        assert "Successful: 1" in out
        assert "Failed: 0" in out
        assert f"Output: {dataset}" in out

    def test_prints_class_distribution_by_label_and_folder(self, tmp_path, capsys):
        input_dir = tmp_path / "training_data"
        (input_dir / "satellite").mkdir(parents=True)
        (input_dir / "noise").mkdir(parents=True)
        _write_chirp_bin(input_dir / "satellite" / "p1.bin")
        _write_noise_bin(input_dir / "noise" / "n1.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, recursive=True, skip_unlabeled=True)
        btd.run(args)

        out = capsys.readouterr().out
        assert "Class distribution (this run, by label):" in out
        assert "Class distribution (this run, by source folder):" in out
        assert "satellite: 1" in out
        assert "noise: 1" in out

    def test_warns_when_only_one_class_present(self, tmp_path, capsys):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "p1.bin")
        _write_chirp_bin(input_dir / "p2.bin", seed=3)
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        btd.run(args)

        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "only one class" in out

    def test_warns_when_dataset_is_imbalanced(self, tmp_path, capsys):
        input_dir = tmp_path / "training_data"
        (input_dir / "satellite").mkdir(parents=True)
        (input_dir / "noise").mkdir(parents=True)
        _write_chirp_bin(input_dir / "satellite" / "p1.bin", seed=1)
        for i in range(4):
            _write_noise_bin(input_dir / "noise" / f"n{i}.bin", seed=100 + i)
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, recursive=True, skip_unlabeled=True)
        btd.run(args)

        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "imbalanced" in out

    def test_no_imbalance_warning_when_classes_are_balanced(self, tmp_path, capsys):
        input_dir = tmp_path / "training_data"
        (input_dir / "satellite").mkdir(parents=True)
        (input_dir / "noise").mkdir(parents=True)
        _write_chirp_bin(input_dir / "satellite" / "p1.bin")
        _write_noise_bin(input_dir / "noise" / "n1.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, recursive=True, skip_unlabeled=True)
        btd.run(args)

        out = capsys.readouterr().out
        assert "imbalanced" not in out


class TestDatasetSchemaMismatchAborts:
    def test_incompatible_existing_header_fails_cleanly_without_crashing(self, tmp_path, capsys):
        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "p1.bin")
        dataset = tmp_path / "features.csv"
        dataset.write_text("capture_id,some_old_column,label\ncap0,1.0,1\n")  # old/incompatible schema

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        exit_code = btd.run(args)

        assert exit_code == 1
        err = capsys.readouterr().err
        assert "FAILED" in err
        # the pre-existing row must not have been touched/duplicated
        assert dataset.read_text().count("cap0") == 1


class TestNonFiniteFeatureHandling:
    """A FeatureVector with an infinite smoothness_score (legitimate when a
    capture has fewer than 3 valid trace points, see calculate_smoothness())
    must be written to the CSV the same sanitised way feature_vector_to_array()
    already handles it everywhere else in the pipeline - never a literal
    'inf' that pandas/sklearn can't consume.
    """

    def test_infinite_smoothness_is_written_as_finite_sentinel(self, tmp_path, monkeypatch):
        from features.extractor import FeatureVector

        def fake_extract_features(spec, snr_threshold_db, time_axis_is_synthetic=False):
            return FeatureVector(
                snr_db=5.0,
                frequency_drift_hz=100.0,
                drift_rate_hz_per_second=10.0,
                smoothness_score=float("inf"),
                valid_signal_ratio=0.5,
                peak_power=-40.0,
                mean_power=-60.0,
                occupied_bandwidth_hz=500.0,
                signal_duration_seconds=1.0,
                strongest_trace_hz=np.array([]),
                smoothed_trace_hz=np.array([]),
                valid_mask=np.array([], dtype=bool),
                time_axis_is_synthetic=time_axis_is_synthetic,
            )

        monkeypatch.setattr(btd, "extract_features", fake_extract_features)

        input_dir = tmp_path / "captures"
        input_dir.mkdir()
        _write_chirp_bin(input_dir / "p1.bin")
        dataset = tmp_path / "features.csv"

        args = _base_args(tmp_path, input_dir, dataset, label=1)
        exit_code = btd.run(args)

        assert exit_code == 0
        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["smoothness_score"] == str(SMOOTHNESS_INF_SENTINEL_HZ)
        assert np.isfinite(float(rows[0]["smoothness_score"]))
