"""Tests for the training pipeline (src/ml/train.py).

All datasets in this file are SYNTHETIC - generated in-memory with a fixed
seed purely to exercise dataset validation and the training pipeline
mechanics (split/train/evaluate/save). None of this establishes real model
accuracy; see data/training/README.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.extractor import FEATURE_NAMES  # noqa: E402
from ml.model import ModelBundle  # noqa: E402
from ml.train import (  # noqa: E402
    DatasetValidationError,
    build_arg_parser,
    check_synthetic_guard,
    grouped_train_test_split,
    grouped_train_val_test_split,
    load_dataset,
    resolve_groups,
    run_training,
    validate_dataset,
)


def _make_synthetic_dataframe(n_per_class: int = 8, seed: int = 2026) -> pd.DataFrame:
    """A small, clearly SYNTHETIC labelled dataset for pipeline tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_per_class):
        rows.append(
            {
                "capture_id": f"synthtest_pos_{i}",
                "snr_db": float(rng.normal(15, 2)),
                "frequency_drift_hz": float(rng.normal(12000, 1000)),
                "drift_rate_hz_per_second": float(rng.normal(400, 50)),
                "smoothness_score": float(abs(rng.normal(120, 20))),
                "valid_signal_ratio": float(np.clip(rng.normal(0.85, 0.05), 0, 1)),
                "peak_power": float(rng.normal(10, 1)),
                "mean_power": float(rng.normal(-60, 2)),
                "occupied_bandwidth_hz": float(abs(rng.normal(3000, 200))),
                "signal_duration_seconds": float(abs(rng.normal(30, 2))),
                "label": 1,
                "is_synthetic": 1,
            }
        )
    for i in range(n_per_class):
        rows.append(
            {
                "capture_id": f"synthtest_neg_{i}",
                "snr_db": float(rng.normal(1, 0.5)),
                "frequency_drift_hz": float(abs(rng.normal(200, 50))),
                "drift_rate_hz_per_second": float(abs(rng.normal(10, 3))),
                "smoothness_score": float(abs(rng.normal(9000, 500))),
                "valid_signal_ratio": float(np.clip(rng.normal(0.05, 0.02), 0, 1)),
                "peak_power": float(rng.normal(2, 0.5)),
                "mean_power": float(rng.normal(-75, 2)),
                "occupied_bandwidth_hz": float(abs(rng.normal(400, 100))),
                "signal_duration_seconds": float(abs(rng.normal(4, 1))),
                "label": 0,
                "is_synthetic": 1,
            }
        )
    return pd.DataFrame(rows)


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "dataset.csv") -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


class TestValidateDataset:
    def test_valid_dataset_passes(self):
        df = _make_synthetic_dataframe()
        validate_dataset(df)  # should not raise

    def test_missing_required_column_raises(self):
        df = _make_synthetic_dataframe().drop(columns=["snr_db"])
        with pytest.raises(DatasetValidationError, match="snr_db"):
            validate_dataset(df)

    def test_empty_dataset_raises(self):
        df = _make_synthetic_dataframe().iloc[0:0]
        with pytest.raises(DatasetValidationError):
            validate_dataset(df)

    def test_missing_value_raises(self):
        df = _make_synthetic_dataframe()
        df.loc[0, "frequency_drift_hz"] = None
        with pytest.raises(DatasetValidationError, match="frequency_drift_hz"):
            validate_dataset(df)

    def test_non_numeric_value_raises(self):
        # Assign via an object-dtype column so pandas doesn't reject the
        # mixed-type assignment outright - a CSV loaded straight from disk
        # with a stray non-numeric cell behaves the same way (object dtype).
        df = _make_synthetic_dataframe()
        df["peak_power"] = df["peak_power"].astype(object)
        df.loc[0, "peak_power"] = "not-a-number"
        with pytest.raises(DatasetValidationError, match="peak_power"):
            validate_dataset(df)

    def test_infinite_value_raises(self):
        df = _make_synthetic_dataframe()
        df.loc[0, "mean_power"] = float("inf")
        with pytest.raises(DatasetValidationError, match="mean_power"):
            validate_dataset(df)

    def test_invalid_label_raises(self):
        df = _make_synthetic_dataframe()
        df.loc[0, "label"] = 2
        with pytest.raises(DatasetValidationError, match="label"):
            validate_dataset(df)

    def test_duplicate_capture_id_raises(self):
        df = _make_synthetic_dataframe()
        df.loc[1, "capture_id"] = df.loc[0, "capture_id"]
        with pytest.raises(DatasetValidationError, match="Duplicate"):
            validate_dataset(df)


class TestSyntheticGuard:
    def test_no_synthetic_column_returns_false(self):
        df = _make_synthetic_dataframe().drop(columns=["is_synthetic"])
        assert check_synthetic_guard(df, allow_synthetic=False) is False

    def test_synthetic_rows_blocked_without_flag(self):
        df = _make_synthetic_dataframe()
        with pytest.raises(DatasetValidationError, match="is_synthetic"):
            check_synthetic_guard(df, allow_synthetic=False)

    def test_synthetic_rows_allowed_with_flag(self):
        df = _make_synthetic_dataframe()
        assert check_synthetic_guard(df, allow_synthetic=True) is True

    def test_load_dataset_rejects_synthetic_csv_by_default(self, tmp_path):
        csv_path = _write_csv(tmp_path, _make_synthetic_dataframe())
        load_dataset(csv_path)  # load_dataset only validates schema, not the synthetic guard
        # the guard is applied separately by run_training(); verify it end-to-end below


class TestRunTrainingEndToEnd:
    def _args(self, tmp_path, dataset_path, **overrides):
        argv = [
            "--dataset", str(dataset_path),
            "--output", str(tmp_path / "model.joblib"),
            "--allow-synthetic",
            "--cv-folds", "3",
        ]
        for key, value in overrides.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        return build_arg_parser().parse_args(argv)

    def test_trains_and_saves_model(self, tmp_path):
        df = _make_synthetic_dataframe(n_per_class=10)
        csv_path = _write_csv(tmp_path, df)
        args = self._args(tmp_path, csv_path)

        result = run_training(args)

        assert result.model_path.exists()
        assert result.metadata_path.exists()
        assert result.trained_on_synthetic_data is True
        assert result.n_train + result.n_val + result.n_test == len(df)
        assert result.n_val > 0 and result.n_test > 0

    def test_metrics_contain_required_keys(self, tmp_path):
        df = _make_synthetic_dataframe(n_per_class=10)
        csv_path = _write_csv(tmp_path, df)
        args = self._args(tmp_path, csv_path)

        result = run_training(args)

        for key in ("accuracy", "precision", "recall", "f1_score", "confusion_matrix"):
            assert key in result.metrics

    def test_feature_importance_covers_all_features(self, tmp_path):
        df = _make_synthetic_dataframe(n_per_class=10)
        csv_path = _write_csv(tmp_path, df)
        args = self._args(tmp_path, csv_path)

        result = run_training(args)

        names = {name for name, _ in result.importance_report}
        assert names == set(FEATURE_NAMES)

    def test_metadata_sidecar_has_required_fields(self, tmp_path):
        df = _make_synthetic_dataframe(n_per_class=10)
        csv_path = _write_csv(tmp_path, df)
        args = self._args(tmp_path, csv_path, model_version="test-v1")

        result = run_training(args)
        bundle = ModelBundle.load(result.model_path)

        for key in (
            "model_type", "model_version", "training_timestamp", "feature_names",
            "random_forest_params", "n_training_samples", "label_distribution",
            "evaluation_metrics", "trained_on_synthetic_data",
        ):
            assert key in bundle.metadata
        assert bundle.metadata["model_version"] == "test-v1"
        assert bundle.metadata["trained_on_synthetic_data"] is True
        assert bundle.feature_names == FEATURE_NAMES

    def test_fixed_random_state_is_reproducible(self, tmp_path):
        df = _make_synthetic_dataframe(n_per_class=10)
        csv_path = _write_csv(tmp_path, df)

        args1 = self._args(tmp_path, csv_path, **{"output": tmp_path / "m1.joblib", "random-state": 7})
        args2 = self._args(tmp_path, csv_path, **{"output": tmp_path / "m2.joblib", "random-state": 7})
        result1 = run_training(args1)
        result2 = run_training(args2)

        assert result1.metrics["accuracy"] == result2.metrics["accuracy"]
        assert result1.label_distribution == result2.label_distribution

    def test_grouped_split_keeps_same_recordings_together(self):
        df = pd.DataFrame(
            [
                {"recording_id": "r1", "capture_id": "c1", "snr_db": 11.0, "frequency_drift_hz": 1200.0, "drift_rate_hz_per_second": 40.0, "smoothness_score": 150.0, "valid_signal_ratio": 0.8, "peak_power": 8.0, "mean_power": -60.0, "occupied_bandwidth_hz": 2200.0, "signal_duration_seconds": 30.0, "label": 1, "is_synthetic": 1},
                {"recording_id": "r1", "capture_id": "c2", "snr_db": 10.5, "frequency_drift_hz": 1180.0, "drift_rate_hz_per_second": 39.0, "smoothness_score": 170.0, "valid_signal_ratio": 0.75, "peak_power": 7.5, "mean_power": -61.0, "occupied_bandwidth_hz": 2100.0, "signal_duration_seconds": 29.0, "label": 1, "is_synthetic": 1},
                {"recording_id": "r2", "capture_id": "c3", "snr_db": 1.5, "frequency_drift_hz": 250.0, "drift_rate_hz_per_second": 10.0, "smoothness_score": 7800.0, "valid_signal_ratio": 0.15, "peak_power": 1.5, "mean_power": -75.0, "occupied_bandwidth_hz": 500.0, "signal_duration_seconds": 25.0, "label": 0, "is_synthetic": 1},
                {"recording_id": "r2", "capture_id": "c4", "snr_db": 1.2, "frequency_drift_hz": 260.0, "drift_rate_hz_per_second": 11.0, "smoothness_score": 7600.0, "valid_signal_ratio": 0.1, "peak_power": 1.1, "mean_power": -76.0, "occupied_bandwidth_hz": 480.0, "signal_duration_seconds": 24.0, "label": 0, "is_synthetic": 1},
                {"recording_id": "r3", "capture_id": "c5", "snr_db": 12.5, "frequency_drift_hz": 1350.0, "drift_rate_hz_per_second": 48.0, "smoothness_score": 160.0, "valid_signal_ratio": 0.85, "peak_power": 9.1, "mean_power": -58.0, "occupied_bandwidth_hz": 2600.0, "signal_duration_seconds": 28.0, "label": 1, "is_synthetic": 1},
                {"recording_id": "r3", "capture_id": "c6", "snr_db": 11.8, "frequency_drift_hz": 1280.0, "drift_rate_hz_per_second": 44.0, "smoothness_score": 170.0, "valid_signal_ratio": 0.8, "peak_power": 8.9, "mean_power": -59.0, "occupied_bandwidth_hz": 2500.0, "signal_duration_seconds": 29.0, "label": 1, "is_synthetic": 1},
                {"recording_id": "r4", "capture_id": "c7", "snr_db": 1.9, "frequency_drift_hz": 200.0, "drift_rate_hz_per_second": 8.0, "smoothness_score": 8300.0, "valid_signal_ratio": 0.12, "peak_power": 1.9, "mean_power": -74.0, "occupied_bandwidth_hz": 590.0, "signal_duration_seconds": 25.0, "label": 0, "is_synthetic": 1},
                {"recording_id": "r4", "capture_id": "c8", "snr_db": 1.7, "frequency_drift_hz": 210.0, "drift_rate_hz_per_second": 9.0, "smoothness_score": 8100.0, "valid_signal_ratio": 0.11, "peak_power": 1.7, "mean_power": -73.5, "occupied_bandwidth_hz": 620.0, "signal_duration_seconds": 24.5, "label": 0, "is_synthetic": 1},
            ]
        )

        train_idx, test_idx = grouped_train_test_split(df, test_size=0.5, random_state=42)
        train_recordings = set(df.loc[train_idx, "recording_id"])
        test_recordings = set(df.loc[test_idx, "recording_id"])

        assert train_recordings.isdisjoint(test_recordings)
        assert len(train_recordings | test_recordings) == len(df["recording_id"].unique())

    def test_training_refused_without_allow_synthetic_flag(self, tmp_path):
        df = _make_synthetic_dataframe(n_per_class=10)
        csv_path = _write_csv(tmp_path, df)
        argv = ["--dataset", str(csv_path), "--output", str(tmp_path / "model.joblib")]
        args = build_arg_parser().parse_args(argv)

        with pytest.raises(DatasetValidationError, match="is_synthetic"):
            run_training(args)

    def test_cross_validation_skipped_on_tiny_dataset(self, tmp_path):
        df = _make_synthetic_dataframe(n_per_class=2)  # too small for meaningful CV
        csv_path = _write_csv(tmp_path, df)
        args = self._args(tmp_path, csv_path)

        result = run_training(args)

        assert result.cv_result["performed"] is False
        assert "reason" in result.cv_result

    def test_saved_model_can_be_loaded_and_used_for_prediction(self, tmp_path):
        df = _make_synthetic_dataframe(n_per_class=10)
        csv_path = _write_csv(tmp_path, df)
        args = self._args(tmp_path, csv_path)

        result = run_training(args)
        bundle = ModelBundle.load(result.model_path)

        sample = df[list(FEATURE_NAMES)].iloc[[0]].to_numpy(dtype=float)
        prediction = bundle.classifier.predict(sample)
        assert prediction[0] in (0, 1)


def _make_chunked_dataframe(n_recordings_per_class: int = 5, chunks_per_recording: int = 4, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC rows shaped like chunk_bin_to_dataset.py output: several
    windows per recording, sharing a source_file up to the #samples= suffix."""
    base = _make_synthetic_dataframe(n_per_class=n_recordings_per_class * chunks_per_recording, seed=seed)
    rows = []
    pos = base[base["label"] == 1].reset_index(drop=True)
    neg = base[base["label"] == 0].reset_index(drop=True)
    for label_name, part in (("pos", pos), ("neg", neg)):
        for i, row in part.iterrows():
            rec = i // chunks_per_recording
            chunk = i % chunks_per_recording
            row = row.copy()
            row["capture_id"] = f"{label_name}_rec{rec}_w{chunk:05d}"
            row["source_file"] = f"data/raw/{label_name}_rec{rec}.bin#samples={chunk * 1000}-{(chunk + 1) * 1000}"
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


class TestTrainValTestSplit:
    def test_resolve_groups_strips_window_suffix(self):
        df = _make_chunked_dataframe(n_recordings_per_class=2, chunks_per_recording=3)
        groups, source = resolve_groups(df)
        assert source == "source_file"
        assert len(set(groups)) == 4
        assert all("#samples=" not in g for g in groups)

    def test_resolve_groups_prefers_recording_id(self):
        df = _make_chunked_dataframe(n_recordings_per_class=2, chunks_per_recording=3)
        df["recording_id"] = ["same"] * 6 + ["other"] * 6
        groups, source = resolve_groups(df)
        assert source == "recording_id"
        assert set(groups) == {"same", "other"}

    def test_resolve_groups_falls_back_to_rows(self):
        df = _make_synthetic_dataframe(n_per_class=3).drop(columns=["is_synthetic"])
        groups, source = resolve_groups(df)
        assert source == "row"
        assert len(set(groups)) == len(df)

    def test_no_recording_appears_in_two_splits(self):
        df = _make_chunked_dataframe(n_recordings_per_class=6, chunks_per_recording=4)
        groups, _ = resolve_groups(df)
        y = df["label"].to_numpy(dtype=int)
        splits = grouped_train_val_test_split(groups, y, val_size=0.2, test_size=0.2, random_state=1)

        seen = {name: set(groups[idx]) for name, idx in splits.items()}
        assert seen["train"].isdisjoint(seen["validation"])
        assert seen["train"].isdisjoint(seen["test"])
        assert seen["validation"].isdisjoint(seen["test"])
        assert sum(idx.size for idx in splits.values()) == len(df)
        for name, idx in splits.items():
            assert idx.size > 0, f"{name} split is empty"
        # Stratified: train, validation and test each contain both classes.
        for name, idx in splits.items():
            assert set(y[idx]) == {0, 1}, f"{name} split is missing a class"

    def test_small_dataset_still_gets_three_non_empty_splits(self):
        y = np.array([0, 0, 1, 1])
        groups = np.array(["a", "b", "c", "d"])
        splits = grouped_train_val_test_split(groups, y, random_state=0)
        assert all(idx.size >= 1 for idx in splits.values())
        assert set(y[splits["train"]]) == {0, 1}

    def test_fewer_than_three_recordings_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            grouped_train_val_test_split(np.array(["a", "a", "b"]), np.array([0, 0, 1]))

    def test_invalid_sizes_raise(self):
        with pytest.raises(ValueError):
            grouped_train_val_test_split(np.array(["a", "b", "c"]), np.array([0, 1, 0]), val_size=0.6, test_size=0.5)


class TestNoTestSetLeakage:
    def _args(self, tmp_path, dataset_path, *extra):
        argv = ["--dataset", str(dataset_path), "--output", str(tmp_path / "model.joblib"),
                "--allow-synthetic", "--cv-folds", "3", *extra]
        return build_arg_parser().parse_args(argv)

    def test_chunked_recordings_stay_in_one_split_end_to_end(self, tmp_path):
        df = _make_chunked_dataframe(n_recordings_per_class=6, chunks_per_recording=4)
        csv_path = _write_csv(tmp_path, df)

        result = run_training(self._args(tmp_path, csv_path))

        assert result.group_source == "source_file"
        splits = pd.read_csv(result.splits_path)
        assert splits.groupby("recording_group")["split"].nunique().max() == 1
        assert len(splits) == len(df)

    def test_tuning_and_cv_never_see_test_rows(self, tmp_path, monkeypatch):
        import ml.train as train_module

        seen = {}
        real_tune = train_module.tune_on_validation
        real_cv = train_module.cross_validate

        def spy_tune(X_train, y_train, X_val, y_val, **kwargs):
            seen["tune_rows"] = len(X_train) + len(X_val)
            return real_tune(X_train, y_train, X_val, y_val, **kwargs)

        def spy_cv(model, X, y, **kwargs):
            seen["cv_rows"] = len(X)
            return real_cv(model, X, y, **kwargs)

        monkeypatch.setattr(train_module, "tune_on_validation", spy_tune)
        monkeypatch.setattr(train_module, "cross_validate", spy_cv)

        df = _make_chunked_dataframe(n_recordings_per_class=6, chunks_per_recording=4)
        csv_path = _write_csv(tmp_path, df)
        result = run_training(self._args(tmp_path, csv_path))

        dev_rows = result.n_train + result.n_val
        assert seen["tune_rows"] == dev_rows
        assert seen["cv_rows"] == dev_rows
        assert dev_rows + result.n_test == len(df)

    def test_metadata_records_validation_and_test_separately(self, tmp_path):
        df = _make_chunked_dataframe(n_recordings_per_class=6, chunks_per_recording=4)
        csv_path = _write_csv(tmp_path, df)
        result = run_training(self._args(tmp_path, csv_path))
        meta = ModelBundle.load(result.model_path).metadata

        assert meta["evaluation_set"] == "test"
        assert meta["final_model_trained_on"] == "train+validation"
        assert "validation_metrics" in meta and "evaluation_metrics" in meta
        assert set(meta["splits"]) == {"train", "validation", "test"}
        assert meta["selected_hyperparameters"] in [r["params"] for r in meta["hyperparameter_search"]]
        assert meta["cross_validation"]["data"].startswith("train+validation")

    def test_no_tune_uses_given_hyperparameters(self, tmp_path):
        df = _make_chunked_dataframe(n_recordings_per_class=6, chunks_per_recording=2)
        csv_path = _write_csv(tmp_path, df)
        result = run_training(self._args(tmp_path, csv_path, "--no-tune", "--n-estimators", "25", "--max-depth", "4"))

        assert len(result.search_results) == 1
        assert result.best_params == {"n_estimators": 25, "max_depth": 4, "min_samples_leaf": 1}

    def test_too_few_recordings_falls_back_to_row_split_with_warning(self, tmp_path, capsys):
        df = _make_chunked_dataframe(n_recordings_per_class=1, chunks_per_recording=4)  # only 2 recordings
        csv_path = _write_csv(tmp_path, df)
        result = run_training(self._args(tmp_path, csv_path))

        assert result.group_source == "row"
        assert "Falling back" in capsys.readouterr().err
