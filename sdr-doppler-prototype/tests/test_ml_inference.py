"""Tests for ML inference (src/detection/ml_detector.py).

The trained model used here is fit on a small, explicitly SYNTHETIC
dataset generated in-memory purely to exercise inference (missing model,
invalid model, feature-order mismatch, successful predict) - it does not
represent real-world accuracy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detection.ml_detector import (  # noqa: E402
    AVAILABLE,
    MODEL_INVALID,
    MODEL_NOT_AVAILABLE,
    predict,
    run_ml_detection,
    try_load_model,
)
from features.extractor import FEATURE_NAMES, FeatureVector  # noqa: E402
from ml.model import ModelBundle  # noqa: E402


def _synthetic_feature_vector(**overrides) -> FeatureVector:
    defaults = dict(
        snr_db=15.0,
        frequency_drift_hz=12000.0,
        drift_rate_hz_per_second=400.0,
        smoothness_score=120.0,
        valid_signal_ratio=0.85,
        peak_power=10.0,
        mean_power=-60.0,
        occupied_bandwidth_hz=3000.0,
        signal_duration_seconds=30.0,
        strongest_trace_hz=np.array([1.0, 2.0]),
        smoothed_trace_hz=np.array([1.0, 2.0]),
        valid_mask=np.array([True, True]),
        time_axis_is_synthetic=False,
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


def _train_synthetic_bundle(seed: int = 42) -> ModelBundle:
    """A tiny RandomForest fit on synthetic data, wrapped as a ModelBundle - test fixture only."""
    rng = np.random.default_rng(seed)
    n = 20
    X_pos = rng.normal(loc=[15, 12000, 400, 120, 0.85, 10, -60, 3000, 30], scale=1, size=(n, 9))
    X_neg = rng.normal(loc=[1, 200, 10, 9000, 0.05, 2, -75, 400, 4], scale=1, size=(n, 9))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n + [0] * n)
    model = RandomForestClassifier(n_estimators=10, random_state=seed)
    model.fit(X, y)
    metadata = {
        "model_type": "RandomForestClassifier",
        "model_version": "test-synthetic-v1",
        "trained_on_synthetic_data": True,
    }
    return ModelBundle(classifier=model, feature_names=FEATURE_NAMES, metadata=metadata)


class TestTryLoadModel:
    def test_missing_file_returns_none(self, tmp_path):
        assert try_load_model(tmp_path / "does_not_exist.joblib") is None

    def test_none_path_returns_none(self):
        assert try_load_model(None) is None

    def test_valid_model_loads(self, tmp_path):
        bundle = _train_synthetic_bundle()
        model_path = tmp_path / "model.joblib"
        bundle.save(model_path)

        loaded = try_load_model(model_path)

        assert loaded is not None
        assert loaded.feature_names == FEATURE_NAMES

    def test_model_file_without_metadata_sidecar_returns_none(self, tmp_path):
        import joblib

        model_path = tmp_path / "model.joblib"
        joblib.dump(RandomForestClassifier(), model_path)  # no matching .json sidecar

        assert try_load_model(model_path) is None

    def test_corrupt_model_file_returns_none(self, tmp_path):
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"this is not a valid joblib file")
        model_path.with_suffix(".json").write_text(json.dumps({"feature_names": list(FEATURE_NAMES)}))

        assert try_load_model(model_path) is None

    def test_metadata_with_no_feature_names_returns_none(self, tmp_path):
        bundle = _train_synthetic_bundle()
        model_path = tmp_path / "model.joblib"
        bundle.save(model_path)
        model_path.with_suffix(".json").write_text(json.dumps({"model_type": "RandomForestClassifier"}))

        assert try_load_model(model_path) is None


class TestPredict:
    def test_returns_available_result_for_valid_bundle(self):
        bundle = _train_synthetic_bundle()
        features = _synthetic_feature_vector()

        result = predict(bundle, features)

        assert result.status == AVAILABLE
        assert result.available is True
        assert isinstance(result.ml_detection_result, bool)
        assert 0.0 <= result.ml_confidence_score <= 1.0
        assert result.model_version == "test-synthetic-v1"

    def test_positive_like_features_predict_candidate(self):
        bundle = _train_synthetic_bundle()
        features = _synthetic_feature_vector()  # matches the "positive" synthetic cluster

        result = predict(bundle, features)

        assert result.ml_detection_result is True
        assert result.ml_confidence_score > 0.5

    def test_negative_like_features_predict_non_candidate(self):
        bundle = _train_synthetic_bundle()
        features = _synthetic_feature_vector(
            snr_db=1.0, frequency_drift_hz=200.0, drift_rate_hz_per_second=10.0,
            smoothness_score=9000.0, valid_signal_ratio=0.05, peak_power=2.0,
            mean_power=-75.0, occupied_bandwidth_hz=400.0, signal_duration_seconds=4.0,
        )

        result = predict(bundle, features)

        assert result.ml_detection_result is False
        assert result.ml_confidence_score < 0.5

    def test_feature_order_mismatch_returns_model_invalid(self):
        bundle = _train_synthetic_bundle()
        bundle.feature_names = tuple(list(FEATURE_NAMES[:-1]) + ["totally_unknown_feature"])
        features = _synthetic_feature_vector()

        result = predict(bundle, features)

        assert result.status == MODEL_INVALID
        assert result.available is False
        assert "totally_unknown_feature" in result.reason


class TestRunMlDetection:
    def test_missing_model_reports_not_available(self, tmp_path):
        features = _synthetic_feature_vector()

        result = run_ml_detection(tmp_path / "no_such_model.joblib", features)

        assert result.status == MODEL_NOT_AVAILABLE
        assert result.available is False
        assert result.ml_detection_result is None
        assert result.ml_confidence_score is None

    def test_available_model_end_to_end(self, tmp_path):
        bundle = _train_synthetic_bundle()
        model_path = tmp_path / "model.joblib"
        bundle.save(model_path)
        features = _synthetic_feature_vector()

        result = run_ml_detection(model_path, features)

        assert result.status == AVAILABLE
        assert result.model_version == "test-synthetic-v1"

    def test_pipeline_never_raises_on_missing_model(self, tmp_path):
        """Mirrors the 'must not crash the pipeline' requirement."""
        features = _synthetic_feature_vector()
        try:
            result = run_ml_detection(tmp_path / "missing.joblib", features)
        except Exception as exc:  # pragma: no cover - the point of this test is that this never happens
            pytest.fail(f"run_ml_detection raised instead of returning MODEL_NOT_AVAILABLE: {exc}")
        assert result.status == MODEL_NOT_AVAILABLE
