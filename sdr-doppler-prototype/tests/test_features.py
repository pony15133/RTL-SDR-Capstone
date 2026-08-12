"""Tests for the shared feature extractor (src/features/extractor.py).

Spectrograms here are synthetic test fixtures, built the same way
tests/test_detect.py builds its rule-detector fixture - not real RTL-SDR
captures.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.extractor import (  # noqa: E402
    FEATURE_NAMES,
    FeatureSchemaError,
    SMOOTHNESS_INF_SENTINEL_HZ,
    extract_features,
    extract_strongest_trace,
    feature_vector_to_array,
)
from spectrogram import SpectrogramData  # noqa: E402


def _drifting_signal_spectrogram() -> SpectrogramData:
    """A synthetic spectrogram with a clean, smooth, drifting trace - same shape as test_detect.py's fixture."""
    rows, cols = 80, 128
    rng = np.random.default_rng(123)
    power = rng.normal(-80, 2, size=(rows, cols))
    bins = np.linspace(30, 90, rows).astype(int)
    power[np.arange(rows), bins] = -45
    return SpectrogramData(
        power_db=power,
        frequencies_hz=np.linspace(136.7e6, 136.9e6, cols),
        times_s=np.arange(rows) * 0.05,  # 50ms per time slice -> real seconds
    )


def _pure_noise_spectrogram() -> SpectrogramData:
    rng = np.random.default_rng(999)
    power = rng.normal(-80, 2, size=(60, 128))
    return SpectrogramData(
        power_db=power,
        frequencies_hz=np.linspace(136.7e6, 136.9e6, 128),
        times_s=np.arange(60) * 0.05,
    )


class TestExtractFeatures:
    def test_returns_all_ml_feature_names(self):
        features = extract_features(_drifting_signal_spectrogram(), snr_threshold_db=8)
        as_dict = features.as_dict()
        assert set(as_dict.keys()) == set(FEATURE_NAMES)

    def test_drifting_signal_has_positive_drift_and_bandwidth(self):
        features = extract_features(_drifting_signal_spectrogram(), snr_threshold_db=8)
        assert features.frequency_drift_hz > 0
        assert features.valid_signal_ratio > 0.9
        assert features.occupied_bandwidth_hz >= 0
        assert features.signal_duration_seconds > 0
        assert features.snr_db > 0  # strong synthetic signal should read as positive SNR

    def test_drift_rate_matches_drift_over_duration(self):
        features = extract_features(_drifting_signal_spectrogram(), snr_threshold_db=8)
        if features.signal_duration_seconds > 0:
            expected_rate = features.frequency_drift_hz / features.signal_duration_seconds
            assert features.drift_rate_hz_per_second == pytest.approx(expected_rate)

    def test_pure_noise_has_low_valid_ratio(self):
        features = extract_features(_pure_noise_spectrogram(), snr_threshold_db=8)
        assert features.valid_signal_ratio < 0.2

    def test_pure_noise_does_not_crash_even_with_few_valid_points(self):
        # smoothness_score may legitimately be inf here (< 3 finite trace
        # points) - extract_features must not raise regardless.
        features = extract_features(_pure_noise_spectrogram(), snr_threshold_db=8)
        assert isinstance(features.peak_power, float)
        assert isinstance(features.mean_power, float)

    def test_time_axis_is_synthetic_flag_defaults_false(self):
        features = extract_features(_drifting_signal_spectrogram(), snr_threshold_db=8)
        assert features.time_axis_is_synthetic is False

    def test_time_axis_is_synthetic_flag_propagates(self):
        features = extract_features(_drifting_signal_spectrogram(), snr_threshold_db=8, time_axis_is_synthetic=True)
        assert features.time_axis_is_synthetic is True

    def test_peak_power_and_mean_power_are_plausible(self):
        spec = _drifting_signal_spectrogram()
        features = extract_features(spec, snr_threshold_db=8)
        assert features.peak_power == pytest.approx(float(np.max(spec.power_db)))
        assert features.mean_power == pytest.approx(float(np.mean(spec.power_db)))


class TestExtractStrongestTraceBackwardCompatibility:
    def test_still_returns_two_arrays(self):
        """The rule detector (detect.py) relies on this exact 2-tuple shape."""
        trace_hz, valid_mask = extract_strongest_trace(_drifting_signal_spectrogram(), snr_threshold_db=8)
        assert trace_hz.shape == (80,)
        assert valid_mask.shape == (80,)
        assert valid_mask.dtype == bool


class TestFeatureVectorToArray:
    def test_matches_as_dict_order_and_values(self):
        features = extract_features(_drifting_signal_spectrogram(), snr_threshold_db=8)
        array = feature_vector_to_array(features)
        as_dict = features.as_dict()
        expected = [as_dict[name] for name in FEATURE_NAMES]
        np.testing.assert_allclose(array, expected)

    def test_replaces_infinite_smoothness_with_sentinel(self):
        features = extract_features(_pure_noise_spectrogram(), snr_threshold_db=8)
        # Force the scenario this guards, regardless of what the noise fixture happened to produce.
        features.smoothness_score = float("inf")
        array = feature_vector_to_array(features)
        smoothness_index = FEATURE_NAMES.index("smoothness_score")
        assert array[smoothness_index] == SMOOTHNESS_INF_SENTINEL_HZ
        assert np.isfinite(array).all()

    def test_raises_on_unknown_feature_name(self):
        features = extract_features(_drifting_signal_spectrogram(), snr_threshold_db=8)
        with pytest.raises(FeatureSchemaError):
            feature_vector_to_array(features, feature_names=("snr_db", "not_a_real_feature"))

    def test_custom_feature_order_is_respected(self):
        features = extract_features(_drifting_signal_spectrogram(), snr_threshold_db=8)
        reordered = ("mean_power", "snr_db")
        array = feature_vector_to_array(features, feature_names=reordered)
        assert array[0] == pytest.approx(features.mean_power)
        assert array[1] == pytest.approx(features.snr_db)
