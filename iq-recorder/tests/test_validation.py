"""Tests for parameter validation in rtl_recorder.config."""

import pytest

from rtl_recorder.config import validate_duration, validate_frequency, validate_gain, validate_sample_rate
from rtl_recorder.exceptions import ValidationError


class TestFrequencyValidation:
    def test_accepts_valid_frequency(self):
        assert validate_frequency(137_900_000) == 137_900_000

    def test_accepts_kiss92_frequency(self):
        assert validate_frequency(92_000_000) == 92_000_000

    def test_accepts_numeric_string(self):
        assert validate_frequency("137900000") == 137_900_000

    def test_rejects_too_low(self):
        with pytest.raises(ValidationError):
            validate_frequency(1000)

    def test_rejects_too_high(self):
        with pytest.raises(ValidationError):
            validate_frequency(5_000_000_000)

    def test_rejects_non_numeric(self):
        with pytest.raises(ValidationError):
            validate_frequency("not-a-frequency")

    def test_rejects_none(self):
        with pytest.raises(ValidationError):
            validate_frequency(None)


class TestSampleRateValidation:
    def test_accepts_2_4_msps(self):
        assert validate_sample_rate(2_400_000) == 2_400_000

    def test_accepts_low_range(self):
        assert validate_sample_rate(250_000) == 250_000

    def test_rejects_gap_between_ranges(self):
        with pytest.raises(ValidationError):
            validate_sample_rate(500_000)

    def test_rejects_zero(self):
        with pytest.raises(ValidationError):
            validate_sample_rate(0)

    def test_rejects_negative(self):
        with pytest.raises(ValidationError):
            validate_sample_rate(-2_400_000)

    def test_rejects_non_numeric(self):
        with pytest.raises(ValidationError):
            validate_sample_rate("fast")


class TestGainValidation:
    def test_accepts_numeric_gain(self):
        assert validate_gain(30) == 30.0

    def test_accepts_float_gain(self):
        assert validate_gain(29.7) == 29.7

    def test_none_means_auto(self):
        assert validate_gain(None) is None

    def test_auto_string_means_auto(self):
        assert validate_gain("auto") is None
        assert validate_gain("AUTO") is None

    def test_rejects_negative_gain(self):
        with pytest.raises(ValidationError):
            validate_gain(-5)

    def test_rejects_excessive_gain(self):
        with pytest.raises(ValidationError):
            validate_gain(999)

    def test_rejects_non_numeric(self):
        with pytest.raises(ValidationError):
            validate_gain("loud")


class TestDurationValidation:
    def test_accepts_normal_duration(self):
        assert validate_duration(60) == 60.0

    def test_accepts_short_duration(self):
        assert validate_duration(1) == 1.0

    def test_rejects_zero(self):
        with pytest.raises(ValidationError):
            validate_duration(0)

    def test_rejects_negative(self):
        with pytest.raises(ValidationError):
            validate_duration(-30)

    def test_rejects_absurdly_long(self):
        with pytest.raises(ValidationError):
            validate_duration(10_000_000)

    def test_rejects_non_numeric(self):
        with pytest.raises(ValidationError):
            validate_duration("a while")
