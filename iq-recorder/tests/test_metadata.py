"""Tests for RecordingMetadata generation, (de)serialisation, and RecordingResult."""

import json
from datetime import datetime, timezone

from rtl_recorder.metadata import RecordingMetadata, RecordingResult, to_iso
from rtl_recorder.states import RecordingStatus


class TestToIso:
    def test_naive_datetime_assumed_utc(self):
        dt = datetime(2026, 8, 9, 19, 34, 30)
        assert to_iso(dt) == "2026-08-09T19:34:30+00:00"

    def test_none_stays_none(self):
        assert to_iso(None) is None

    def test_aware_datetime_converted_to_utc(self):
        from datetime import timedelta

        dt = datetime(2026, 8, 9, 3, 34, 30, tzinfo=timezone(timedelta(hours=8)))
        assert to_iso(dt) == "2026-08-08T19:34:30+00:00"


class TestRecordingMetadata:
    def _sample(self, **overrides) -> RecordingMetadata:
        defaults = dict(
            satellite_name="METEOR-M2-4",
            frequency_hz=137_900_000,
            sample_rate=2_400_000,
            gain=30.0,
            output_filename="METEOR-M2-4_20260809_193430_137900000Hz.iq",
            recording_status=RecordingStatus.SUCCESS.value,
            norad_id=40069,
        )
        defaults.update(overrides)
        return RecordingMetadata(**defaults)

    def test_contains_required_fields(self):
        meta = self._sample()
        data = meta.to_dict()
        required_fields = {
            "satellite_name", "norad_id", "frequency_hz", "sample_rate", "gain",
            "scheduled_aos", "scheduled_los", "actual_recording_start", "actual_recording_stop",
            "pre_buffer_seconds", "post_buffer_seconds", "recording_duration_seconds",
            "output_filename", "output_file_size", "recording_status", "error_message",
            "creation_timestamp",
        }
        assert required_fields.issubset(data.keys())

    def test_to_json_round_trips_via_write_and_read(self, tmp_path):
        meta = self._sample(output_file_size=48_000_000, recording_duration_seconds=10.0)
        path = tmp_path / "meta.json"
        meta.write(path)

        loaded = RecordingMetadata.read(path)

        assert loaded.satellite_name == "METEOR-M2-4"
        assert loaded.norad_id == 40069
        assert loaded.output_file_size == 48_000_000
        assert json.loads(path.read_text())["frequency_hz"] == 137_900_000

    def test_simulated_flag_defaults_false(self):
        assert self._sample().simulated is False

    def test_simulated_flag_can_be_set(self):
        assert self._sample(simulated=True).simulated is True

    def test_error_message_present_on_failure(self):
        meta = self._sample(recording_status=RecordingStatus.FAILED.value, error_message="disk full")
        assert meta.error_message == "disk full"


class TestRecordingResult:
    def test_success_property_true_for_success_status(self):
        result = RecordingResult(status=RecordingStatus.SUCCESS)
        assert result.success is True

    def test_success_property_false_for_other_statuses(self):
        for status in (RecordingStatus.FAILED, RecordingStatus.CANCELLED, RecordingStatus.DEVICE_BUSY):
            assert RecordingResult(status=status).success is False
