"""Tests for the --start-time/--start-at scheduling flags on the CLI
(rtl_recorder.main) - all in simulation mode, no real hardware or real
waiting required.
"""

import argparse
from datetime import datetime, timedelta
from datetime import time as dt_time

import pytest

from rtl_recorder.main import _parse_hhmm, _parse_iso_datetime, build_arg_parser, main


class TestArgParsing:
    def test_start_time_parses_hhmm(self):
        args = build_arg_parser().parse_args(["--start-time", "19:00"])
        assert args.start_time == dt_time(19, 0)

    def test_start_at_parses_iso_datetime(self):
        args = build_arg_parser().parse_args(["--start-at", "2026-08-24T19:00:00"])
        assert args.start_at == datetime(2026, 8, 24, 19, 0, 0)

    def test_neither_flag_defaults_to_none(self):
        args = build_arg_parser().parse_args([])
        assert args.start_time is None
        assert args.start_at is None

    def test_mutually_exclusive_flags_rejected(self):
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--start-time", "19:00", "--start-at", "2026-08-24T19:00:00"])

    def test_invalid_start_time_format_rejected(self):
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--start-time", "not-a-time"])

    def test_invalid_start_at_format_rejected(self):
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--start-at", "not-a-date"])


class TestParseHelpers:
    def test_parse_hhmm_valid(self):
        assert _parse_hhmm("07:30") == dt_time(7, 30)

    def test_parse_hhmm_invalid_raises_argparse_error(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_hhmm("25:99")

    def test_parse_iso_datetime_valid(self):
        assert _parse_iso_datetime("2026-01-01T00:00:00") == datetime(2026, 1, 1, 0, 0, 0)

    def test_parse_iso_datetime_invalid_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_iso_datetime("nonsense")


class TestScheduledRunEndToEnd:
    def test_start_at_in_the_past_records_immediately(self, tmp_path, capsys):
        past = (datetime.now() - timedelta(seconds=5)).isoformat()
        argv = [
            "--simulate", "--duration", "1", "--start-at", past,
            "--output-dir", str(tmp_path), "--satellite", "PAST-TEST",
        ]

        exit_code = main(argv)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Status: SUCCESS" in captured.out
        assert list(tmp_path.glob("*.iq"))

    def test_start_time_resolved_via_next_occurrence(self, tmp_path, monkeypatch):
        """--start-time should go through scheduler.next_occurrence(), not be used raw."""
        import rtl_recorder.main as main_module

        fixed_past = datetime.now() - timedelta(seconds=1)
        calls = {}

        def fake_next_occurrence(target_time):
            calls["target_time"] = target_time
            return fixed_past

        monkeypatch.setattr(main_module, "next_occurrence", fake_next_occurrence)

        argv = [
            "--simulate", "--duration", "1", "--start-time", "19:00",
            "--output-dir", str(tmp_path), "--satellite", "SCHEDULED-TEST",
        ]

        exit_code = main(argv)

        assert exit_code == 0
        assert calls["target_time"] == dt_time(19, 0)
        assert list(tmp_path.glob("*.iq"))

    def test_keyboard_interrupt_during_wait_returns_130_and_never_records(self, tmp_path, monkeypatch):
        import rtl_recorder.main as main_module

        def fake_wait_until(target_dt, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(main_module, "wait_until", fake_wait_until)

        future = (datetime.now() + timedelta(hours=1)).isoformat()
        argv = [
            "--simulate", "--duration", "1", "--start-at", future,
            "--output-dir", str(tmp_path), "--satellite", "INTERRUPT-TEST",
        ]

        exit_code = main(argv)

        assert exit_code == 130
        assert not list(tmp_path.glob("*.iq"))

    def test_no_schedule_flags_records_immediately_as_before(self, tmp_path, capsys):
        """Backward compatibility: omitting --start-time/--start-at behaves exactly as before."""
        argv = ["--simulate", "--duration", "1", "--output-dir", str(tmp_path), "--satellite", "NO-SCHEDULE-TEST"]

        exit_code = main(argv)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Waiting until" not in captured.out
        assert list(tmp_path.glob("*.iq"))
