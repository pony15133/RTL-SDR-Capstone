"""Command-line interface for the RTL-SDR recorder - manual and simulation modes.

Examples
--------
Manual recording against real hardware::

    python -m rtl_recorder.main --frequency 137900000 --sample-rate 2400000 \\
        --gain 30 --duration 60 --satellite "METEOR-M2-4"

Simulation mode (no hardware required)::

    python -m rtl_recorder.main --simulate --duration 10

Scheduled recording - e.g. auto-capture an FM station at 7pm local time
(waits, in the foreground, until the clock reaches that time, then runs
exactly like a manual recording)::

    python -m rtl_recorder.main --frequency 92000000 --sample-rate 2400000 \\
        --gain 30 --duration 1800 --satellite "KISS92-7PM" --start-time 19:00

(The repo also ships a thin ``recorder.py`` wrapper at the project root so
``python recorder.py --simulate --duration 10`` works too.)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from datetime import time as dt_time

from .config import RecorderConfig
from .recorder import RTLSDRRecorder
from .scheduler import next_occurrence, wait_until
from .states import RecordingStatus


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtl_recorder",
        description="RTL-SDR automatic IQ recorder - manual/simulation mode (Phase 1).",
    )
    parser.add_argument("--satellite", default="MANUAL-TEST", help="Satellite/target name (default: MANUAL-TEST)")
    parser.add_argument("--norad-id", type=int, default=None, help="NORAD catalog ID (optional)")
    parser.add_argument("--frequency", type=int, default=100_000_000,
                         help="Centre frequency in Hz (default: 100000000)")
    parser.add_argument("--sample-rate", type=int, default=2_400_000,
                         help="Sample rate in samples/sec (default: 2400000)")
    parser.add_argument("--gain", type=str, default=None,
                         help="Tuner gain in dB, or omit/'auto' for AGC (default: auto)")
    parser.add_argument("--duration", type=float, default=60, help="Recording duration in seconds (default: 60)")
    parser.add_argument("--output-dir", default="recordings", help="Directory for .iq/.json output (default: recordings)")
    parser.add_argument("--rtl-sdr-path", default=None, help="Explicit path to rtl_sdr(.exe) if not on PATH")
    parser.add_argument("--rtl-test-path", default=None, help="Explicit path to rtl_test(.exe) if not on PATH")
    parser.add_argument("--device-index", type=int, default=0, help="RTL-SDR device index (default: 0)")
    parser.add_argument("--simulate", action="store_true", help="Simulation mode - no real RTL-SDR hardware required")
    parser.add_argument("--skip-device-check", action="store_true",
                         help="Skip the rtl_test device-availability check before recording")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=None, help="Optional path to also write logs to a file")

    schedule_group = parser.add_mutually_exclusive_group()
    schedule_group.add_argument(
        "--start-time", type=_parse_hhmm, default=None, metavar="HH:MM",
        help="Wait until this local clock time (today, or tomorrow if it's already passed today), then record",
    )
    schedule_group.add_argument(
        "--start-at", type=_parse_iso_datetime, default=None, metavar="ISO8601",
        help="Wait until this exact local date+time (e.g. 2026-08-24T19:00:00), then record",
    )
    return parser


def _parse_hhmm(value: str) -> dt_time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--start-time must be HH:MM (24-hour), got {value!r}") from exc


def _parse_iso_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--start-at must be an ISO-8601 datetime, got {value!r}") from exc


def configure_logging(level: str, log_file: str = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=handlers,
    )


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_file)
    logger = logging.getLogger("rtl_recorder.main")

    config = RecorderConfig(
        output_dir=args.output_dir,
        rtl_sdr_path=args.rtl_sdr_path,
        rtl_test_path=args.rtl_test_path,
        simulate=args.simulate,
        device_index=args.device_index,
    )
    recorder = RTLSDRRecorder(config)

    if not args.simulate and not args.skip_device_check:
        check = recorder.check_device()
        if not check.available:
            logger.error("RTL-SDR not available, aborting: %s", check.message)
            print(f"FAILED: RTL-SDR not available - {check.message}")
            return 1

    target_dt = None
    if args.start_time is not None:
        target_dt = next_occurrence(args.start_time)
    elif args.start_at is not None:
        target_dt = args.start_at

    if target_dt is not None:
        wait_seconds = (target_dt - datetime.now()).total_seconds()
        if wait_seconds <= 0:
            logger.info("Scheduled start time %s has already arrived - recording immediately", target_dt.isoformat())
        else:
            logger.info("Recording scheduled for %s (waiting %.0f seconds)", target_dt.isoformat(), wait_seconds)
            print(f"Waiting until {target_dt.isoformat()} ({wait_seconds:.0f}s from now)... press Ctrl+C to cancel.")
        try:
            wait_until(target_dt)
        except KeyboardInterrupt:
            logger.warning("Interrupted while waiting for scheduled start time")
            print("CANCELLED: interrupted while waiting for scheduled start time")
            return 130
        logger.info("Scheduled start time reached - starting recording")

    try:
        result = recorder.record(
            satellite_name=args.satellite,
            norad_id=args.norad_id,
            frequency_hz=args.frequency,
            sample_rate=args.sample_rate,
            gain=args.gain,
            duration=args.duration,
            output_dir=args.output_dir,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted before recording could start")
        print("CANCELLED: interrupted before recording could start")
        return 130

    print(f"Status: {result.status.value}")
    if result.output_file:
        print(f"Output file: {result.output_file}")
    if result.output_file_size is not None:
        print(f"Output file size: {result.output_file_size} bytes")
    if result.metadata_file:
        print(f"Metadata file: {result.metadata_file}")
    if result.error_message:
        print(f"Error: {result.error_message}")

    return 0 if result.status == RecordingStatus.SUCCESS else 1


if __name__ == "__main__":
    sys.exit(main())
