#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "iq-recorder"))

from rtl_recorder.config import RecorderConfig
from rtl_recorder.recorder import RTLSDRRecorder
from rtl_recorder.scheduler import wait_until


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scheduled_capture.py <config.json>")
        return 2

    config_path = Path(sys.argv[1]).resolve()

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 2

    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    required = [
        "name",
        "frequency_hz",
        "sample_rate",
        "gain",
        "duration_seconds",
        "start_at",
    ]

    missing = [key for key in required if key not in cfg]
    if missing:
        print(f"Missing config fields: {', '.join(missing)}")
        return 2

    name = str(cfg["name"])
    frequency_hz = int(cfg["frequency_hz"])
    sample_rate = int(cfg["sample_rate"])
    gain = cfg["gain"]
    duration = float(cfg["duration_seconds"])
    output_dir = Path(cfg.get("output_dir", "recordings/scheduled_test"))

    # Example:
    # 2026-09-25T17:30:00+08:00
    start_at = datetime.fromisoformat(cfg["start_at"])

    if start_at.tzinfo is None:
        print(
            "ERROR: start_at must include a timezone offset, "
            'e.g. "2026-09-25T17:30:00+08:00"'
        )
        return 2

    now = datetime.now(timezone.utc)
    start_utc = start_at.astimezone(timezone.utc)

    if start_utc <= now:
        print(f"ERROR: scheduled time has already passed: {start_at.isoformat()}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    recorder = RTLSDRRecorder(
        RecorderConfig(
            output_dir=str(output_dir),
            simulate=False,
        )
    )

    print("Scheduled hardware capture")
    print("==========================")
    print(f"Name:        {name}")
    print(f"Frequency:   {frequency_hz / 1e6:.3f} MHz")
    print(f"Sample rate: {sample_rate}")
    print(f"Gain:        {gain}")
    print(f"Duration:    {duration:.1f} s")
    print(f"Start local: {start_at.isoformat()}")
    print(f"Start UTC:   {start_utc.isoformat()}")
    print(f"Output:      {output_dir.resolve()}")
    print()
    print("Waiting. Ctrl+C cancels before recording starts.")

    try:
        reached = wait_until(
            start_utc,
            now_fn=lambda: datetime.now(timezone.utc),
        )
    except KeyboardInterrupt:
            print("\nCancelled before scheduled start.")
            return 130

    if not reached:
        print("Scheduled wait was cancelled.")
        return 130

    print("\nStarting capture...")

    result = recorder.record(
        satellite_name=name,
        frequency_hz=frequency_hz,
        sample_rate=sample_rate,
        gain=gain,
        duration=duration,
    )

    print()
    print("Capture result")
    print("==============")
    print(f"Status:   {result.status.value}")
    print(f"IQ file:  {result.output_file}")
    print(f"Metadata: {result.metadata_file}")
    print(f"Bytes:    {result.output_file_size}")

    if result.error_message:
        print(f"Error:    {result.error_message}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())