#!/usr/bin/env python3
"""One-minute RTL-SDR hardware test - run by check_dongle.bat / check_dongle.sh.

Checks the tools and dongle, records a few seconds of a strong local FM
broadcast station, and draws its waterfall. If you can see a bright
~200 kHz-wide band in the middle of the picture, the dongle, driver,
antenna connection and the whole recording path work.

    python check_dongle.py                    # 92.0 MHz (Kiss92 Singapore), 5 s
    python check_dongle.py --freq 95.0e6      # another station
    python check_dongle.py --simulate         # no hardware: just checks the software path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "iq-recorder"))
sys.path.insert(0, str(REPO / "sdr-doppler-prototype" / "src"))

from rtl_recorder.config import RecorderConfig  # noqa: E402
from rtl_recorder.doctor import format_report, run_checks  # noqa: E402
from rtl_recorder.recorder import RTLSDRRecorder  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Quick RTL-SDR hardware test with an FM broadcast station.")
    p.add_argument("--freq", type=float, default=92.0e6, help="FM station in Hz (default 92.0 MHz)")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--sample-rate", type=int, default=1_024_000)
    p.add_argument("--gain", default="30")
    p.add_argument("--output-dir", default=str(REPO / "recordings" / "dongle_check"))
    p.add_argument("--simulate", action="store_true")
    args = p.parse_args(argv)

    print("Step 1/3 - environment")
    checks = run_checks(output_dir=args.output_dir, probe_device=not args.simulate)
    print(format_report(checks))
    tools_ok = all(c.ok for c in checks if c.name in ("rtl_sdr", "rtl_test"))
    if not tools_ok and not args.simulate:
        print("\nFix the items above (or use --simulate) and run this again.")
        return 1

    print(f"\nStep 2/3 - recording {args.seconds:g} s at {args.freq / 1e6:.3f} MHz")
    rec = RTLSDRRecorder(RecorderConfig(output_dir=args.output_dir, simulate=args.simulate))
    result = rec.record(satellite_name="DONGLE-CHECK", frequency_hz=int(args.freq), sample_rate=args.sample_rate,
                        gain=args.gain, duration=args.seconds)
    print(f"  status: {result.status.value}  file: {result.output_file}  bytes: {result.output_file_size}")
    if not result.success:
        print(f"  error: {result.error_message}")
        print("  Common fixes: unplug/replug the dongle; close SDR#/other SDR apps; on Windows reinstall the "
              "WinUSB driver with Zadig; on Linux blacklist dvb_usb_rtl28xxu.")
        return 1

    print("\nStep 3/3 - waterfall")
    from visualize import build_parser, run as draw

    out = Path(args.output_dir) / "dongle_check_waterfall.png"
    draw(build_parser().parse_args(["--input", result.output_file, "--output", str(out), "--rows", "200"]))
    print(f"\nOpen {out}")
    if args.simulate:
        print("Simulation: the picture is random noise - that's expected.")
    else:
        print("You should see a bright band about 200 kHz wide in the middle (the FM station).\n"
              "If it's all flat noise: check the antenna cable, raise --gain, or try another --freq.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
