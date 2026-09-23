"""Environment check - "will the capture pipeline work on this computer?"

    python -m rtl_recorder.doctor            # from iq-recorder/
    python pipeline.py --doctor              # from the repo root

Reports the OS, where rtl_sdr / rtl_test were found (PATH or the usual
install folders for Windows / macOS / Linux), whether a dongle answers
``rtl_test -t``, whether the output folder is writable, and which optional
Python packages are present. Every missing piece comes with the install
step for *this* OS, so a teammate on a Mac or Linux laptop can get set up
without anyone touching a Windows ``.exe`` path.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .device import check_device
from .exceptions import ExecutableNotFoundError
from .utils import find_executable, install_hint, platform_name


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_checks(*, output_dir: str = "recordings", probe_device: bool = True,
               rtl_sdr_path: Optional[str] = None, rtl_test_path: Optional[str] = None) -> List[Check]:
    checks: List[Check] = [Check("platform", True, f"{platform_name()} {platform.release()} ({platform.machine()}), "
                                                   f"Python {platform.python_version()}")]

    tools = {}
    for tool, explicit in (("rtl_sdr", rtl_sdr_path), ("rtl_test", rtl_test_path)):
        try:
            tools[tool] = find_executable(tool, explicit)
            checks.append(Check(tool, True, tools[tool]))
        except ExecutableNotFoundError:
            where = f"at {explicit}" if explicit else f"on PATH or in the usual {platform_name()} install folders"
            checks.append(Check(tool, False, f"not found {where}"))

    if probe_device and "rtl_test" in tools:
        result = check_device(tools["rtl_test"])
        checks.append(Check("device", result.available, result.message, required=False))
    elif probe_device:
        checks.append(Check("device", False, "skipped - rtl_test not found", required=False))

    out = Path(output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        writable = os.access(out, os.W_OK)
        checks.append(Check("output folder", writable, f"{out.resolve()} {'writable' if writable else 'NOT writable'}"))
    except OSError as exc:
        checks.append(Check("output folder", False, f"{out}: {exc}"))

    for module, why, required in (
        ("numpy", "detection / visualisation", True),
        ("scipy", "spectrograms", True),
        ("sklearn", "ML detector (scikit-learn)", True),
        ("matplotlib", "waterfall images", True),
        ("sgp4", "accurate pass prediction (falls back to a simpler built-in model without it)", False),
        ("tkinter", "desktop GUI", False),
    ):
        present = importlib.util.find_spec(module) is not None
        detail = "installed" if present else f"missing - pip install {'scikit-learn' if module == 'sklearn' else module}"
        if module == "tkinter" and not present:
            detail = "missing - install your OS's python3-tk package (only needed for the GUI)"
        checks.append(Check(f"python: {module}", present, f"{detail} ({why})", required=required))
    return checks


def format_report(checks: List[Check]) -> str:
    lines = ["RTL-SDR capstone environment check", "=" * 34]
    for c in checks:
        mark = "OK " if c.ok else ("!! " if c.required else "-- ")
        lines.append(f"[{mark}] {c.name}: {c.detail}")
    missing_tools = [c for c in checks if c.name in ("rtl_sdr", "rtl_test") and not c.ok]
    if missing_tools:
        lines += ["", "To install the RTL-SDR tools on this computer:", "  " + install_hint()]
    blocking = [c for c in checks if c.required and not c.ok]
    lines += ["", "Ready to record." if not blocking else
              f"{len(blocking)} required item(s) missing - simulation mode (--simulate) still works."]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check this computer is ready for RTL-SDR capture.")
    parser.add_argument("--output-dir", default="recordings")
    parser.add_argument("--no-device", action="store_true", help="Don't run rtl_test -t")
    parser.add_argument("--rtl-sdr-path", default=None)
    parser.add_argument("--rtl-test-path", default=None)
    args = parser.parse_args(argv)
    checks = run_checks(output_dir=args.output_dir, probe_device=not args.no_device,
                        rtl_sdr_path=args.rtl_sdr_path, rtl_test_path=args.rtl_test_path)
    print(format_report(checks))
    return 0 if all(c.ok for c in checks if c.required) else 1


if __name__ == "__main__":
    sys.exit(main())
