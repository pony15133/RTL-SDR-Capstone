#!/usr/bin/env python3
"""A fake ``rtl_sdr`` used by the unit tests, so process-control logic can be
exercised without real RTL-SDR hardware.

Behaviour is selected via the ``FAKE_RTLSDR_MODE`` environment variable:

- ``normal`` (default): writes IQ-shaped bytes to the output file until it
  receives SIGINT/SIGTERM (mirroring real rtl_sdr's clean-shutdown behaviour
  on POSIX), then exits 0.
- ``busy``: exits immediately with a "usb_claim_interface error" on stderr,
  mimicking the device already being claimed by another process.
- ``not_found``: exits immediately with "No supported devices found".
- ``immediate_fail``: exits immediately with an unrecognised error.
- ``crash``: writes a little data, then exits non-zero shortly after,
  simulating a mid-recording crash.

Only exercised on POSIX in this test suite (SIGINT/SIGTERM); the
Windows-specific CTRL_BREAK_EVENT stop path in rtl_recorder.process is not
covered here and should be verified manually on Windows.
"""
import os
import signal
import sys
import time


def main() -> int:
    args = sys.argv[1:]
    if not args:
        return 2
    output_path = args[-1]
    mode = os.environ.get("FAKE_RTLSDR_MODE", "normal")

    sample_rate = 2_400_000
    for i, arg in enumerate(args):
        if arg == "-s" and i + 1 < len(args):
            try:
                sample_rate = int(args[i + 1])
            except ValueError:
                pass

    if mode == "busy":
        sys.stderr.write("usb_claim_interface error -6\n")
        sys.stderr.flush()
        return 1
    if mode == "not_found":
        sys.stderr.write("No supported devices found.\n")
        sys.stderr.flush()
        return 1
    if mode == "immediate_fail":
        sys.stderr.write("rtl_sdr: some other unrecognised failure\n")
        sys.stderr.flush()
        return 1

    stop_flag = {"set": False}

    def _handle_stop(signum, frame):
        stop_flag["set"] = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    bytes_per_second = sample_rate * 2
    chunk = max(1, bytes_per_second // 10)

    with open(output_path, "wb") as f:
        if mode == "crash":
            f.write(b"\x00" * chunk)
            f.flush()
            time.sleep(0.6)
            sys.stderr.write("rtl_sdr: unexpected crash\n")
            sys.stderr.flush()
            return 2

        while not stop_flag["set"]:
            f.write(b"\x00" * chunk)
            f.flush()
            time.sleep(0.1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
