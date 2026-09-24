# RTL-SDR Automatic IQ Recording System

The **RTL-SDR Recording Manager** for the RTL-SDR-Capstone project: a
modular, testable Python component that controls an RTL-SDR Blog V3 via the
official `rtl_sdr` command-line tool and records raw IQ samples to disk for
a given frequency/period, with JSON metadata alongside every recording.

It is one stage of the project's automatic satellite-capture pipeline:

```
Pass prediction (rtl_recorder/passes.py) → auto_capture.py → RTL-SDR Recording Manager → Raw IQ File
    → Signal Processing / Feature Extraction → rule + ML detection → database → keep/archive/delete
```

This package holds the **Recording Manager** plus the pass-prediction
(`passes.py`) and environment-check (`doctor.py`) helpers. The recorder
itself does not do signal processing; the rest of the chain lives in
`../sdr-doppler-prototype/` and is driven by `../auto_capture.py` /
`../pipeline.py` through the recorder's public interface.

## 1. Purpose

Given a satellite name, centre frequency, sample rate, gain, and a
duration (or later, an AOS/LOS pass window), the recorder:

1. Confirms an RTL-SDR is available.
2. Launches `rtl_sdr` with the requested parameters.
3. Monitors the process and the output file.
4. Stops the process cleanly (on a timer, on request, or on cancellation).
5. Verifies the resulting IQ file is real and plausible in size.
6. Writes a JSON metadata sidecar.
7. Returns a `RecordingResult` - `SUCCESS`, `FAILED`, `CANCELLED`, or
   `DEVICE_BUSY` - and never reports success for a failed capture.

## 2. Architecture

```
rtl_recorder/
    __init__.py     Public exports (RTLSDRRecorder, RecorderConfig, ...)
    recorder.py      RTLSDRRecorder: state machine, record()/record_pass(),
                       start/stop/cancel/check_status, finalisation & metadata
    config.py         RecorderConfig + parameter validation
    metadata.py       RecordingMetadata (JSON sidecar) + RecordingResult
    process.py        Cross-platform subprocess start/stop (SIGINT vs CTRL_BREAK)
    device.py          rtl_test-based device availability/busy check
    filenames.py       Name sanitisation + collision-safe unique filenames
    states.py          RecorderState / RecordingStatus enums
    exceptions.py       Exception hierarchy used internally
    utils.py             Executable discovery, expected file size, disk space
    scheduler.py           Wall-clock wait-until-a-time-of-day (--start-time/--start-at)
    passes.py              TLE download/cache + pass prediction (AOS/LOS/max elevation/Doppler)
    doctor.py              Environment check: rtl_sdr/rtl_test, device, folders, packages
    main.py                 CLI entry point (manual, simulation, and scheduled modes)
recorder.py         Thin root-level wrapper: `python recorder.py ...`
tests/
    test_validation.py, test_filenames.py, test_metadata.py, test_recorder.py,
    test_scheduler.py, test_main_scheduling.py, test_passes.py, test_cross_platform.py
    fixtures/fake_rtl_sdr.py   Fake rtl_sdr used to unit-test process control
    hardware/test_hardware_kiss92.py   Real-hardware-only tests (see §12)
```

`RTLSDRRecorder` never touches pass prediction or signal processing.
`auto_capture.py` (repo root) calls it through `record_pass()`, and
`pipeline.py` through `record()` (or the
lower-level `start_recording()`/`stop_recording()`/`cancel_recording()`/
`check_recording_status()` primitives) and only ever gets a
`RecordingResult` back - it never needs to catch exceptions for expected
operational failures.

## 3. Requirements

- Python 3.9+ (standard library only for the recorder itself; `pytest` for
  the test suite - see `requirements.txt`).
- An RTL-SDR Blog V3 dongle.
- The RTL-SDR command-line tools (`rtl_sdr`, `rtl_test`) installed and on
  `PATH`, or their paths passed explicitly via `RecorderConfig`/CLI flags.

## 4. RTL-SDR Driver/Software Requirements

- **Windows**: install RTL-SDR drivers via [Zadig](https://zadig.akeo.ie/)
  (WinUSB driver for the RTL2832U device) and the `rtl_sdr`/`rtl_test`
  binaries from a build such as
  [rtlsdrblog/rtl-sdr-blog](https://github.com/rtlsdrblog/rtl-sdr-blog) or
  the osmocom Windows release.
- **Linux**: install `rtl-sdr` (e.g. `sudo apt install rtl-sdr`) and add a
  udev rule / blocklist the in-kernel `dvb_usb_rtl28xxu` driver so it
  doesn't claim the device (standard RTL-SDR setup step, documented in the
  rtl-sdr project's README).

## 5. Installing the RTL-SDR Command-Line Tools

- **Windows**: download a prebuilt `rtl-sdr` release zip, extract it
  somewhere (e.g. `C:\rtl-sdr\`), and either add that folder to `PATH` or
  pass `--rtl-sdr-path C:\rtl-sdr\rtl_sdr.exe` / `--rtl-test-path ...` on
  the CLI.
- **Linux**: `sudo apt install rtl-sdr` (Debian/Ubuntu) or build from
  source per the [osmocom rtl-sdr](https://osmocom.org/projects/rtl-sdr/wiki)
  instructions.

## 6. Verifying the RTL-SDR Is Detected

```bash
rtl_test -t
```

This should report the tuner type (e.g. Rafael Micro R820T2) and a list of
supported gain values, then exit. The recorder's `check_device()` /
`--skip-device-check`-free CLI runs the equivalent check automatically
before a real (non-simulated) recording.

## 7. Setup

```bash
cd iq-recorder
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 8. Simulation Mode (No Hardware Required)

```bash
python recorder.py --simulate --duration 10
```

This exercises the full state machine, filename generation, and metadata
generation without touching any hardware - the recorded `.iq` file is a
small placeholder (capped, not the full theoretical size, to keep
simulated captures fast) and its metadata is marked `"simulated": true`.
Use this for development, CI, and the automated test suite.

## 9. Manual Recording

```bash
python recorder.py \
    --frequency 137900000 \
    --sample-rate 2400000 \
    --gain 30 \
    --duration 60 \
    --satellite "METEOR-M2-4"
```

This checks the device, launches `rtl_sdr`, records for the given
duration, stops automatically, verifies the output, writes metadata, and
prints a clear `Status: SUCCESS|FAILED|CANCELLED|DEVICE_BUSY` line plus
the output/metadata file paths.

Useful flags: `--gain auto` (or omit `--gain`) for AGC, `--output-dir`,
`--rtl-sdr-path`/`--rtl-test-path` (explicit executable paths),
`--device-index` (multi-dongle), `--skip-device-check`, `--log-level`,
`--log-file`.

### Auto-capture at a fixed clock time (e.g. an FM station at 7pm)

`--start-time HH:MM` (or `--start-at` for an exact date+time) waits until
that moment, then runs a normal manual recording - no satellite pass
prediction involved, just "record this frequency at this wall-clock
time." This is the `rtl_recorder.scheduler` module (`next_occurrence()` /
`wait_until()`), separate from and much simpler than the AOS/LOS pass
scheduling in §10 below.

```bash
python recorder.py \
    --frequency 92000000 \
    --sample-rate 2400000 \
    --gain 30 \
    --duration 1800 \
    --satellite "KISS92-7PM" \
    --start-time 19:00
```

Run this any time before 7pm and it waits (printing how long, updating
you via the log) until the clock reaches `19:00` local time - today, or
tomorrow if `19:00` has already passed today - then records for
`--duration` seconds exactly as a normal manual recording would. Ctrl+C
during the wait cancels cleanly (`CANCELLED`, exit code 130) without
starting `rtl_sdr` at all. `--start-at 2026-08-24T19:00:00` schedules an
exact date+time instead of "the next occurrence of a time of day."

This only works while the process keeps running in the foreground (or
under something like `tmux`/`screen`/a background service) for however
long the wait is - it isn't a persistent scheduler. For a capture that
must survive a closed terminal or a machine restart, use your OS's own
scheduler to launch the (non-scheduled) manual-recording command at the
right time instead:

- **Linux/macOS (`cron`)**: `0 19 * * * cd /path/to/iq-recorder && python3 recorder.py --frequency 92000000 --sample-rate 2400000 --gain 30 --duration 1800 --satellite KISS92-7PM`
- **Windows (Task Scheduler)**: create a Basic Task, trigger "Daily" at 7:00 PM, action "Start a program" running `python.exe` with arguments `recorder.py --frequency 92000000 --sample-rate 2400000 --gain 30 --duration 1800 --satellite KISS92-7PM` and "Start in" set to the `iq-recorder` folder.

Both approaches call the exact same CLI, so `--start-time`/`--start-at`
and OS-level scheduling are interchangeable - use whichever fits how
you're running this.

## 10. Scheduled Recording from Satellite Passes

`record_pass()` waits for a pass and records from AOS - `pre_buffer` to
LOS + `post_buffer`. It joins a pass already in progress, refuses one
that's already over, and can be cancelled while waiting
(`cancel_recording()`). Pass times come from `rtl_recorder.passes`:

```python
from rtl_recorder import RTLSDRRecorder, RecorderConfig
from rtl_recorder.passes import GroundStation, get_tle, find_passes

station = GroundStation(lat_deg=1.3521, lon_deg=103.8198, alt_m=15)
tle = get_tle(25544)                       # CelesTrak, cached in tle_cache/ for 12 h
next_pass = find_passes(tle, station, hours=24, min_max_elevation_deg=15)[0]

recorder = RTLSDRRecorder(RecorderConfig())
result = recorder.record_pass(
    satellite_name="ISS", norad_id=25544,
    frequency_hz=145_800_000, sample_rate=1_024_000, gain=30,
    aos=next_pass.aos, los=next_pass.los, pre_buffer=30, post_buffer=30,
)
```

`pip install sgp4` for the standard SGP4 propagator. Without it, a
built-in Kepler + J2 model is used, which is within about 7.5 km of the official
SGP4 verification case, or roughly 1 s of pass timing. For the full loop
(many satellites, detection, database, retention) use `auto_capture.py` at
the repo root.

**Cross-platform:** the RTL-SDR tools are found on PATH, in the usual
install folders for Windows / macOS / Linux, or in `$RTL_SDR_HOME`. Run
`python -m rtl_recorder.doctor` to check a machine.

## 11. Python Interface

```python
from rtl_recorder import RTLSDRRecorder, RecorderConfig

recorder = RTLSDRRecorder(RecorderConfig(output_dir="recordings"))

result = recorder.record(
    satellite_name="METEOR-M2-4",
    norad_id=40069,
    frequency_hz=137_900_000,
    sample_rate=2_400_000,
    gain=30,
    duration=60,
)

result.status              # RecordingStatus.SUCCESS / FAILED / CANCELLED / DEVICE_BUSY
result.output_file
result.metadata_file
result.error_message

# Lower-level primitives, for advanced/async use:
recorder.start_recording(...)
recorder.stop_recording()
recorder.cancel_recording()
recorder.check_recording_status()   # RecorderState
```

## 12. Output IQ Format

`rtl_sdr`'s default output is **interleaved 8-bit unsigned I/Q samples**
(1 byte I, 1 byte Q per sample, no file header). Expected size in bytes is
`sample_rate * 2 * duration_seconds`; the recorder computes this and flags
(fails) a recording that comes in well under that, as a sign the capture
died partway through.

Filenames: `SATELLITE_YYYYMMDD_HHMMSS_FREQUENCYHz.iq`, e.g.
`METEOR-M2-4_20260809_193430_137900000Hz.iq`. Satellite names are
sanitised for filesystem safety, and an existing recording is **never**
overwritten - a numeric suffix (`_2`, `_3`, ...) is appended on collision.

## 13. Metadata Format

Every recording attempt (success, failure, or cancellation) writes a JSON
sidecar next to the `.iq` file, e.g.
`METEOR-M2-4_20260809_193430_137900000Hz.json`:

```json
{
  "satellite_name": "METEOR-M2-4",
  "norad_id": 40069,
  "frequency_hz": 137900000,
  "sample_rate": 2400000,
  "gain": 30.0,
  "output_filename": "METEOR-M2-4_20260809_193430_137900000Hz.iq",
  "recording_status": "SUCCESS",
  "scheduled_aos": null,
  "scheduled_los": null,
  "actual_recording_start": "2026-08-09T19:34:30+00:00",
  "actual_recording_stop": "2026-08-09T19:35:30+00:00",
  "pre_buffer_seconds": 0.0,
  "post_buffer_seconds": 0.0,
  "recording_duration_seconds": 60.02,
  "output_file_size": 288048000,
  "expected_file_size": 288096000,
  "error_message": null,
  "simulated": false,
  "device_index": 0,
  "command": ["rtl_sdr", "-d", "0", "-f", "137900000", "-s", "2400000", "-g", "30.0", "..."],
  "creation_timestamp": "2026-08-09T19:35:30.123456+00:00"
}
```

## 14. Known Limitations

- Only one RTL-SDR device is managed per `RTLSDRRecorder` instance; a
  second concurrent `record()` call reports `DEVICE_BUSY` rather than
  queuing or interrupting the active recording (by design - see spec §16).
- Frequency validation assumes the R820T2's normal tuning range
  (24 MHz-1766 MHz); RTL-SDR "direct sampling" mode for HF reception below
  that is out of scope.
- Gain is validated as a 0-50 dB outer range only; `rtl_sdr` itself snaps
  the requested value to the nearest value the tuner actually supports.
- The Windows graceful-stop path (CTRL_BREAK_EVENT) is implemented but
  only exercised by manual testing on Windows - the automated test suite's
  fake-`rtl_sdr` fixture only covers the POSIX SIGINT path.
- No frequency tracking or retuning during a recording - the centre
  frequency is fixed for the whole capture. Doppler is removed afterwards,
  in processing (`sdr-doppler-prototype/src/doppler.py`), from the TLE
  prediction.

## 15. How the rest of the project uses this package

Pass scheduling, Doppler correction, detection, ML scoring, keep/discard
and database storage are built on top of this package without changing
the recorder: see `../auto_capture.py` (predict -> `record_pass()` ->
process -> store), `../pipeline.py` (`record()` -> process -> store) and
the root README. Still not provided: a system service (systemd / Windows
service) for running unattended across reboots - `run_station` runs in a
terminal window.

## 16. Testing

```bash
pytest                       # unit tests only (no hardware required)
pytest -m hardware -v tests/hardware/test_hardware_kiss92.py   # real hardware
```

Hardware-dependent tests live under `tests/hardware/` and are excluded by
default (`pytest.ini`: `addopts = -m "not hardware"`) so CI and day-to-day
development never need real RTL-SDR hardware attached. The rest of the
suite covers configuration validation, filename generation/collision
handling, metadata generation, state transitions, simulated recording,
and - via a fake `rtl_sdr` executable (`tests/fixtures/fake_rtl_sdr.py`) -
real process launch/stop/crash/busy/not-found handling without needing an
actual dongle.

## 17. Phase 1 Hardware Test - KISS92 Singapore (92.0 MHz)

Before pointing this at a satellite pass, validate the full pipeline
against a strong local FM broadcast station. This is purely a convenient
real-world RF test signal - the recorder itself is modulation-agnostic and
records raw IQ regardless of what's on the frequency; the eventual target
is the ~137 MHz weather-satellite band.

```bash
# 1. Confirm the device is detected
rtl_test -t

# 2-5. Automated capture, auto-stop, size sanity check, and a repeat
#      recording to confirm the device was released
pytest -m hardware -v tests/hardware/test_hardware_kiss92.py

# Or manually, equivalent to steps 2-4:
python recorder.py --frequency 92000000 --sample-rate 2400000 --gain 30 \
    --duration 30 --satellite "KISS92-SINGAPORE"
```

Expected outcome: `rtl_test` reports a detected device; the recorder tunes
to 92.0 MHz, records for ~30 seconds, stops automatically, and reports
`Status: SUCCESS` with an output file size close to
`2,400,000 samples/sec × 2 bytes/sample × 30 sec ≈ 144,000,000 bytes`; a
second recording started a few seconds later also succeeds (proving the
device was released), with a distinct filename.

If any stage fails, the CLI reports which one (device not detected,
launch failure, device busy, undersized file, etc.) along with the
underlying `rtl_sdr`/`rtl_test` diagnostic output captured in the logs.

### Verifying a capture manually (Test 6, optional)

The recorder only produces raw IQ - it is not, and should not become, an
audio recorder. To manually confirm a `.iq` file actually contains the
KISS92 signal (or any RF signal), open it in a general-purpose SDR/IQ
viewer, for example:

- **inspectrum** (Linux/Mac) - open the raw file directly as
  8-bit unsigned interleaved I/Q for a spectrogram view.
- **GQRX** / **SDR++** "IQ file" input mode, set to the same sample rate
  and centre frequency used for the capture, to listen back live.
- **GNU Radio** `File Source` (`uchar`) → `Waterfall Sink`/`Audio Sink`
  for a quick custom flowgraph.

This is a verification aid only, external to the recorder itself.
