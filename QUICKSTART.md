# Quick start: running the station with an RTL-SDR

For anyone on the team with a dongle and antenna. It takes about 15 minutes the first time.
Use `.bat` on Windows and `.sh` on macOS/Linux; everything else is identical.

## 0. What you need

- Python 3.10 or newer ([python.org](https://www.python.org/downloads/); on Windows tick **"Add python.exe to PATH"**)
- The RTL-SDR tools:
  - **Windows:** download the release zip from <https://ftp.osmocom.org/binaries/windows/rtl-sdr/>, unzip it (e.g. to `C:\rtl-sdr`), and install the **WinUSB** driver for the dongle with [Zadig](https://zadig.akeo.ie/) (Options → List all devices → "Bulk-In, Interface (Interface 0)" → WinUSB → Replace Driver)
    - **RTL-SDR Blog V4** (the silver dongle labelled V4, R828D tuner): use the RTL-SDR Blog driver release (<https://github.com/rtlsdrblog/rtl-sdr-blog/releases>) instead of the osmocom zip. The V4 needs it to tune correctly. `rtl_test` prints the tuner name (R820T = V3, R828D = V4)
  - **macOS:** `brew install librtlsdr`
  - **Linux:** `sudo apt install rtl-sdr`
- An antenna. A V-dipole (~53 cm per arm, 120° apart, horizontal) is ideal for the METEOR-M weather satellites at 137.9 MHz.
- About 8 GB of free RAM while a pass is being processed (it peaks around 6.7 GB). On an 8 GB computer, set
  `"max_detection_seconds": 60` in the `recording` section of `capture_config.json` (step 3) to use about half as much.

## 1. Set up (once)

```
setup.bat                        (Windows: double-click, or run in PowerShell)
./setup.sh                       (macOS / Linux)
```

This creates `.venv`, installs the packages, creates your own `capture_config.json`, trains the IQ model on the real RSP-03 data, and runs an environment check. If the check says `rtl_sdr` wasn't found, follow the instruction it prints. Or set `RTL_SDR_HOME` to the folder you unzipped (e.g. `setx RTL_SDR_HOME C:\rtl-sdr` on Windows, then open a new terminal).

Optional, with internet (10–20 minutes): download real SatNOGS training data and train the waterfall model:

```
setup.bat --fetch-satnogs
./setup.sh --fetch-satnogs
```

It finishes by testing the new model on a real pass it has never seen (the CAMRAS RSP-03 data). For a bigger, better-labelled dataset (a few hours, resumable), and for adding your own passes, see [TRAINING.md](TRAINING.md).

## 2. Check the dongle (1 minute)

```
check_dongle.bat
./check_dongle.sh
```

This records 5 seconds of Kiss92 (92.0 MHz) and saves a waterfall picture. A bright band about 200 kHz wide in the middle means the dongle, driver and antenna work. For another station, add `--freq 95.0e6`.

## 3. Edit your station (once)

Open `capture_config.json`:

- `station`: your latitude/longitude (Google Maps → right-click → copy coordinates) and height in metres
- `satellites`: the METEOR-M2-3 and METEOR-M2-4 weather satellites (137.9 MHz, always transmitting) are the best first targets
- `min_elevation_deg`: raise to 25–30 if buildings block the horizon
- all keys are listed in the README's Configuration table

## 4. Run the station

```
run_station.bat
./run_station.sh
```

The station's web page opens at <http://localhost:8050>. Leave it running; the computer is kept awake and the capture loop restarts itself if it ever crashes. Each pass is recorded, Doppler-corrected, checked by the rule detector and both ML models, and logged in the database. The result is one of three:

- **Satellite found**: the raw IQ is kept.
- **Needs review** (the model wasn't sure): the raw IQ is kept in `recordings/uncertain/` until someone looks at it on the **Review** page and presses *Signal* or *Noise*.
- **Nothing found**: the raw IQ is archived to `recordings/rejected/` or deleted, as chosen in **Settings**. The waterfall picture is always kept.

Web pages: **Overview** (what's happening now, next pass, results), **Passes**, **Captures** (click one for everything about it), **Review**, **Images** (Earth pictures decoded from the METEOR-M weather satellites), **Settings** (station, satellites, what to do with recordings), **Health** (checks, events, log).

Optional, for Earth pictures from the METEOR-M weather satellites: install [SatDump](https://www.satdump.org) (free). Their kept passes are then decoded automatically.

To try it without hardware: `run_station.bat --simulate`. To see the whole chain in 20 seconds: `.venv\Scripts\python auto_capture.py --config capture_config.json --demo` (Windows) or `.venv/bin/python auto_capture.py --config capture_config.json --demo`.

## Useful commands

| What | Command (prefix with `.venv\Scripts\python` or `.venv/bin/python`) |
|---|---|
| List upcoming passes | `auto_capture.py --config capture_config.json --list-only` |
| Look at any recording | `sdr-doppler-prototype/src/visualize.py --input recordings/<file>.iq` |
| …with Doppler correction | add `--doppler --lat 1.3521 --lon 103.8198` |
| What's in the database | `sdr-doppler-prototype/src/history.py` |
| Desktop GUI (maximise the window to see the Results pane) | `sdr-doppler-prototype\launch_gui.bat` / `sdr-doppler-prototype/launch_gui.sh` (run directly, no python prefix) |
| Full real-pass test with evidence | follow [LIVE_HARDWARE_TEST.md](LIVE_HARDWARE_TEST.md) |
| Environment check | `pipeline.py --doctor` |

## If something goes wrong

- **`usb_claim_interface error` or DEVICE_BUSY:** another program (SDR#, SDR++) has the dongle open. Close it, or unplug and replug the dongle.
- **"rtl_sdr not found":** see step 1 (install the tools, or set `RTL_SDR_HOME`).
- **Dashboard says "stalled?":** `auto_capture` stopped. Check its window for the error.
- **Only noise in every pass:** check the antenna connection, raise `gain` (up to 49), raise `min_elevation_deg`, and try the METEOR-M weather satellites first.
