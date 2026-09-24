# Manual test plan

A check of every feature, in order. Each test gives the command, what you should see, and a box to tick. Most tests need no hardware. The ones marked 📡 need the RTL-SDR and antenna.

**Conventions**

- Run commands from the repo root in a terminal with the venv active: `.\.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on macOS/Linux.
- On macOS/Linux, write paths with `/` instead of `\`.
- 📡 marks tests that need the dongle.
- Write down anything that doesn't match the "Expect" text, and paste the terminal output to the team or to Claude.

| Tester | Date | OS / Python | Dongle |
|---|---|---|---|
| | | | |

---

## A. Setup and automated tests

| # | Do | Expect | ✓ |
|---|---|---|---|
| A1 | `setup.bat` (or `./setup.sh`) | Ends with "Setup finished". `capture_config.json` exists. `sdr-doppler-prototype\models\random_forest.joblib` is new | ☐ |
| A2 | `python -m pytest` | About 334 passed, 3 deselected, 0 failed (2 skipped without the sgp4 package is fine) | ☐ |
| A3 | `python pipeline.py --doctor` | `[OK ]` for platform, output folder and Python packages. rtl_sdr/rtl_test are OK once the tools are installed | ☐ |
| A4 | Rename `.venv` to `.venv-broken`, run `setup.bat` again | It builds a fresh `.venv` and finishes. Delete `.venv-broken` afterwards | ☐ |

## B. Hardware 📡

| # | Do | Expect | ✓ |
|---|---|---|---|
| B1 | Plug in the dongle. Run `check_dongle.bat` | Step 1: `[OK ] device: RTL-SDR device detected`. Step 2: `status: SUCCESS`, file about 10 MB | ☐ |
| B2 | Open `recordings\dongle_check\dongle_check_waterfall.png` | A bright band about 200 kHz wide in the centre (Kiss92) | ☐ |
| B3 | `check_dongle.bat --freq 95.0e6 --gain 40` | Another station's band in the centre | ☐ |
| B4 | Open SDR# (or any SDR app) using the dongle, then run `check_dongle.bat` | Recording fails with a "busy" message and fix suggestions, no crash. Close SDR# afterwards | ☐ |
| B5 | Unplug the dongle and run `check_dongle.bat` | "No RTL-SDR device found", no crash | ☐ |

## C. Recorder on its own

| # | Do | Expect | ✓ |
|---|---|---|---|
| C1 | `cd iq-recorder` then `python -m rtl_recorder.main --simulate --duration 5` | Status SUCCESS; a `.iq` and a `.json` appear in `recordings\` | ☐ |
| C2 | 📡 `python -m rtl_recorder.main --frequency 92000000 --sample-rate 1024000 --gain 30 --duration 10 --satellite TEST` | SUCCESS; file about 20 MB | ☐ |
| C3 | 📡 Same command with `--duration 60`, press Ctrl+C after 5 s | Stops cleanly; status CANCELLED; the JSON sidecar is still written | ☐ |
| C4 | `python -m rtl_recorder.main --simulate --duration 5 --start-time HH:MM`, set to 2 minutes from now | Waits, then records at that time | ☐ |
| C5 | `python -m rtl_recorder.doctor` | Same report as A3 | ☐ |

## D. Pass prediction

| # | Do | Expect | ✓ |
|---|---|---|---|
| D1 | `python auto_capture.py --config capture_config.json --list-only` | A table of METEOR-M2-3, METEOR-M2-4 and ISS passes for the next 24 h (UTC), with max elevation ≥ 15° | ☐ |
| D2 | Compare two of those passes with an online predictor (heavens-above.com or n2yo.com, same location) | Rise/set times within about 1 minute. For best accuracy run `pip install sgp4` first; D1's lines then say `[sgp4]` | ☐ |
| D3 | Delete the `tle_cache` folder, run D1 again | It re-downloads the TLEs and gives the same schedule | ☐ |
| D4 | Disconnect from the internet, run D1 | Still works from the cache, with a warning | ☐ |
| D5 | `python auto_capture.py --config capture_config.json --list-only --min-elevation 60` | Fewer passes; all ≥ 60° | ☐ |

## E. Full capture chain without hardware

| # | Do | Expect | ✓ |
|---|---|---|---|
| E1 | `python auto_capture.py --config capture_config.json --demo` | About 20 s; ends with `SUCCESS, db row N … IQ archived` | ☐ |
| E2 | `python sdr-doppler-prototype\src\history.py` | The demo capture is at the top of "Recent captures"; the status log has WAITING, SUCCESS and DONE | ☐ |
| E3 | `python sdr-doppler-prototype\src\history.py --capture N` (N from E1) | The full record, including `recording_status`, `frequency_hz` 150 kHz below `target_frequency_hz`, `iq_retention` | ☐ |
| E4 | `python pipeline.py --satellite TEST --frequency 137900000 --sample-rate 1024000 --duration 3 --simulate --retention keep-all` | `iq_retention=kept (policy keep-all)` | ☐ |
| E5 | Same as E4 with `--retention delete-negatives` | `iq_retention=deleted`; the `.iq` file is gone, the `.json` stays (simulated noise scores low, about 0.14; if a model ever scores it ≥ 0.5 it is kept instead - note it) | ☐ |

## F. Visualisation (no ML)

| # | Do | Expect | ✓ |
|---|---|---|---|
| F1 | `python sdr-doppler-prototype\src\visualize.py --input Data\Satellite_Data_snapshots\RSP-03_4.raw --sample-rate 1e6 --center-freq 436.95e6` | PNG in `sdr-doppler-prototype\data\results\`; bursts on a centre line at 436.95 MHz | ☐ |
| F2 | F1 plus `--start-seconds 5 --duration-seconds 5` | Zoomed to 5–10 s | ☐ |
| F3 | 📡 `visualize.py --input recordings\dongle_check\<file>.iq` | Rate and frequency found automatically (`source=sidecar`); FM band visible | ☐ |
| F4 | `visualize.py --input sdr-doppler-prototype\data\raw\lora_process_energy_1.bin --sample-rate 4e6 --center-freq 401.3e6` | LoRa chirps visible (the complex64 format) | ☐ |
| F5 | `visualize.py --input Data\SDRSharp_20200728_093524Z_1544500000Hz_NOAA-15_SARP-3_PDS.wav --duration-seconds 60` | Rate from the WAV header, frequency from the file name | ☐ |
| F6 | `visualize.py --input Data\Satellite_Data_snapshots\RSP-03_4.raw` (no `--sample-rate`) | Clear "sample rate unknown - pass --sample-rate" message, no crash | ☐ |
| F7 | 📡 After a real pass: `visualize.py --input recordings\<pass>.iq --doppler --lat 1.3521 --lon 103.8198` | A second PNG `…_doppler_corrected.png`: the satellite's slope becomes a straight line at 0 kHz | ☐ |

## G. Detection on real data

| # | Do | Expect | ✓ |
|---|---|---|---|
| G1 | `cd sdr-doppler-prototype` then `python src\main.py --input ..\Data\Satellite_Data_snapshots\RSP-03_4.raw --output data\results --sample-rate 1e6 --center-freq 436.95e6 --save-image` | Rule: "NO SATELLITE CANDIDATE" (known weakness); ML: "SATELLITE CANDIDATE" | ☐ |
| G2 | G1 with `RSP-03_1` instead of `RSP-03_4` | Runs. Note both results; this file is mostly noise | ☐ |

## H. Machine learning

| # | Do | Expect | ✓ |
|---|---|---|---|
| H1 | `python train_model.py --dataset data\training\rsp03_camras_features.csv --output models\random_forest.joblib` | Train/validation/test counts, "Grouped by: recording_id", separate validation and test metrics | ☐ |
| H2 | Open `models\random_forest_splits.csv` | Each `recording_group` appears in only one `split` | ☐ |
| H3 | `python train_model.py --dataset data\training\synthetic_example.csv --output models\t.joblib` | Refused: synthetic data needs `--allow-synthetic` | ☐ |
| H4 | `python scripts\fetch_satnogs_dataset.py --per-class 150` (internet) | Prints "N good / M bad so far" as it goes; both numbers grow; it ends without errors | ☐ |
| H5 | Open 3 images in `data\satnogs\parse_check\` | Right half matches left half (same features, not upside-down) | ☐ |
| H6 | `python train_model.py --feature-set waterfall --dataset data\training\satnogs_waterfall_features.csv --output models\waterfall_rf.joblib` | Trains; "Grouped by: recording_id" (station); note the test F1: ____ | ☐ |
| H7 | `python scripts\evaluate_model.py --model models\waterfall_rf.joblib --dataset data\training\rsp03_waterfall_features.csv` | External test on RSP-03; note the F1 / ROC AUC: ____ | ☐ |
| H8 | Train on a copy of the SatNOGS CSV with all `label` 0 rows deleted | Refused: "only one class" | ☐ |

## I. Database and history

| # | Do | Expect | ✓ |
|---|---|---|---|
| I1 | Open `sdr-doppler-prototype\data\results\captures.sqlite3` in "DB Browser for SQLite" | Tables `capture_results`, `pass_positions`, `status_log` | ☐ |
| I2 | Run E1 on an **old** copy of the database (e.g. `..\captures-backup.sqlite3` via `--db`) | Works; old rows keep their data, new columns are empty for them (migration) | ☐ |

## J. Live dashboard

| # | Do | Expect | ✓ |
|---|---|---|---|
| J1 | `python dashboard.py --config capture_config.json`, open http://localhost:8050 | Page loads; "not running" (no heartbeat yet); totals and log match history.py | ☐ |
| J2 | Second terminal: `python auto_capture.py --config capture_config.json --demo` | Within 5 s: "waiting for pass", then "recording" (recorder RECORDING), then a new row in Recent captures | ☐ |
| J3 | Stop the demo, wait 2 minutes | "Now" shows "stalled?" or "stopped", not a frozen "recording" | ☐ |
| J4 | Sky track box | A curve with a green start and red end once a capture has a track | ☐ |
| J5 | `python dashboard.py --config capture_config.json --host 0.0.0.0`, open `http://<your-PC-IP>:8050` from a phone on the same Wi-Fi | Page loads (Windows may ask to allow it through the firewall) | ☐ |
| J6 | Open `http://localhost:8050/image?path=C:\Windows\win.ini` | "not found": files outside the project aren't served | ☐ |

## K. Desktop GUI (`sdr-doppler-prototype\launch_gui.bat`)

| # | Do | Expect | ✓ |
|---|---|---|---|
| K1 | Visualise IQ tab: choose `RSP-03_4.raw`, format ci16, rate 1000000, "Draw Waterfall" | A results window with the waterfall image | ☐ |
| K2 | Run Detection tab: same file, IQ format auto, "Run Detection" | Summary shows the rule and ML results | ☐ |
| K3 | Satellite Passes tab: "List Upcoming Passes" | The schedule text | ☐ |
| K4 | Satellite Passes tab: "Run Demo Capture" | After about 20 s, results with the demo capture | ☐ |
| K5 | Capture History tab: "Show History", then with a capture id | The tables from I/E2 and E3 | ☐ |
| K6 | Train Model tab with `rsp03_camras_features.csv` (untick "Allow synthetic") | Training summary with validation and test sections | ☐ |
| K7 | Convert SigMF tab with an `.sigmf-meta` + `.sigmf-data` pair | Writes the `.npy` | ☐ |

## L. Real satellite pass 📡 (the big one)

| # | Do | Expect | ✓ |
|---|---|---|---|
| L1 | Set your coordinates in `capture_config.json`; `run_station.bat` | Dashboard opens; schedule lists METEOR passes | ☐ |
| L2 | Leave it through one METEOR pass (max el ≥ 30°) | Dashboard: waiting → recording (about 10 min) → processing → new capture row | ☐ |
| L3 | Open that capture's waterfall (dashboard or the `…_waterfall.png` in results) | With a signal: a bright band about 120–150 kHz wide, straight (Doppler-corrected) | ☐ |
| L4 | `history.py --capture N` | `doppler_corrected: 1`, `doppler_max_hz` about 3000; track points present | ☐ |
| L5 | Optional ground truth: open the `.iq` in SatDump (METEOR M2-x LRPT, 1.024 Msps, baseband u8) | If SatDump decodes an image, the signal was real. Compare with the model's verdict. The recording is tuned 150 kHz below 137.9 MHz: give SatDump that offset, or record one test pass with `"tuning_offset_hz": 0` in the config | ☐ |
| L6 | Leave it running overnight | Several passes, all logged; disk use in the dashboard is sensible | ☐ |

## M. Other operating systems

| # | Do | Expect | ✓ |
|---|---|---|---|
| M1 | On a Mac or Linux laptop: `./setup.sh`, `python -m pytest` | Same results as A1/A2 | ☐ |
| M2 | `./run_station.sh --simulate` | Dashboard and schedule work | ☐ |

---

### Results summary

| Area | Pass | Fail | Notes |
|---|---|---|---|
| A Setup | | | |
| B Hardware | | | |
| C Recorder | | | |
| D Passes | | | |
| E Chain | | | |
| F Visualisation | | | |
| G Detection | | | |
| H ML | | | |
| I Database | | | |
| J Dashboard | | | |
| K GUI | | | |
| L Real pass | | | |
| M Other OS | | | |
