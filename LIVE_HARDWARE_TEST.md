# Live hardware test - checklist

For the teammate with the RTL-SDR dongle. It turns "implemented, verified in simulation" into real
evidence. Allow about 1 hour, most of it waiting for a pass. **Record what actually happens.** A
failed step with its error message is useful evidence. A tidied-up result is not.

Commands are for **Windows Command Prompt (`cmd`, not PowerShell) from the repository root**. On
macOS/Linux use the `.sh` scripts, `.venv/bin/python` instead of `.venv\Scripts\python`, and `$EV` instead of `%EV%`.

## 0. Before you start

- [ ] RTL-SDR Blog V3 (or similar) dongle, antenna for 137 MHz (V-dipole ~53 cm per arm is ideal), USB cable
- [ ] Laptop with internet (TLE download), Python 3.10+ ("Add python.exe to PATH" ticked), ~5 GB free disk
- [ ] RTL-SDR tools unzipped (e.g. `C:\rtl-sdr`) and the WinUSB driver installed with Zadig - see QUICKSTART.md step 0
  (**V4 dongle:** use the RTL-SDR Blog release, not the osmocom zip)
- [ ] Close SDR#, SDR++ and anything else that might hold the dongle
- [ ] **RAM check:** processing one pass peaks at about **6.7 GB** (evidence/performance/). On an
      **8 GB** laptop, add `"max_detection_seconds": 60` to the `recording` section of
      `capture_config.json` (about 3.5 GB peak), and write that down in the evidence

Make the evidence folder (use today's date) and keep this terminal open for everything below:

```bat
set EV=evidence\live\2026-09-25
mkdir %EV%
```

## 1. Record the environment

```bat
git rev-parse HEAD                      > %EV%\environment.txt
.venv\Scripts\python --version          >> %EV%\environment.txt
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"Total Physical Memory" >> %EV%\environment.txt
```

## 2. Set up and check the environment

```bat
setup.bat
.venv\Scripts\python pipeline.py --doctor > %EV%\doctor.txt
```

- [ ] `setup.bat` finishes with "Setup finished." (or its notes say what is missing)
- [ ] `%EV%\doctor.txt`: `rtl_sdr`, `rtl_test` and the device all show `[OK ]`. If not, follow the printed instruction
      (or `setx RTL_SDR_HOME C:\rtl-sdr`, then open a **new** terminal) and repeat

## 3. Dongle check (1 minute)

```bat
.venv\Scripts\python check_dongle.py > %EV%\check_dongle.txt
```

- [ ] Status `SUCCESS` and a non-zero byte count
- [ ] Open `recordings\dongle_check\dongle_check_waterfall.png` and copy it to `%EV%`. It should show a bright band ~200 kHz wide
      in the middle (the 92.0 MHz FM station; try `--freq 95.0e6` for another)

Optional, more thorough (3 recordings of 30 s, device release between them):

```bat
.venv\Scripts\python -m pytest -m hardware -v > %EV%\hardware_pytest.txt
```

- [ ] 3 passed. If any fail, keep the whole output file

## 4. Station config and next passes

`setup.bat` created `capture_config.json`. Check `station` (lat/lon/alt of where you are standing - Google Maps,
right-click, copy coordinates) and leave METEOR-M2-3 / METEOR-M2-4 in `satellites`.

```bat
.venv\Scripts\python auto_capture.py --config capture_config.json --list-only --hours 12 > %EV%\schedule.txt
```

- [ ] A table of passes (UTC times, max elevation, `[sgp4]`; `[kepler-j2]` means the `sgp4` package is missing - pass times then drift by 10-30 s). Pick one with max elevation >= 30 degrees
- [ ] `tle_cache\57166.tle` / `59051.tle` exist (TLEs downloaded from CelesTrak)

## 5. Capture a real pass automatically

Start **10+ minutes before** the chosen pass's AOS. Two ways (pick one):

**A - the full station (dashboard + capture loop):**

```bat
run_station.bat
```

**B - one planning window, then exit** (easier to capture the console output):

```bat
.venv\Scripts\python auto_capture.py --config capture_config.json --hours 2 > %EV%\auto_capture_console.txt 2>&1
```

(B records every non-overlapping pass in the next 2 h and then stops; Ctrl+C stops early and finalises the
current recording.) While it runs:

- [ ] Dashboard at <http://localhost:8050> (A opens it; for B run `.venv\Scripts\python dashboard.py --config capture_config.json`
      in a second terminal). The "Now" box moves from waiting for pass to recording to processing to idle. Screenshot each state into `%EV%`
- [ ] Recording starts ~30 s before AOS and stops ~30 s after LOS (pre/post buffers)
- [ ] Processing then takes ~1-2 minutes

## 6. Files that should exist afterwards

| File | Where | Check |
|---|---|---|
| Raw IQ | `recordings\METEOR-M2-3_<YYYYMMDD_HHMMSS>_137750000Hz.iq`, or `recordings\rejected\...` if judged "no satellite" | Size ~= 2 x 1,024,000 x seconds recorded (~1.6-2 GB for a 13-15 min window) |
| Recorder sidecar | same name `.json`, next to the `.iq` | `scheduled_aos`/`los`, `actual_start`/`stop`, `simulated: false` |
| Summary | `sdr-doppler-prototype\data\results\<name>_<time>_summary.json` | rule + ML results |
| Spectrogram | `...\data\results\<name>_<time>_spectrogram.png` | opens; copy to `%EV%` |
| Doppler-corrected waterfall | `...\data\results\<name>_<time>_waterfall.png` | title says "Doppler-corrected"; a METEOR signal is a ~120-150 kHz wide vertical band near the centre; copy to `%EV%` |
| Schedule / heartbeat | `recordings\schedule.json`, `recordings\live_status.json` | present |
| Database | `sdr-doppler-prototype\data\results\captures.sqlite3` | see section 7 |

(The `.iq` is named after the **tuned** frequency, 150 kHz below the downlink, because of offset tuning. That is expected.)

## 7. Database fields to verify

```bat
.venv\Scripts\python sdr-doppler-prototype\src\history.py > %EV%\history.txt
.venv\Scripts\python sdr-doppler-prototype\src\history.py --capture <id> > %EV%\capture_<id>.txt
```

In `capture_<id>.txt` check:

- [ ] `satellite_name`, `norad_id` (57166 / 59051), `target_frequency_hz` 137900000, `frequency_hz` 137750000
- [ ] `recording_status` SUCCESS, `processing_status` DETECTED, **`simulated` 0**
- [ ] `scheduled_aos` / `scheduled_los` match the schedule; `actual_recording_start` / `stop` about 30 s outside them
- [ ] `recording_duration_seconds` and `output_file_size` agree (size ~= 2 x sample rate x duration)
- [ ] `rule_detection_result` / `rule_confidence_score`, `ml_detection_result` / `ml_confidence_score` / `model_version`
- [ ] `doppler_corrected` 1 and `doppler_max_hz` a few kHz (~3-3.5 kHz for METEOR at 137.9 MHz)
- [ ] `iq_retention` + `retention_reason` + `decision_source`, and the file is really where `raw_iq_file_path` says
- [ ] "Position history" section: one row every 10 s, elevation rising and then falling, peak close to the scheduled max elevation
- [ ] `history.txt` status log: scheduler WAITING, recorder SUCCESS, pipeline DONE

## 8. Dashboard and GUI (manual)

Dashboard (<http://localhost:8050>):

- [ ] Recent captures shows the pass with its rule / IQ-model / waterfall-model results and IQ file outcome
- [ ] Latest waterfall image shows the pass; the sky track shows the pass arc; totals count it
- [ ] Stop `auto_capture` and wait 60 s: the phase changes to "stalled?" (or "stopped" after a clean exit)

GUI (`sdr-doppler-prototype\launch_gui.bat`). **Maximise the window.** At the default 1150x820 size the
Results pane at the bottom is squeezed to nothing and command output is hard to see:

- [ ] Opens without errors; all seven tabs switch
- [ ] Capture History: Show History lists the real capture; entering its id shows the position history
- [ ] Visualise IQ: choose the real `.iq`, Draw Waterfall: a pop-up with the waterfall appears
- [ ] Run Detection on the real `.iq` (tick Save spectrogram image): Results and Image Preview show it
- [ ] Satellite Passes: List Upcoming Passes shows the same schedule as step 4
- [ ] Screenshot each into `%EV%`

**Do not count the IQ model's verdict as detection.** It has not been validated on RTL-SDR data. In a pre-flight run
through the real capture code with a fake device that wrote only zero bytes, it reported "satellite" at 0.95
confidence, while the rule detector said no. Use the waterfall picture (7b) as the judge, and record the model
values only as observations.

## 9. Record the outcome

Fill this in as `%EV%\RESULT.md` (copy the table). Mark each line PASS, FAIL or NOT RUN, and put
the exact error text in Notes for anything that is not PASS:

| # | Step | Result | Notes (errors, what you saw) |
|---|---|---|---|
| 1 | Environment recorded | | |
| 2 | Doctor: tools + device OK | | |
| 3 | Dongle check: FM band visible | | |
| 3b | Hardware pytest 3/3 (optional) | | |
| 4 | Pass list from real TLEs | | |
| 5 | Automatic capture started at AOS - buffer, stopped at LOS + buffer | | |
| 6 | All expected files present | | |
| 7 | DB row complete, `simulated`=0, position history stored | | |
| 7b | Satellite visible in the waterfall (your own judgement) | | |
| 7c | Rule / ML verdicts (write the values; say if they agree with 7b) | | |
| 8 | Dashboard checks | | |
| 8b | GUI checks | | |

Name, date, dongle model, antenna, location, weather/obstructions: ______________________

Then commit `evidence/live/<date>/` (not the `.iq` files, which are too big for git; note where they are kept).
A pass where the satellite is **not** visible is still a valid hardware run: it verifies capture, storage and
processing, but not detection. Say which one you got.
