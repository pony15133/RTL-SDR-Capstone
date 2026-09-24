# RTL-SDR Satellite Capture, Detection & Archive

Capstone project: **database of signals, position histories, archiving and status monitoring** for a low-cost RTL-SDR ground station.

The system predicts when satellites pass overhead and records their radio signal with an RTL-SDR. It removes the Doppler shift, then decides with a rule-based detector and up to two machine-learning models whether a satellite was actually captured. It stores everything in a database, keeps only the recordings worth keeping, and shows it all on a live dashboard. The second (waterfall) model only exists after `setup --fetch-satnogs` has downloaded its training data.

> **Running the station with a dongle? Start with [QUICKSTART.md](QUICKSTART.md).** `setup`, then `check_dongle`, then `run_station`.
>
> **Handover:** current verified status, test results and known limitations are in [FINAL_HANDOVER_STATUS.md](FINAL_HANDOVER_STATUS.md). The real-hardware test procedure is [LIVE_HARDWARE_TEST.md](LIVE_HARDWARE_TEST.md). Saved evidence is in [`evidence/`](evidence/).

```
 TLE (CelesTrak)            iq-recorder                     sdr-doppler-prototype
 ───────────────     ────────────────────────     ───────────────────────────────────────────
 pass prediction ──> wait for AOS ─> rtl_sdr ──> spectrogram ─> features ─┬─> rule detector
 (AOS/LOS/max el)    record AOS..LOS (+buffers)      (.iq + .json)          └─> Random Forest (ML)
                                                                                     │
          SQLite: capture_results · pass_positions · status_log  <───────────────────┤
          IQ retention: keep / archive / delete by ML confidence <───────────────────┘
```

## Layout

| Path | What it is |
|---|---|
| `auto_capture.py` | **The whole pipeline in one command**: predict passes → wait → record → detect → database → retention |
| `pipeline.py` | One recording → detection → database (used by `auto_capture.py`; also a manual CLI) |
| `capture_config.example.json` | Ground station (Singapore), recording defaults, satellite list (copied to your own `capture_config.json` by setup) |
| `dashboard.py` | Live status dashboard at http://localhost:8050 |
| `setup.*` / `check_dongle.*` / `run_station.*` | One-time setup, 1-minute hardware test, start dashboard + capture (Windows `.bat`, macOS/Linux `.sh`) |
| `iq-recorder/` | `rtl_recorder` package: cross-platform `rtl_sdr` control, pass prediction (`passes.py`), `record_pass()`, environment `doctor` |
| `sdr-doppler-prototype/` | Signal processing, features, rule + ML detectors, training, database, GUI, visualisation |
| `sdr-doppler-prototype/gui_app.py` | Desktop GUI (Windows: `sdr-doppler-prototype\launch_gui.bat`, macOS/Linux: `sdr-doppler-prototype/launch_gui.sh`) |
| `tools/` | Evidence tools: `verify_simulated_pipeline.py`, `benchmark_processing.py`, `gui_smoke_test.py` |
| `evidence/` | Saved test, pipeline, ML, performance and GUI evidence (see FINAL_HANDOVER_STATUS.md) |

## Setup (Windows, macOS, Linux)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r sdr-doppler-prototype/requirements.txt -r iq-recorder/requirements.txt
python pipeline.py --doctor        # checks rtl_sdr/rtl_test, the dongle, folders and packages
```

`--doctor` finds the RTL-SDR tools on PATH or in the usual install folders for your OS. If they're missing, it prints the install step for that OS: `brew install librtlsdr` (macOS), `sudo apt install rtl-sdr` (Linux), or the osmocom zip plus Zadig (Windows). You can also point `RTL_SDR_HOME` at the folder. Nothing hard-codes `rtl_sdr.exe`.

## Everyday commands (from the repo root)

```bash
# 1. What passes are coming? (TLEs downloaded + cached in tle_cache/)
python auto_capture.py --config capture_config.example.json --list-only

# 2. 20-second end-to-end demo, no hardware
python auto_capture.py --config capture_config.example.json --demo

# 3. Real unattended capture (Ctrl+C stops cleanly)
python auto_capture.py --config capture_config.json --forever

# Look at any recording - waterfall only, no ML
python sdr-doppler-prototype/src/visualize.py --input recordings/ISS_..._145800000Hz.iq

# What's in the database? (captures, status log, one capture's satellite track)
python sdr-doppler-prototype/src/history.py
python sdr-doppler-prototype/src/history.py --capture 12

# Desktop GUI
sdr-doppler-prototype/launch_gui.sh          # Windows: sdr-doppler-prototype\launch_gui.bat

# Live dashboard (reads the database and the heartbeat auto_capture.py writes)
python dashboard.py --config capture_config.json          # http://localhost:8050

# Train the model (grouped train/validation/test split, see below)
cd sdr-doppler-prototype
python train_model.py --dataset data/training/rsp03_camras_features.csv --output models/random_forest.joblib
python scripts/ml_evaluation_report.py      # evidence/ml/ML_EVALUATION.md
```

## Configuration (`capture_config.json`)

`setup` copies `capture_config.example.json` to `capture_config.json`, which git ignores. Paths are relative to the repo root. Run everything from there.

| Section / key | Meaning (default) |
|---|---|
| `station.name`, `lat_deg`, `lon_deg`, `alt_m` | Ground station (Singapore campus) |
| `recording.sample_rate` | Samples/s for every satellite unless it sets its own (1,024,000) |
| `recording.hours` | How far ahead to plan passes (24) |
| `recording.min_elevation_deg` | Skip passes that peak lower (15) |
| `recording.pre_buffer`, `post_buffer` | Seconds recorded before AOS / after LOS (30 / 30) |
| `recording.output_dir` | Where `.iq` + `.json` recordings, `schedule.json` and `live_status.json` go (`recordings`) |
| `recording.retention`, `keep_threshold` | `keep-all` / `archive-negatives` (to `recordings/rejected/`) / `delete-negatives`, and the ML confidence needed to keep (archive-negatives, 0.5) |
| `recording.ml_model`, `waterfall_model` | Trained models (`sdr-doppler-prototype/models/random_forest.joblib`, `.../waterfall_rf.joblib`). A missing model is reported as `MODEL_NOT_AVAILABLE` rather than an error |
| `recording.save_image` | Save spectrogram PNGs (true) |
| `recording.tuning_offset_hz` | Tune this far below the downlink so the dongle's DC spike stays off the signal (150,000) |
| `recording.max_detection_seconds` | Seconds of each recording used for detection (120). Peak memory grows with it: about 6.7 GB at 120 s, 3.5 GB at 60 s at 1.024 Msps. **Use 60 on an 8 GB computer** |
| `recording.db_path`, `results_dir`, `tle_cache_dir`, `tle_file`, `device_index`, `rtl_sdr_path` | Optional overrides (DB below; results next to the DB; `tle_cache/`; download TLEs; 0; auto-detect) |
| `satellites[]`: `name`, `norad_id`, `frequency_hz`, `gain`, `waterfall_span_hz`, optional `sample_rate`, `min_elevation_deg` | What to record |

Most `recording` keys also have an `auto_capture.py` option (`--help`), and the option wins over the config. `tuning_offset_hz`, `waterfall_model`, `tle_cache_dir`, `device_index` and `rtl_sdr_path` can only be set in the config.

## Outputs

| What | Where |
|---|---|
| Raw IQ + recorder JSON sidecar | `recordings/<SAT>_<YYYYMMDD_HHMMSS>_<tunedHz>Hz.iq/.json`, or `recordings/rejected/` if archived |
| Summary JSON, spectrogram PNG, Doppler-corrected waterfall PNG | `sdr-doppler-prototype/data/results/` (next to the database) |
| Database | `sdr-doppler-prototype/data/results/captures.sqlite3` |
| Pass schedule, dashboard heartbeat | `recordings/schedule.json`, `recordings/live_status.json` |
| TLE cache | `tle_cache/<norad>.tle` (12 h) |
| Trained models (not in git) | `sdr-doppler-prototype/models/*.joblib` + `.json` metadata |
| `src/main.py` / GUI "Run Detection" runs | `sdr-doppler-prototype/data/results/session_<date>_<time>/` |

## How the supervisors' requests are covered

| Request | Where |
|---|---|
| Retrieve pass data, schedule and set start/stop/duration automatically | `iq-recorder/rtl_recorder/passes.py` (TLE → AOS/LOS/max elevation/Doppler; uses `sgp4` if installed, otherwise a built-in model checked against the official SGP4 test case), `RTLSDRRecorder.record_pass()`, `auto_capture.py` |
| Database updates every time the recorder runs (Bijaya, 24 Aug) | `pipeline.process_recording()` writes the recorder metadata (satellite, NORAD, frequency, rate, gain, AOS/LOS, actual start/stop, sizes). Failed or busy recordings are logged too |
| Components tested together as one package | `tests/test_pipeline.py`, `tests/test_auto_capture.py` (simulated end-to-end) |
| Separate visualisation, no model needed (Bijaya, 24 Aug) | `src/visualize.py` + GUI tab **Visualise IQ**. Reads rtl_sdr `.iq` (cu8), SigMF/CAMRAS (ci16), `.bin` (complex64), SDR# `.wav` |
| Cross-platform, no `rtl_sdr.exe` (Bijaya, 24 Aug) | `rtl_recorder/utils.find_executable()` (per-OS search), `rtl_recorder/doctor.py`, `launch_gui.sh` |
| ML confidence decides whether to keep the IQ file | `src/retention.py`: `keep-all` / `archive-negatives` / `delete-negatives`, with the decision logged per row |
| Proper train / validation / test methodology (Xinyi, 9 Sep) | `src/ml/train.py`: split by recording, tuned on validation, test scored once, `<model>_splits.csv` |
| Doppler correction | `src/doppler.py`: predicted Doppler curve from TLE + station, removed from every recording; offset tuning keeps the dongle's DC spike away from the signal; `visualize.py --doppler` |
| Real training data from online sources | `scripts/fetch_satnogs_dataset.py`: vetted good/bad observations from the worldwide SatNOGS network → waterfall model (`train_model.py --feature-set waterfall`); external test on the CAMRAS RSP-03 pass (`scripts/evaluate_model.py`) |
| Live status dashboard | `dashboard.py` (heartbeat from `auto_capture.py`) |
| Position histories and status monitoring (project title) | Tables `pass_positions` (az/el/range/Doppler every 10 s per pass) and `status_log`, shown by `history.py` and the GUI tab **Capture History** |

## Database (`sdr-doppler-prototype/data/results/captures.sqlite3`)

- **`capture_results`**: one row per recorder run. Detection (rule and ML), recording metadata, `processing_status`, and `iq_retention` with its reason and score. Older databases are migrated in place without losing data.
- **`pass_positions`**: the satellite's position during each recorded pass, linked by `capture_id`.
- **`status_log`**: timestamped scheduler, recorder and pipeline events.

## Tests

```bash
python -m pytest            # from the repo root: all three projects, hardware tests skipped
python -m pytest -m hardware   # only with a real dongle attached (see LIVE_HARDWARE_TEST.md)
python tools/verify_simulated_pipeline.py   # demo + TLE pass, checks every stored field -> evidence/pipeline/
python tools/benchmark_processing.py        # time + peak memory per recording -> evidence/performance/
python tools/gui_smoke_test.py              # needs a display (or: xvfb-run -a ...) -> evidence/gui/
```

Latest results are in FINAL_HANDOVER_STATUS.md.

## Data

The client's recordings live in `Data/` (git-ignored, several GB). `sdr-doppler-prototype/data/training/README.md` describes the labelled datasets and how they were built (`scripts/label_known_carrier.py`).
