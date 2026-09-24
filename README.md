# RTL-SDR Satellite Capture, Detection & Archive

Capstone project: **database of signals, position histories, archiving and status monitoring** for a low-cost RTL-SDR ground station.

The system predicts when satellites pass overhead, records their radio signal with an RTL-SDR, removes the Doppler shift, decides with a rule-based detector and two machine-learning models whether a satellite was actually captured, stores everything in a database, keeps only the recordings worth keeping, and shows it all on a live dashboard.

> **Running the station with a dongle? Start with [QUICKSTART.md](QUICKSTART.md).** `setup`, then `check_dongle`, then `run_station`.

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
| `sdr-doppler-prototype/gui_app.py` | Desktop GUI (Windows: `launch_gui.bat`, macOS/Linux: `launch_gui.sh`) |

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

# Train the model (grouped train/validation/test split, see below)
cd sdr-doppler-prototype
python train_model.py --dataset data/training/rsp03_camras_features.csv --output models/random_forest.joblib
```

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
python -m pytest -m hardware   # only with a real dongle attached
```

## Data

<<<<<<< HEAD
The client's recordings live in `Data/` (git-ignored, several GB). `sdr-doppler-prototype/data/training/README.md` describes the labelled datasets and how they were built (`scripts/label_known_carrier.py`).
=======
```bash
python src/main.py --input data/spectrograms/spectrogram0_136800000.txt --output data/results/
```

Useful tuning options:

```bash
python src/main.py \
  --input data/raw/sample.npy \
  --output data/results/ \
  --sample-rate 240000 \
  --center-freq 136800000 \
  --snr-threshold-db 6 \
  --min-valid-ratio 0.45 \
  --min-drift-hz 1500 \
  --max-smoothness-hz 8000 \
  --save-image
```

## Input Formats

- `.npy` 1D complex array: raw IQ samples.
- `.bin`, `.iq`, `.dat`: raw binary IQ samples, default dtype `complex64`.
- `.npy` 2D array: spectrogram matrix in dB.
- `.txt`, `.csv`: spectrogram matrix in dB.

The notebook material this prototype was based on saved spectrogram text files with time rows and frequency columns. This prototype uses the same assumption for text matrices.

## Database

Initialize manually:

```bash
bash scripts/init_db.sh
```

The main pipeline also initializes the database automatically. Results are written to:

```text
data/results/captures.sqlite3
```

Table: `capture_results`

- `id`
- `input_file`
- `timestamp_utc`
- `detection_result`
- `confidence_score`
- `valid_signal_ratio`
- `frequency_drift_hz`
- `smoothness_score`
- `spectrogram_image_path`
- `raw_iq_file_path`
- `notes`

## Notes

This detector is a rule-based first pass. It is useful for sorting captures into "worth inspecting" and "probably noise" groups, not for final scientific classification.

## Connecting Auto Capture to the Database

`iq-recorder` (the RTL-SDR recording manager) and this detection pipeline
used to be completely disconnected: the recorder wrote a `.iq` file +
JSON metadata sidecar and stopped, and detection only ran manually
against a file you pointed `src/main.py` at by hand - nothing ever wrote
an auto-captured recording into `capture_results`.

`pipeline.py` at the repo root connects them: it drives one recording
through `rtl_recorder.RTLSDRRecorder`, then feeds the resulting IQ file
through the same spectrogram -> feature extraction -> rule/ML detection
pipeline as `src/main.py`, and inserts the result into this SQLite
database.

From the repo root:

```bash
python pipeline.py \
  --satellite METEOR-M2-4 \
  --frequency 137900000 \
  --sample-rate 2400000 \
  --duration 60 \
  --gain 30
```

Add `--simulate` to exercise the full capture -> detect -> database path
without RTL-SDR hardware attached. `capture_and_detect()` /
`process_recording()` in `pipeline.py` are also importable directly for a
future scheduler to call after each pass.
>>>>>>> 482f8559d715eca22d15970253af36deb909de15
