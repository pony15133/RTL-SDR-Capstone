# SDR Doppler Prototype

Backend-only Python prototype for detecting candidate satellite Doppler signals from RTL-SDR IQ captures or precomputed spectrogram matrices.

The code is intentionally small and readable for a university capstone prototype. It does not include a GUI, orbital mechanics, cloud deployment, or production capture orchestration.

Two detectors run side by side on every capture, so their performance can be compared:

- **Rule-based baseline** (`src/detect.py`) - threshold logic on valid-signal ratio, frequency drift, and trace smoothness. The original detector, unchanged in behaviour.
- **Random Forest classifier** (`src/ml/`, `src/detection/ml_detector.py`) - a supervised scikit-learn model trained on labelled captures. See "Machine Learning Component" below.

## What Is Included

- Loads sample input data from a file.
- Converts raw IQ samples to a spectrogram with `scipy.signal.spectrogram`.
- Loads existing spectrogram matrices from `.txt`, `.csv`, or 2D `.npy` files.
- Extracts a shared feature vector (`src/features/extractor.py`) consumed by **both** detectors - no duplicated feature-extraction logic.
- Rule-based detector: simple thresholds on valid-signal ratio, frequency drift, and trace smoothness.
- Random Forest detector: trained classifier + estimated confidence (probability), with graceful `MODEL_NOT_AVAILABLE` behaviour when no model has been trained yet.
- Saves a JSON result summary (both detectors' results).
- Optionally saves a spectrogram PNG.
- Stores one row per capture in SQLite, including both detectors' results and the ML model version used.
- Stores the raw IQ file path only when detection is positive.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

From this folder:

```bash
python scripts/make_synthetic_sample.py
```

```bash
python src/main.py --input data/raw/sample.npy --output data/results/ --save-image
```

For a text spectrogram matrix:

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

### Converting SigMF captures

If your capture is in [SigMF](https://github.com/sigmf/SigMF) format (a `.sigmf-meta` JSON file plus a sibling `.sigmf-data` raw binary file - common when downloading third-party SDR datasets), convert it to a project-ready `.npy` first:

```bash
python scripts/convert_sigmf_to_iq.py --meta capture.sigmf-meta --data capture.sigmf-data --output data/raw/capture.npy
```

This reads `global."core:datatype"` from the metadata (`ci16_le`, `cu8`, `cf32_le`, etc. - any complex SigMF datatype) to interpret the raw bytes, deinterleaves I/Q, scales integer formats to roughly `[-1, 1]` (unsigned formats like `cu8` are DC-centred the same way RTL-SDR's own native format is), and writes a `complex64` `.npy` that `main.py`/`label_capture.py`/the other tools all already understand. It also prints `core:sample_rate` and the first capture's `core:frequency` from the metadata, if present, as a reminder of what to pass as `--sample-rate`/`--center-freq` downstream - the SigMF metadata isn't embedded in the `.npy` itself. A real-valued (non-IQ) SigMF file is rejected with a clear error rather than silently misread. This is also wired up as the "Convert SigMF" option in `gui_app.py`.

## Machine Learning Component

**Status: IMPLEMENTED BUT NOT VALIDATED.** The Random Forest pipeline is fully operational, but no model has been trained on real, labelled RTL-SDR captures yet - only synthetic data (clearly marked as such) has been used to verify the pipeline runs correctly. Do not treat any evaluation metrics produced so far as real-world accuracy.

### Architecture

```
Spectrogram --> features/extractor.py --> FeatureVector --+--> detect.py (rule baseline)
                                                            +--> detection/ml_detector.py (Random Forest)
```

Both detectors consume the exact same `FeatureVector` from `extract_features()` - the feature-extraction math exists in one place only.

- `src/features/extractor.py` - `extract_features()`, the fixed-order `FEATURE_NAMES`, and `feature_vector_to_array()` (order-safe, sklearn-safe conversion).
- `src/detect.py` - the rule-based baseline, refactored to consume the shared `FeatureVector` (behaviour unchanged - `tests/test_detect.py` is the regression guard).
- `src/detection/ml_detector.py` - inference: `try_load_model()` / `predict()` / `run_ml_detection()`. Never raises; a missing or incompatible model results in `MODEL_NOT_AVAILABLE`/`MODEL_INVALID`, not a crash.
- `src/ml/model.py` - `ModelBundle`: pairs a fitted classifier (`.joblib`) with a metadata sidecar (`.json`) - a model file is never used without knowing its feature schema and provenance.
- `src/ml/evaluation.py` - metrics, cross-validation (skipped with a stated reason on datasets too small to be meaningful), feature importance.
- `src/ml/train.py` / `train_model.py` - the training CLI.

### Feature Schema

| Feature | Notes |
|---|---|
| `snr_db` | Estimated, not calibrated |
| `frequency_drift_hz` | |
| `drift_rate_hz_per_second` | Depends on a real time axis - see limitation below |
| `smoothness_score` | Can be `inf` in the raw FeatureVector; sanitised to a large finite sentinel for ML input |
| `valid_signal_ratio` | |
| `peak_power` | dB |
| `mean_power` | dB |
| `occupied_bandwidth_hz` | Threshold-crossing estimate, not a formal 99% OBW |
| `signal_duration_seconds` | Depends on a real time axis - see limitation below |

Full column-by-column documentation, the CSV dataset format, and how to collect real labelled data: [`data/training/README.md`](data/training/README.md).

### Training

```bash
python train_model.py --dataset data/training/features.csv --output models/random_forest.joblib
```

- Validates required columns and values before doing anything else.
- Refuses to train on any row marked `is_synthetic=1` unless `--allow-synthetic` is passed; a model trained that way is permanently marked `trained_on_synthetic_data: true` in its metadata.
- Stratified train/test split (falls back to unstratified with a warning if the dataset is too small/imbalanced for it).
- Fixed `--random-state` (default 42) for reproducible experiments.
- Prints accuracy, precision, recall, F1, confusion matrix, ROC-AUC where computable, and cross-validation (or a stated reason it was skipped).
- Saves the model + a metadata sidecar (model type/version, timestamp, feature schema, RF params, sample counts, label distribution, evaluation metrics) and a feature-importance CSV (`--save-importance-chart` for a PNG too).

Try it with the bundled synthetic demo dataset (pipeline verification only - do not read its metrics as real accuracy):

```bash
python train_model.py --dataset data/training/synthetic_example.csv --output models/random_forest.joblib --allow-synthetic
```

### Inference

`src/main.py` runs ML detection automatically if `models/random_forest.joblib` (or `--ml-model <path>`) exists; pass `--no-ml` to skip it. If no model exists, the pipeline still runs the rule-based baseline and reports `ml_status=MODEL_NOT_AVAILABLE` rather than failing.

```python
from detection.ml_detector import run_ml_detection
result = run_ml_detection(model_path, features)
# result.ml_detection_result: bool | None
# result.ml_confidence_score: float | None  - the model's estimated probability
#   of the positive class, NOT a scientifically guaranteed probability
```

### Feature Importance

`train_model.py` always prints and saves a `feature | importance` table from the trained forest's impurity-based `feature_importances_`. This shows which features the trees found useful for splitting on the given dataset - it does not establish that a feature causally determines whether a signal is a satellite pass, and several features are highly correlated with the rule detector's own thresholds by construction (see `data/training/README.md`).

### Collecting Real Training Data

```bash
python scripts/label_capture.py --input <capture> --dataset data/training/features.csv --save-image --output data/results/
```

Runs the pipeline on a real capture, shows both detectors' opinions and the extracted features, and appends your label to the dataset (always `is_synthetic=0`). See `data/training/README.md` for the full collection workflow (recording real passes vs. negative examples, minimum dataset size guidance, etc.).

### Building a Training Dataset from Raw IQ Recordings

For a whole folder of raw captures at once (instead of one file per command), use `scripts/build_training_dataset.py`. It walks a directory of labelled `.bin`/`.iq`/`.dat`/`.npy` recordings and, for each one, runs the exact same pipeline as everywhere else in the project - `load_data.load_input()` -> `spectrogram.iq_to_spectrogram()` -> `features.extractor.extract_features()` -> `feature_row_values()` (the shared, order-safe `FEATURE_NAMES` conversion) - then appends one labelled row to the training CSV. Training and inference always compute features with this same code, so a model can never see numbers that were derived differently at training time vs. prediction time.

```text
RAW IQ                         RAW IQ
   |                              |
Spectrogram                  Spectrogram
   |                              |
Shared Feature Extraction    SAME Shared Feature Extraction
   |                              |
Training CSV                 Saved Random Forest
   |                              |
Random Forest Training       Prediction / probability
   |
Saved Model
```

**Directory structure.** Point `--input-dir` at a folder of subfolders named for their class, e.g.:

```text
training_data/
    satellite/
        meteor_m2_4_001.npy
        meteor_m2_4_002.bin
        noaa_19_001.npy
    noise/
        empty_001.npy
        interference_001.bin
        terrestrial_signal_001.npy
```

**Supported file formats:** `.bin`/`.iq`/`.dat` (raw binary IQ, dtype set by `--binary-dtype`, default `complex64`) and `.npy` (1D/complex = raw IQ, 2D = an already-computed spectrogram matrix - `load_input()` inspects the array itself rather than assuming every `.npy` is raw IQ). Each recording produces **one** CSV row; a raw file is not silently split into several. If a capture's real signal only occupies a small fraction of a much longer recording, use `scripts/chunk_bin_to_dataset.py` instead (see below) - that is an explicit, separate tool, not something this script does automatically.

**Labelling.** The folder name is the label source by default - `satellite`/`1`/`positive`/`pos`/`candidate` -> `1`, `noise`/`0`/`negative`/`neg` -> `0` (see `DEFAULT_CLASS_MAP` in the script). This isn't hard-wired for a fixed set of class names: pass `--class-map "interference=0,unknown_sat=1"` to add or override folder-name -> label mappings without touching the code, or use `--labels-csv`/`--label` for manifest- or whole-folder-based labelling instead (see `data/training/README.md` for all label-source modes).

**Required metadata.** `--sample-rate`, `--center-freq`, `--nperseg`, `--noverlap`, and `--snr-threshold-db` all default to the same project-wide values `src/config.py` and `src/main.py` use, but real recordings usually need their true sample rate/centre frequency passed explicitly - the recorder does not currently embed this metadata in the `.npy`/`.bin` file itself, so **this script will not guess it**. If your recordings vary in sample rate or centre frequency, either process each subfolder in a separate run with the matching flags, or convert captures to SigMF first (`scripts/convert_sigmf_to_iq.py`) and record the parameters per capture some other way (e.g. per-folder or in a notes column).

Example command:

```bash
python scripts/build_training_dataset.py \
    --input-dir training_data --recursive \
    --dataset data/training/features.csv \
    --sample-rate 2400000 --center-freq 137900000 --binary-dtype complex64
```

**Output.** A CSV at `--dataset` with columns `capture_id`, the nine `FEATURE_NAMES` columns (`snr_db`, `frequency_drift_hz`, `drift_rate_hz_per_second`, `smoothness_score`, `valid_signal_ratio`, `peak_power`, `mean_power`, `occupied_bandwidth_hz`, `signal_duration_seconds`), `label`, and provenance columns (`is_synthetic`, `source_file`, `sample_rate_hz`, `nperseg`, `noverlap`, `notes`) - this is exactly `train_model.py`'s expected schema, not a second incompatible format. A file that fails to load or process (corrupt/truncated capture, wrong `--binary-dtype`, ...) is reported and skipped, never aborting the rest of the batch; if an existing `--dataset` file has an incompatible header (an older schema), the run fails cleanly with a clear error instead of writing misaligned rows. When finished, it prints:

```text
Processed: 6
Successful: 6
Failed: 0

Class distribution (this run, by label):
  0: 3
  1: 3

Class distribution (this run, by source folder):
  noise: 3
  satellite: 3

Output: data/training/features.csv
Train with: python train_model.py --dataset data/training/features.csv --output models/random_forest.joblib
```

...plus a `WARNING` if the resulting dataset ends up with only one class, or one class outnumbering the other by 3x or more.

Then train exactly as before:

```bash
python train_model.py --dataset data/training/features.csv --output models/random_forest.joblib
```

The CSV stores **extracted numerical features only** - the Random Forest never trains directly on raw IQ samples; `train_model.py` reads the CSV, not the recordings.

Training itself is unchanged either way - `label_capture.py` and `build_training_dataset.py` write the same CSV schema, so `train_model.py --dataset data/training/features.csv --output models/random_forest.joblib` works regardless of how the CSV was built. See `data/training/README.md` for the full label-source and schema documentation.

## Database

Initialize manually:

```bash
bash scripts/init_db.sh
```

The main pipeline also initializes (and safely migrates) the database automatically. Results are written to:

```text
data/results/captures.sqlite3
```

Table: `capture_results`

- `id`, `input_file`, `timestamp_utc`, `notes`, `spectrogram_image_path`, `raw_iq_file_path`
- `detection_result`, `confidence_score`, `valid_signal_ratio`, `frequency_drift_hz`, `smoothness_score` - the original rule-detector columns, unchanged, still populated by it.
- `rule_detection_result`, `rule_confidence_score` - the same rule-detector result, added under names that read naturally next to the ML columns below.
- `ml_detection_result`, `ml_confidence_score`, `model_version` - the Random Forest result; all three are `NULL` for a capture processed while no trained model was available.

An existing database created before the ML component is migrated automatically and non-destructively (`ALTER TABLE ... ADD COLUMN`, existing rows keep their data with `NULL` in the new columns) the next time `init_db()` runs.

## Known Limitations

- The rule-based detector is a first pass, useful for sorting captures into "worth inspecting" vs. "probably noise", not final scientific classification.
- The Random Forest classifier is **implemented but not validated** - no model has been trained on real, independently-verified labelled captures yet.
- `occupied_bandwidth_hz` is a threshold-crossing bandwidth estimate, not a formal 99%-power occupied bandwidth measurement.
- `signal_duration_seconds` and `drift_rate_hz_per_second` are only in real seconds for raw-IQ input; a bare spectrogram-matrix input (`.txt`/`.csv`) has no real time axis, and `FeatureVector.time_axis_is_synthetic` flags this.
- Cross-validation and train/test splitting are statistically unreliable on very small datasets; `src/ml/train.py` degrades gracefully (skips/falls back with a stated reason) rather than reporting misleadingly precise numbers.
- Only a binary label (satellite candidate / not) is supported - no per-satellite classification yet.
- `frequency_drift_hz`/`drift_rate_hz_per_second` are magnitude-only (`abs(end - start)`); direction (upward vs. downward chirp) isn't captured in the current feature schema. Signed variants (`signed_frequency_drift_hz`, `signed_drift_rate_hz_per_second`) were deliberately **not** added alongside the dataset-building work in this change - doing so cleanly would mean extending `FeatureVector`/`FEATURE_NAMES` (touching the rule detector, the ML feature schema, `dataset_io.CSV_COLUMNS`, and every existing test that enumerates features) as a second, separable change, not a quick addition to a dataset-generation script. Left as a follow-up.
