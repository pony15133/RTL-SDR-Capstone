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
