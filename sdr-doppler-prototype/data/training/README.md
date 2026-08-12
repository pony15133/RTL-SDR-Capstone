# Training Data

This directory holds the labelled CSV dataset(s) used to train the Random
Forest satellite-candidate classifier (`src/ml/train.py`).

## CSV Format

One row per capture. Required columns:

| Column | Type | Meaning |
|---|---|---|
| `capture_id` | string, unique | Human-readable identifier for the capture (e.g. filename stem) |
| `snr_db` | float | Estimated SNR (see `features/extractor.py` for method) |
| `frequency_drift_hz` | float | Total frequency drift across the capture |
| `drift_rate_hz_per_second` | float | `frequency_drift_hz / signal_duration_seconds` |
| `smoothness_score` | float | Std-dev of frequency jumps between time slices (lower = smoother) |
| `valid_signal_ratio` | float, 0-1 | Fraction of time slices with a signal above the noise floor |
| `peak_power` | float (dB) | Maximum power anywhere in the spectrogram |
| `mean_power` | float (dB) | Mean power across the whole spectrogram |
| `occupied_bandwidth_hz` | float | Threshold-crossing bandwidth estimate (not a formal 99% OBW) |
| `signal_duration_seconds` | float | Time span of valid signal (real seconds for raw-IQ input; see caveat below) |
| `label` | int, 0 or 1 | **1 = satellite signal candidate, 0 = non-satellite/noise/interference** |

These are exactly the columns `features.extractor.FEATURE_NAMES` produces,
in the same order the model is trained/inferred on - `src/ml/train.py`
validates the file has all of them before doing anything else.

Optional columns (not fed to the model, but kept for traceability/reproducibility):

| Column | Meaning |
|---|---|
| `is_synthetic` | `1` if this row is synthetic/fabricated data, `0`/absent otherwise. **See "Synthetic vs. real data" below - this is load-bearing, not decorative.** |
| `source_file` | Path to the original `.iq`/spectrogram capture this row came from |
| `sample_rate_hz`, `nperseg`, `noverlap` | Spectrogram parameters used to derive the features, so results stay comparable/reproducible as the dataset grows |
| `notes` | Free-text, e.g. why a row was labelled the way it was |

Generate a row's feature values with the pipeline itself (`src/main.py` or
`scripts/label_capture.py`, not by hand) so they're computed exactly the
same way training and inference will compute them.

## Synthetic vs. Real Data

**`synthetic_example.csv` in this directory is entirely synthetic** -
generated with a fixed random seed to exercise the training pipeline
(dataset validation, train/test split, `RandomForestClassifier` training,
evaluation, feature importance, model save/load). It is marked
`is_synthetic=1` on every row.

`src/ml/train.py` **refuses to train** on a dataset containing any
`is_synthetic=1` rows unless you pass `--allow-synthetic` explicitly, and
any model trained that way is permanently marked
`trained_on_synthetic_data: true` in its metadata sidecar.

**Do not** add synthetic rows to a real dataset file, and do not present a
model trained (even partly) on synthetic data as validated. Per the
project's requirements, until a trained model has real, independently
verified labelled captures behind it, it must be considered:

> **IMPLEMENTED BUT NOT VALIDATED**

## Collecting Real Labelled Data

1. Use the RTL-SDR recorder (`iq-recorder/`) to capture real signals:
   - **Positive candidates**: recordings made during a known satellite
     pass (predictable AOS/LOS from a pass-prediction tool such as
     Gpredict, ahead of the automated SatNOGS integration planned for
     later).
   - **Negative examples**: recordings made outside any known pass window
     at the same frequency, deliberately mistuned captures, or known
     local interference.
2. Run `src/main.py --input <capture> --output data/results/ --save-image`
   on each capture. This produces the spectrogram image, the extracted
   features, and a JSON summary - without yet needing a trained model
   (`ml_status` will just read `MODEL_NOT_AVAILABLE` until one exists).
3. Inspect the spectrogram PNG (and optionally the recorder's suggested
   IQ playback/verification tools) to decide the label.
4. Append a row to a real dataset CSV (e.g. `data/training/features.csv`,
   which does not exist until you start collecting real data) with the
   extracted feature values and your label. `scripts/label_capture.py`
   automates steps 2-4 into one command with an interactive label prompt.
5. Repeat until there's enough real, labelled data (see the main README's
   "Known limitations" for rough guidance on minimum dataset size) to
   train and meaningfully evaluate a model:

   ```bash
   python train_model.py --dataset data/training/features.csv --output models/random_forest.joblib
   ```

## Known Limitation: `signal_duration_seconds` / `drift_rate_hz_per_second`

These two features are only in real seconds when the underlying capture
was raw IQ (processed through `iq_to_spectrogram()`, which has a genuine
time axis from `scipy.signal.spectrogram`). A bare 2D spectrogram matrix
loaded from `.txt`/`.csv` (`matrix_to_spectrogram()`) has no real time
axis and gets a synthetic, index-based one - `signal_duration_seconds`
and `drift_rate_hz_per_second` would then be in "time-bin units", not
seconds. `extract_features()` records this on `FeatureVector.time_axis_is_synthetic`;
avoid mixing rows derived from matrix-only input into a dataset trained
primarily on raw-IQ-derived rows without accounting for this.
