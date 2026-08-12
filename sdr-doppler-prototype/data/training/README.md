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

### Batch-building from a folder of raw captures (`.bin`/`.iq`/`.dat`/`.npy`)

`scripts/label_capture.py` above is one file, one command, with an
interactive prompt - fine for a handful of captures, tedious for a
folder of them. `scripts/build_training_dataset.py` does the same feature
extraction and CSV-append in bulk, over every raw capture file in a
directory. Reading the file and extracting features is fully automatic
(the same `extract_features()` used everywhere else); the label still has
to come from somewhere, since this is supervised learning - the script
supports three ways to supply it, checked in this order per file:

1. **A manifest CSV** (`--labels-csv`) with columns `filename,label`,
   matched against each file's path relative to `--input-dir` or its bare
   basename:

   ```bash
   python scripts/build_training_dataset.py \
       --input-dir /path/to/captures \
       --labels-csv data/training/my_labels.csv \
       --dataset data/training/features.csv \
       --sample-rate 2400000 --center-freq 137900000 --binary-dtype complex64
   ```

2. **Folder-name convention** - no manifest needed if your captures are
   already sorted into subfolders named `1`/`positive`/`pos`/`candidate`
   and `0`/`negative`/`neg`/`noise`:

   ```
   captures/
     positive/pass_001.bin
     positive/pass_002.bin
     negative/noise_001.bin
   ```

   ```bash
   python scripts/build_training_dataset.py --input-dir captures --recursive \
       --dataset data/training/features.csv --sample-rate 2400000 --center-freq 137900000
   ```

3. **A single `--label`** applied to every file in the run - useful when
   an entire folder is confirmed one way (e.g. a folder of known
   background-noise recordings):

   ```bash
   python scripts/build_training_dataset.py --input-dir captures/known_noise \
       --label 0 --dataset data/training/features.csv
   ```

If none of these resolve a label for a given file, you're prompted
interactively (same as `label_capture.py`) unless `--skip-unlabeled` is
passed, in which case that file is skipped rather than blocking the batch.
Files already present in the dataset (matched by their source path) are
skipped automatically on a re-run, so it's safe to point this at a
growing folder repeatedly; pass `--reprocess` to force re-adding them
anyway. A capture that fails to load (corrupt/truncated file, wrong
`--binary-dtype`) is logged and skipped - it never aborts the rest of the
batch. Rows from this tool are always `is_synthetic=0`.

### Windowed/chunked captures (sparse, low-duty-cycle signals)

`label_capture.py` and `build_training_dataset.py` both label a whole
file as one row. That's wrong for a capture where a real signal occupies
only a small fraction of a much longer file - e.g. the "LoRadar"
satellite-LoRa dataset: 4 MHz, complex64, only ~3% packet duty cycle per
session. Averaging a few real packets
across minutes of silence into one feature row washes the packets out
into statistical noise. `scripts/chunk_bin_to_dataset.py` instead slices
one large raw-IQ file into fixed-length, overlapping windows and scores
each window individually with the same rule detector the rest of the
pipeline uses:

```bash
python scripts/chunk_bin_to_dataset.py \
    --input /path/to/session_001.bin \
    --dataset data/training/features.csv \
    --sample-rate 4000000 --center-freq 401300000 --binary-dtype complex64 \
    --window-seconds 2.0 --stride-seconds 1.0 \
    --snr-threshold-db 15 --min-valid-ratio 0.2 \
    --save-candidate-images --output data/results/lora_review
```

Each window the rule detector rejects is auto-labelled 0 in bulk (tagged
`auto-negative` in `notes`); each window it flags is a candidate you
confirm interactively (or accept automatically with
`--auto-accept-candidates`, tagged `auto-accepted candidate, not manually
reviewed` so a weak label is never silently indistinguishable from a
reviewed one). Windows are deduplicated by exact sample range, so
re-running against the same file is safe.

**Tune `--snr-threshold-db` and `--min-valid-ratio` before trusting the
defaults on real data.** The pipeline's satellite-pass defaults
(`--snr-threshold-db 6`, `--min-valid-ratio 0.45`) were validated against
a signal spanning nearly an entire capture, not a short burst inside a
much longer window. Verified empirically on a synthetic burst-in-noise
window: at the default 6 dB threshold and ~1000 frequency bins, pure
noise alone crosses the strongest-bin-vs-median-power check on almost
every time slice (extreme-value statistics over that many bins), so
`valid_signal_ratio` saturates near 1.0 for noise and real signal alike -
useless as a discriminator - and a real burst's clean trace gets diluted
by the noise-dominated rest of the window, failing `max_smoothness_hz`
too. Raising `--snr-threshold-db` to ~15 dB fixed the false-"valid" rate
under noise in that test, and lowering `--min-valid-ratio` to ~0.2
correctly let a burst occupying under a third of the window still
register. Treat those as a validated *starting point*, not a tuned
result on your actual captures - confirm against a couple of
known/expected-positive windows (`--save-candidate-images` helps here)
before running `--auto-accept-candidates` across a whole file.

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
