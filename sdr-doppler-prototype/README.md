# SDR Doppler Prototype

Backend-only Python prototype for detecting candidate satellite Doppler signals from RTL-SDR IQ captures or precomputed spectrogram matrices.

The code is intentionally small and readable for a university capstone prototype. It does not include a GUI, ML model, orbital mechanics, cloud deployment, or production capture orchestration.

## What Is Included

- Loads sample input data from a file.
- Converts raw IQ samples to a spectrogram with `scipy.signal.spectrogram`.
- Loads existing spectrogram matrices from `.txt`, `.csv`, or 2D `.npy` files.
- Extracts the strongest frequency bin for each time slice.
- Applies median filtering to smooth the frequency trace.
- Uses simple rules for candidate detection:
  - enough valid signal points,
  - enough frequency drift,
  - smoother than random noise.
- Saves a JSON result summary.
- Optionally saves a spectrogram PNG.
- Stores one row per capture in SQLite.
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
