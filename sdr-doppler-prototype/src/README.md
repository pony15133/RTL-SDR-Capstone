# Source Code

This folder contains the actual project logic for the SDR Doppler detection pipeline.

## Contents

- `config.py` – shared configuration constants.
- `database.py` – SQLite setup and result persistence.
- `detect.py` – rule-based detection logic.
- `load_data.py` – input loading for raw IQ and spectrogram matrices.
- `spectrogram.py` – spectrogram conversion and image saving.
- `storage.py` – result summary writing and output handling.
- `main.py` – high-level entry point for a full processing run.

## Subfolders

- `detection/` – machine-learning inference logic.
- `features/` – feature extraction and schema definitions.
- `ml/` – model training, evaluation, and metadata handling.

Use this folder when updating the underlying detection pipeline or feature logic.
