# Detection

This package holds the signal classification logic used after feature extraction.

## Purpose

- `ml_detector.py` loads a trained Random Forest model and predicts whether a capture resembles a satellite candidate.
- The rule-based detector remains in the parent `src/` folder and is kept separate for comparison and debugging.

## Typical Flow

1. Feature extraction creates a shared feature vector.
2. Detection logic evaluates the vector.
3. Results are reported back to `src/main.py` for output and database storage.
