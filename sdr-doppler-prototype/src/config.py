from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPECTROGRAM_DIR = DATA_DIR / "spectrograms"
RESULTS_DIR = DATA_DIR / "results"
DB_PATH = RESULTS_DIR / "captures.sqlite3"
TRAINING_DATA_DIR = DATA_DIR / "training"

# ML component. The default model path may simply not exist yet (no
# trained model) - main.py's pipeline must keep working in that case; see
# detection/ml_detector.py MODEL_NOT_AVAILABLE.
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_ML_MODEL_PATH = MODELS_DIR / "random_forest.joblib"

DEFAULT_SAMPLE_RATE_HZ = 240_000.0
DEFAULT_CENTER_FREQ_HZ = 136_800_000.0
DEFAULT_NPERSEG = 1024
DEFAULT_NOVERLAP = 512

# Rule-based detector defaults. These are intentionally conservative and easy
# to tune from the command line for student experiments.
DEFAULT_MIN_VALID_RATIO = 0.45
DEFAULT_MIN_DRIFT_HZ = 1_500.0
DEFAULT_MAX_SMOOTHNESS_HZ = 8_000.0
DEFAULT_SNR_THRESHOLD_DB = 6.0
