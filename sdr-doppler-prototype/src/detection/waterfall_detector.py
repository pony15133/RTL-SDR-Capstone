"""Inference for the waterfall model (trained on SatNOGS observations).

Same contract as detection/ml_detector.py: never raises; a missing or
incompatible model gives MODEL_NOT_AVAILABLE / MODEL_INVALID.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import MODELS_DIR
from detection.ml_detector import AVAILABLE, MODEL_INVALID, MODEL_NOT_AVAILABLE, MLDetectionResult, try_load_model
from features.waterfall_features import WATERFALL_FEATURE_NAMES, waterfall_features

logger = logging.getLogger(__name__)

DEFAULT_WATERFALL_MODEL_PATH = MODELS_DIR / "waterfall_rf.joblib"


def predict_waterfall(model_path, matrix: np.ndarray) -> MLDetectionResult:
    bundle = try_load_model(Path(model_path) if model_path else DEFAULT_WATERFALL_MODEL_PATH)
    if bundle is None:
        return MLDetectionResult(MODEL_NOT_AVAILABLE, None, None, None, "No waterfall model found")
    missing = [n for n in bundle.feature_names if n not in WATERFALL_FEATURE_NAMES]
    if missing:
        return MLDetectionResult(MODEL_INVALID, None, None, bundle.model_version,
                                 f"Model expects non-waterfall features {missing}")
    try:
        feats = waterfall_features(matrix)
        row = np.array([[feats[n] for n in bundle.feature_names]], dtype=float)
        clf = bundle.classifier
        classes = list(clf.classes_)
        confidence = float(clf.predict_proba(row)[0][classes.index(1)]) if 1 in classes else 0.0
        return MLDetectionResult(AVAILABLE, bool(clf.predict(row)[0]), confidence, bundle.model_version)
    except Exception as exc:  # never take the pipeline down
        logger.warning("Waterfall inference failed: %s", exc)
        return MLDetectionResult(MODEL_INVALID, None, None, bundle.model_version, f"Inference failed: {exc}")
