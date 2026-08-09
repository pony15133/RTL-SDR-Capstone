"""ML inference: loads a trained Random Forest model (if one exists) and
runs it against a FeatureVector already computed by features.extractor -
the same feature vector the rule-based detector (detect.py) uses.

The main processing pipeline must keep working even when no trained model
exists yet, or an existing model file is corrupt/incompatible with the
current feature schema. This module never raises out of
try_load_model()/predict()/run_ml_detection() - callers always get a
clear, inspectable MLDetectionResult instead of an exception.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from features.extractor import FeatureSchemaError, FeatureVector, feature_vector_to_array
from ml.model import ModelBundle, ModelLoadError

logger = logging.getLogger(__name__)

AVAILABLE = "AVAILABLE"
MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
MODEL_INVALID = "MODEL_INVALID"


@dataclass
class MLDetectionResult:
    """The ML counterpart to detect.DetectionResult.

    ``ml_confidence_score`` is the trained model's estimated probability
    of the positive class (predict_proba) - it is the model's own
    confidence estimate, NOT a scientifically calibrated or guaranteed
    probability that a signal is actually a satellite pass.
    """

    status: str  # AVAILABLE / MODEL_NOT_AVAILABLE / MODEL_INVALID
    ml_detection_result: Optional[bool]
    ml_confidence_score: Optional[float]
    model_version: Optional[str]
    reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.status == AVAILABLE


def _unavailable(status: str, reason: str) -> MLDetectionResult:
    return MLDetectionResult(
        status=status, ml_detection_result=None, ml_confidence_score=None, model_version=None, reason=reason
    )


def try_load_model(model_path) -> Optional[ModelBundle]:
    """Load a model bundle, or return None (logging why) if unavailable.

    Never raises: a missing file, a missing/corrupt metadata sidecar, or a
    joblib load failure all just mean "no usable model" to the caller.
    """
    if model_path is None:
        logger.info("No ML model path configured - ML detector unavailable")
        return None
    model_path = Path(model_path)
    if not model_path.exists():
        logger.info("No trained ML model found at %s - ML detector unavailable (MODEL_NOT_AVAILABLE)", model_path)
        return None
    try:
        return ModelBundle.load(model_path)
    except ModelLoadError as exc:
        logger.warning("ML model at %s could not be loaded, treating as unavailable: %s", model_path, exc)
        return None


def predict(bundle: ModelBundle, features: FeatureVector) -> MLDetectionResult:
    """Run an already-loaded model against an already-computed FeatureVector."""
    try:
        array = feature_vector_to_array(features, feature_names=bundle.feature_names)
    except FeatureSchemaError as exc:
        logger.warning("ML model's feature schema is incompatible with the current extractor: %s", exc)
        return _unavailable(MODEL_INVALID, str(exc))

    try:
        classifier = bundle.classifier
        row = array.reshape(1, -1)
        prediction = classifier.predict(row)[0]
        proba = classifier.predict_proba(row)[0]
        classes = list(classifier.classes_)
        if 1 in classes:
            confidence = float(proba[classes.index(1)])
        else:
            # Degenerate case: the model was trained on a single class
            # (e.g. a tiny/unbalanced dataset), so there is no positive-
            # class probability to report.
            confidence = 0.0
    except Exception as exc:  # sklearn/numpy can raise a range of exception types on bad input
        logger.warning("ML inference failed: %s", exc)
        return _unavailable(MODEL_INVALID, f"Inference failed: {exc}")

    return MLDetectionResult(
        status=AVAILABLE,
        ml_detection_result=bool(prediction),
        ml_confidence_score=confidence,
        model_version=bundle.model_version,
    )


def run_ml_detection(model_path, features: FeatureVector) -> MLDetectionResult:
    """Convenience one-shot: load the model (if any) and predict.

    This is what main.py calls; it never raises, so the ML step can never
    take down the rule-detector-driven pipeline.
    """
    bundle = try_load_model(model_path)
    if bundle is None:
        return _unavailable(MODEL_NOT_AVAILABLE, "No usable trained model found")
    return predict(bundle, features)
