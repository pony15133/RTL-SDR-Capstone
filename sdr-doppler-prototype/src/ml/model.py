"""Persistence for trained ML models: a classifier plus its metadata,
saved/loaded as a matched pair so a model file is never used without
knowing what it is.

Layout on disk, e.g. for ``models/random_forest_v1.joblib``:

- ``random_forest_v1.joblib``: the fitted scikit-learn estimator (joblib).
- ``random_forest_v1.json``: a metadata sidecar (model type/version,
  training timestamp, feature schema, RF params, dataset size/label
  distribution, evaluation metrics, whether training data was synthetic).

This mirrors the IQ recorder's IQ-file + JSON-sidecar convention
elsewhere in this repo, so a model is never distributed as a bare
``.joblib`` with no provenance.
"""

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import sklearn


class ModelLoadError(Exception):
    """A model file and/or its metadata sidecar could not be loaded or is invalid.

    Callers that must keep running without ML (see detection/ml_detector.py)
    should catch this rather than let it crash the pipeline.
    """


@dataclass
class ModelBundle:
    """A fitted classifier plus everything needed to use and audit it."""

    classifier: Any
    feature_names: tuple
    metadata: dict = field(default_factory=dict)

    def save(self, model_path: Path) -> Path:
        """Save the classifier (joblib) and a metadata JSON sidecar next to it."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, model_path)

        metadata_path = self.metadata_path_for(model_path)
        payload = dict(self.metadata)
        payload["feature_names"] = list(self.feature_names)
        payload.setdefault("sklearn_version", sklearn.__version__)
        payload.setdefault("python_version", platform.python_version())
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return model_path

    @classmethod
    def load(cls, model_path: Path) -> "ModelBundle":
        """Load a classifier + its metadata sidecar. Raises ModelLoadError on any problem.

        Deliberately strict: a missing file, missing/corrupt sidecar, or a
        feature-schema that doesn't look like a list of names all raise
        ModelLoadError rather than returning something partially usable.
        `detection.ml_detector.try_load_model()` is the layer that turns
        this into a graceful "no model available" for the main pipeline.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")

        metadata_path = cls.metadata_path_for(model_path)
        if not metadata_path.exists():
            raise ModelLoadError(
                f"Model metadata sidecar not found: {metadata_path} "
                "(a model file without its metadata sidecar cannot be used safely - "
                "the feature schema it expects is unknown)"
            )

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError(f"Could not read model metadata {metadata_path}: {exc}") from exc

        feature_names = metadata.get("feature_names")
        if not isinstance(feature_names, list) or not all(isinstance(n, str) for n in feature_names) or not feature_names:
            raise ModelLoadError(
                f"Model metadata {metadata_path} has no valid 'feature_names' list - "
                "cannot safely build a feature array for this model"
            )

        try:
            classifier = joblib.load(model_path)
        except Exception as exc:  # joblib/pickle can raise many exception types
            raise ModelLoadError(f"Could not load model file {model_path}: {exc}") from exc

        n_features_in = getattr(classifier, "n_features_in_", None)
        if n_features_in is not None and n_features_in != len(feature_names):
            raise ModelLoadError(
                f"Model {model_path} expects {n_features_in} features but its metadata "
                f"lists {len(feature_names)} feature names - metadata/model mismatch"
            )

        return cls(classifier=classifier, feature_names=tuple(feature_names), metadata=metadata)

    @staticmethod
    def metadata_path_for(model_path: Path) -> Path:
        return Path(model_path).with_suffix(".json")

    @property
    def model_version(self) -> Optional[str]:
        return self.metadata.get("model_version")
