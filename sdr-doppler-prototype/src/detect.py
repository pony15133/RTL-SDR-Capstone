from dataclasses import dataclass

import numpy as np

from features.extractor import (
    FeatureVector,
    calculate_smoothness,
    extract_features,
    extract_strongest_trace,
    smooth_trace,
)
from spectrogram import SpectrogramData

# Re-exported for backward compatibility: this module used to define
# extract_strongest_trace/smooth_trace/calculate_smoothness itself. They now
# live in features.extractor (shared with the ML detector), and are
# imported back here unchanged so any existing `from detect import ...`
# usage keeps working.
__all__ = [
    "DetectionResult",
    "extract_strongest_trace",
    "smooth_trace",
    "calculate_smoothness",
    "detect_candidate",
]


@dataclass
class DetectionResult:
    detected: bool
    confidence_score: float
    valid_signal_ratio: float
    frequency_drift_hz: float
    smoothness_score: float
    strongest_trace_hz: np.ndarray
    smoothed_trace_hz: np.ndarray
    valid_mask: np.ndarray
    notes: str


def detect_candidate(
    spec: SpectrogramData,
    min_valid_ratio: float,
    min_drift_hz: float,
    max_smoothness_hz: float,
    snr_threshold_db: float,
    features: FeatureVector = None,
) -> DetectionResult:
    """The rule-based (threshold) detector - the project's baseline.

    This is deliberately unchanged in behaviour from the original
    implementation: same thresholds, same confidence heuristic, same
    output. The only refactor is that the underlying trace/valid-ratio/
    drift/smoothness numbers now come from `features.extractor` instead
    of being computed twice (once here, once for the ML detector).

    Pass `features` (e.g. one already computed for the ML detector) to
    avoid recomputing it; otherwise it's computed internally.
    """
    if features is None:
        features = extract_features(spec, snr_threshold_db)

    valid_ratio = features.valid_signal_ratio
    drift = features.frequency_drift_hz
    smoothness = features.smoothness_score

    enough_signal = valid_ratio >= min_valid_ratio
    enough_drift = drift >= min_drift_hz
    smooth_enough = smoothness <= max_smoothness_hz
    detected = enough_signal and enough_drift and smooth_enough

    valid_component = min(valid_ratio / max(min_valid_ratio, 1e-9), 1.0)
    drift_component = min(drift / max(min_drift_hz, 1e-9), 1.0)
    smooth_component = 0.0 if not np.isfinite(smoothness) else max(0.0, 1.0 - smoothness / max(max_smoothness_hz, 1e-9))
    confidence = float(np.mean([valid_component, drift_component, smooth_component]))

    notes = []
    if not enough_signal:
        notes.append("not enough valid signal points")
    if not enough_drift:
        notes.append("frequency drift below threshold")
    if not smooth_enough:
        notes.append("trace too noisy")
    if detected:
        notes.append("candidate satellite Doppler-like drift")

    return DetectionResult(
        detected=detected,
        confidence_score=confidence,
        valid_signal_ratio=valid_ratio,
        frequency_drift_hz=drift,
        smoothness_score=smoothness,
        strongest_trace_hz=features.strongest_trace_hz,
        smoothed_trace_hz=features.smoothed_trace_hz,
        valid_mask=features.valid_mask,
        notes="; ".join(notes),
    )
