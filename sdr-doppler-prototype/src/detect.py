from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from spectrogram import SpectrogramData


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


def extract_strongest_trace(
    spec: SpectrogramData,
    snr_threshold_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    power = spec.power_db
    strongest_bins = np.argmax(power, axis=1)
    strongest_power = power[np.arange(power.shape[0]), strongest_bins]

    # Estimate the local noise floor for each time slice from the median bin.
    noise_floor = np.median(power, axis=1)
    valid_mask = strongest_power >= (noise_floor + snr_threshold_db)
    trace_hz = spec.frequencies_hz[strongest_bins].astype(float)
    trace_hz[~valid_mask] = np.nan
    return trace_hz, valid_mask


def smooth_trace(trace_hz: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    finite = np.isfinite(trace_hz)
    if not finite.any():
        return trace_hz.copy()

    x = np.arange(trace_hz.size)
    filled = np.interp(x, x[finite], trace_hz[finite])
    smoothed = median_filter(filled, size=kernel_size, mode="nearest")
    smoothed[~finite] = np.nan
    return smoothed


def calculate_smoothness(smoothed_trace_hz: np.ndarray) -> float:
    finite = np.isfinite(smoothed_trace_hz)
    if finite.sum() < 3:
        return float("inf")
    diffs = np.diff(smoothed_trace_hz[finite])
    return float(np.std(diffs))


def detect_candidate(
    spec: SpectrogramData,
    min_valid_ratio: float,
    min_drift_hz: float,
    max_smoothness_hz: float,
    snr_threshold_db: float,
) -> DetectionResult:
    trace_hz, valid_mask = extract_strongest_trace(spec, snr_threshold_db)
    smoothed = smooth_trace(trace_hz)
    valid_ratio = float(np.mean(valid_mask)) if valid_mask.size else 0.0

    finite = np.isfinite(smoothed)
    if finite.sum() >= 2:
        drift = float(abs(smoothed[finite][-1] - smoothed[finite][0]))
    else:
        drift = 0.0

    smoothness = calculate_smoothness(smoothed)
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
        strongest_trace_hz=trace_hz,
        smoothed_trace_hz=smoothed,
        valid_mask=valid_mask,
        notes="; ".join(notes),
    )
