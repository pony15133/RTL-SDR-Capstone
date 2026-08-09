"""Shared feature extraction for the rule-based and ML detectors.

Both `detect.detect_candidate()` (the rule-based baseline) and
`detection.ml_detector.predict()` (the Random Forest classifier) consume
the *same* `FeatureVector`, computed once by `extract_features()`. This is
the single place the underlying signal-characteristic math lives, so the
two detectors are guaranteed to see identical numbers and neither
duplicates the other's logic.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import median_filter

from spectrogram import SpectrogramData

#: Canonical, fixed order of numeric ML feature names. Used everywhere a
#: feature vector is turned into a plain numeric array (CSV export, model
#: training, and inference) so a mismatch is always caught explicitly
#: rather than silently misaligning columns.
FEATURE_NAMES: tuple[str, ...] = (
    "snr_db",
    "frequency_drift_hz",
    "drift_rate_hz_per_second",
    "smoothness_score",
    "valid_signal_ratio",
    "peak_power",
    "mean_power",
    "occupied_bandwidth_hz",
    "signal_duration_seconds",
)

#: smoothness_score is the std-dev of frequency jumps (Hz) between
#: adjacent time slices; it is mathematically `inf` when fewer than 3
#: finite trace points exist (see calculate_smoothness). RandomForest
#: cannot consume inf/NaN, so ML-facing code (feature_vector_to_array)
#: replaces it with this large-but-finite sentinel ("obviously not a
#: smooth trace"). The raw FeatureVector - used by the rule detector and
#: for human-readable reporting - always keeps the true `inf` value.
SMOOTHNESS_INF_SENTINEL_HZ = 1_000_000.0


class FeatureSchemaError(ValueError):
    """A requested feature name isn't produced by FeatureVector.

    Raised when converting a FeatureVector to a numeric array against a
    `feature_names` list that doesn't match what this version of
    extract_features() actually computes - e.g. an older/newer saved
    model's schema. Callers should treat this as "model incompatible",
    not let it silently misalign a feature array.
    """


@dataclass
class FeatureVector:
    """One capture's worth of extracted features.

    The first nine fields are the fixed ML feature set (order defined by
    FEATURE_NAMES). The remaining fields are supporting per-time-slice
    arrays and provenance used by the rule detector and by reporting/
    visualisation code - they are intentionally not part of the ML
    feature set.
    """

    # --- ML feature set (order defined by FEATURE_NAMES) ---
    snr_db: float
    frequency_drift_hz: float
    drift_rate_hz_per_second: float
    smoothness_score: float
    valid_signal_ratio: float
    peak_power: float
    mean_power: float
    occupied_bandwidth_hz: float
    signal_duration_seconds: float

    # --- supporting arrays / provenance, not fed to the ML model ---
    strongest_trace_hz: np.ndarray
    smoothed_trace_hz: np.ndarray
    valid_mask: np.ndarray
    #: True when spec.times_s came from matrix_to_spectrogram() (a bare
    #: spectrogram matrix with a synthetic, index-based time axis) rather
    #: than iq_to_spectrogram() (a real time axis in seconds). When True,
    #: signal_duration_seconds and drift_rate_hz_per_second are in
    #: "time-bin units", not real seconds - see README "Known
    #: limitations".
    time_axis_is_synthetic: bool

    def as_dict(self) -> dict:
        """Only the numeric ML feature fields, as {name: value}."""
        return {name: getattr(self, name) for name in FEATURE_NAMES}


def _strongest_trace_stats(
    spec: SpectrogramData,
    snr_threshold_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-time-slice strongest-bin trace, plus the power/noise arrays behind it.

    Returns (trace_hz, valid_mask, strongest_power, noise_floor). Exposed
    separately from extract_strongest_trace() so extract_features() can
    reuse strongest_power/noise_floor for SNR and occupied-bandwidth
    estimates without recomputing the argmax/median pass twice.
    """
    power = spec.power_db
    strongest_bins = np.argmax(power, axis=1)
    strongest_power = power[np.arange(power.shape[0]), strongest_bins]

    # Estimate the local noise floor for each time slice from the median bin.
    noise_floor = np.median(power, axis=1)
    valid_mask = strongest_power >= (noise_floor + snr_threshold_db)
    trace_hz = spec.frequencies_hz[strongest_bins].astype(float)
    trace_hz[~valid_mask] = np.nan
    return trace_hz, valid_mask, strongest_power, noise_floor


def extract_strongest_trace(
    spec: SpectrogramData,
    snr_threshold_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Public, backward-compatible entry point used by the rule detector.

    Same signature and behaviour as before this module existed - moved
    here (not rewritten) as part of sharing feature-extraction code with
    the ML detector.
    """
    trace_hz, valid_mask, _strongest_power, _noise_floor = _strongest_trace_stats(spec, snr_threshold_db)
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


def _estimate_snr_db(strongest_power: np.ndarray, noise_floor: np.ndarray, valid_mask: np.ndarray) -> float:
    """Mean (peak - noise floor) over valid slices, falling back to all slices.

    This is an estimate derived from the same per-slice noise-floor model
    the rule detector already uses (median power across frequency bins),
    not a calibrated receiver SNR measurement.
    """
    if strongest_power.size == 0:
        return 0.0
    if valid_mask.any():
        return float(np.mean(strongest_power[valid_mask] - noise_floor[valid_mask]))
    return float(np.mean(strongest_power - noise_floor))


def _estimate_occupied_bandwidth_hz(
    spec: SpectrogramData,
    noise_floor: np.ndarray,
    valid_mask: np.ndarray,
    snr_threshold_db: float,
) -> float:
    """Median, across valid time slices, of the frequency span of bins that
    cross (noise_floor + snr_threshold_db) in that slice.

    This is a threshold-crossing bandwidth estimate, NOT a formal
    99%-power occupied bandwidth (which would need a calibrated power
    reference and an agreed energy-percentage definition this prototype
    doesn't have). Documented as an estimate rather than presented as a
    standard measurement.
    """
    if not valid_mask.any():
        return 0.0
    power = spec.power_db
    freqs = spec.frequencies_hz
    widths = []
    for t in np.nonzero(valid_mask)[0]:
        above = power[t, :] >= (noise_floor[t] + snr_threshold_db)
        if not above.any():
            continue
        bin_freqs = freqs[above]
        widths.append(float(bin_freqs.max() - bin_freqs.min()))
    if not widths:
        return 0.0
    return float(np.median(widths))


def _estimate_signal_duration_seconds(spec: SpectrogramData, valid_mask: np.ndarray) -> float:
    valid_indices = np.nonzero(valid_mask)[0]
    if valid_indices.size < 2:
        return 0.0
    return float(spec.times_s[valid_indices[-1]] - spec.times_s[valid_indices[0]])


def extract_features(
    spec: SpectrogramData,
    snr_threshold_db: float,
    time_axis_is_synthetic: bool = False,
) -> FeatureVector:
    """Compute the full feature set for one capture's spectrogram.

    ``time_axis_is_synthetic`` should be True when ``spec`` came from
    ``matrix_to_spectrogram()`` (a bare spectrogram matrix with no real
    time axis) rather than ``iq_to_spectrogram()``. It is recorded on the
    returned FeatureVector rather than silently trusting fabricated time
    units for duration/drift-rate.
    """
    trace_hz, valid_mask, strongest_power, noise_floor = _strongest_trace_stats(spec, snr_threshold_db)
    smoothed = smooth_trace(trace_hz)
    valid_ratio = float(np.mean(valid_mask)) if valid_mask.size else 0.0

    finite = np.isfinite(smoothed)
    if finite.sum() >= 2:
        drift = float(abs(smoothed[finite][-1] - smoothed[finite][0]))
    else:
        drift = 0.0

    smoothness = calculate_smoothness(smoothed)
    signal_duration = _estimate_signal_duration_seconds(spec, valid_mask)
    drift_rate = drift / signal_duration if signal_duration > 0 else 0.0

    return FeatureVector(
        snr_db=_estimate_snr_db(strongest_power, noise_floor, valid_mask),
        frequency_drift_hz=drift,
        drift_rate_hz_per_second=drift_rate,
        smoothness_score=smoothness,
        valid_signal_ratio=valid_ratio,
        peak_power=float(np.max(spec.power_db)) if spec.power_db.size else 0.0,
        mean_power=float(np.mean(spec.power_db)) if spec.power_db.size else 0.0,
        occupied_bandwidth_hz=_estimate_occupied_bandwidth_hz(spec, noise_floor, valid_mask, snr_threshold_db),
        signal_duration_seconds=signal_duration,
        strongest_trace_hz=trace_hz,
        smoothed_trace_hz=smoothed,
        valid_mask=valid_mask,
        time_axis_is_synthetic=time_axis_is_synthetic,
    )


def feature_vector_to_array(
    features: FeatureVector,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Convert a FeatureVector into a 1D numeric array in `feature_names` order.

    Used identically by CSV export (training) and inference, so both
    always sanitise/order features the same way. Non-finite values
    (currently only a possible `inf` smoothness_score) are replaced with
    +/-SMOOTHNESS_INF_SENTINEL_HZ so the array is always safe to hand to
    scikit-learn.

    Raises FeatureSchemaError if `feature_names` contains a name this
    FeatureVector doesn't have (e.g. a model trained against an older or
    newer feature schema) - callers should treat that as an incompatible
    model, not let it corrupt an inference.
    """
    values = []
    for name in feature_names:
        if not hasattr(features, name):
            raise FeatureSchemaError(
                f"Unknown feature '{name}' - not produced by the current extract_features(). "
                "This usually means a saved model's feature schema no longer matches this codebase."
            )
        value = float(getattr(features, name))
        if not np.isfinite(value):
            value = SMOOTHNESS_INF_SENTINEL_HZ if value > 0 else -SMOOTHNESS_INF_SENTINEL_HZ
        values.append(value)
    return np.array(values, dtype=float)
