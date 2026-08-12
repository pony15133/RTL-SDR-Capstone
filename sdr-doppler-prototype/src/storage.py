import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from detect import DetectionResult
from detection.ml_detector import MLDetectionResult


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def session_dir_name(prefix: str = "session") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{ts}"


def ensure_session_output_dir(base_output_dir: Path, *, prefix: str = "session") -> Path:
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    session_dir = base_output_dir / session_dir_name(prefix=prefix)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _human_float(value, *, digits: int = 2, default: str = "N/A") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return default
    return f"{float(value):.{digits}f}"


def format_detection_summary(payload: dict) -> str:
    if not payload:
        return "No result available."

    rule_detected = bool(payload.get("rule_detection_result"))
    ml_detected = payload.get("ml_detection_result")
    drift_hz = payload.get("frequency_drift_hz")
    confidence = payload.get("rule_confidence_score")
    ml_confidence = payload.get("ml_confidence_score")
    valid_ratio = payload.get("valid_signal_ratio")
    smoothness = payload.get("smoothness_score")
    notes = payload.get("notes") or "No notes recorded."

    rule_label = "SATELLITE CANDIDATE" if rule_detected else "NO SATELLITE CANDIDATE"
    ml_label = "N/A"
    if ml_detected is not None:
        ml_label = "SATELLITE CANDIDATE" if bool(ml_detected) else "NOT A SATELLITE CANDIDATE"

    lines = [
        "Detection result summary",
        "========================",
        f"Status: {rule_label}",
        f"Estimated drift: { _human_float(drift_hz, digits=1, default='N/A') } Hz",
        f"Rule confidence: { _human_float(confidence * 100 if confidence is not None else None, digits=1, default='N/A') }%",
        f"Valid signal ratio: { _human_float(valid_ratio, digits=3, default='N/A') }",
        f"Smoothness score: { _human_float(smoothness, digits=2, default='N/A') }",
        f"ML result: {ml_label}",
        f"ML confidence: { _human_float(ml_confidence * 100 if ml_confidence is not None else None, digits=1, default='N/A') }%",
        f"Notes: {notes}",
    ]
    return "\n".join(lines)


def format_training_summary(payload: dict) -> str:
    if not payload:
        return "No training result available."

    metrics = payload.get("evaluation_metrics") or {}
    confusion = metrics.get("confusion_matrix") or [[0, 0], [0, 0]]
    cv = payload.get("cross_validation") or {}
    feature_importance = payload.get("feature_importance") or []
    model_version = payload.get("model_version") or "unknown"

    lines = [
        "Model training summary",
        "======================",
        f"Model: {model_version}",
        f"Training samples: {payload.get('n_training_samples', 'N/A')}",
        f"Test samples: {payload.get('n_test_samples', 'N/A')}",
        f"Accuracy: { _human_float(metrics.get('accuracy'), digits=3, default='N/A') }",
        f"Precision: { _human_float(metrics.get('precision'), digits=3, default='N/A') }",
        f"Recall: { _human_float(metrics.get('recall'), digits=3, default='N/A') }",
        f"F1 score: { _human_float(metrics.get('f1_score'), digits=3, default='N/A') }",
        f"ROC AUC: { _human_float(metrics.get('roc_auc'), digits=3, default='N/A') }",
    ]

    if cv.get("performed"):
        lines.append(f"Cross-validation F1: {cv.get('f1_mean', 'N/A'):.3f} +/- {cv.get('f1_std', 'N/A'):.3f}")
    elif cv.get("reason"):
        lines.append(f"Cross-validation: {cv.get('reason')}")

    tn, fp = confusion[0]
    fn, tp = confusion[1]
    lines.extend([
        "",
        "Confusion matrix",
        "----------------",
        f"Actual 0 -> predicted 0: {tn}, predicted 1: {fp}",
        f"Actual 1 -> predicted 0: {fn}, predicted 1: {tp}",
    ])

    if feature_importance:
        lines.extend(["", "Top feature importances:"])
        for entry in feature_importance[:5]:
            name = entry.get("feature", "unknown")
            importance = entry.get("importance", 0.0)
            lines.append(f"- {name}: { _human_float(importance, digits=4, default='N/A') }")

    return "\n".join(lines)


def save_summary(
    output_dir: Path,
    input_path: Path,
    timestamp_utc: str,
    detection: DetectionResult,
    ml_detection: Optional[MLDetectionResult] = None,
) -> Path:
    """Write the per-capture JSON summary.

    Always includes the rule-based (baseline) result. Includes the ML
    result too when a prediction was made (``ml_detection`` is not None);
    when no ML model was available, the summary still records that
    explicitly (`ml_status`) rather than omitting the field.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{safe_stem(input_path)}_{timestamp_utc.replace(':', '')}_summary.json"
    payload = {
        "input_file": str(input_path),
        "timestamp_utc": timestamp_utc,
        "rule_detection_result": detection.detected,
        "rule_confidence_score": detection.confidence_score,
        "valid_signal_ratio": detection.valid_signal_ratio,
        "frequency_drift_hz": detection.frequency_drift_hz,
        "smoothness_score": detection.smoothness_score if np.isfinite(detection.smoothness_score) else None,
        "notes": detection.notes,
        "ml_status": ml_detection.status if ml_detection else "MODEL_NOT_AVAILABLE",
        "ml_detection_result": ml_detection.ml_detection_result if ml_detection else None,
        "ml_confidence_score": ml_detection.ml_confidence_score if ml_detection else None,
        "model_version": ml_detection.model_version if ml_detection else None,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path
