import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from detect import DetectionResult
from detection.ml_detector import MLDetectionResult


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


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
