import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from detect import DetectionResult


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def save_summary(
    output_dir: Path,
    input_path: Path,
    timestamp_utc: str,
    detection: DetectionResult,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{safe_stem(input_path)}_{timestamp_utc.replace(':', '')}_summary.json"
    payload = {
        "input_file": str(input_path),
        "timestamp_utc": timestamp_utc,
        "detection_result": detection.detected,
        "confidence_score": detection.confidence_score,
        "valid_signal_ratio": detection.valid_signal_ratio,
        "frequency_drift_hz": detection.frequency_drift_hz,
        "smoothness_score": detection.smoothness_score if np.isfinite(detection.smoothness_score) else None,
        "notes": detection.notes,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path
