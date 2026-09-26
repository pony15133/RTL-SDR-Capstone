"""A running record of every training run and external test.

Each `train_model.py` run and each `scripts/evaluate_model.py` run appends one
row to ``ml_history.csv`` next to the model file (normally ``models/``). The
file is never rewritten, so it is the project's lab notebook: which data, how
much of it, which numbers - in order. ``scripts/plot_ml_history.py`` turns it
into presentation charts.

Recording must never break training: any error here is printed and ignored.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

HISTORY_NAME = "ml_history.csv"
COLUMNS = [
    "timestamp_utc", "kind", "model_version", "feature_set", "model_file", "dataset",
    "n_rows", "n_signal", "n_noise", "n_train", "n_val", "n_test", "threshold",
    "accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1_mean", "cv_f1_std",
    "tn", "fp", "fn", "tp", "git_commit", "notes",
]


def history_path_for(model_path: Path) -> Path:
    return Path(model_path).parent / HISTORY_NAME


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _names(dataset) -> str:
    items = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    return " + ".join(Path(str(p)).name for p in items)


def _round(v) -> Optional[float]:
    return None if v is None else round(float(v), 4)


def _row(kind, *, metrics: dict, model_path, dataset, model_version=None, feature_set=None, threshold=None,
         label_distribution=None, n_train=None, n_val=None, n_test=None, cv=None, notes=None) -> dict:
    cm = metrics.get("confusion_matrix") or [[None, None], [None, None]]
    labels = label_distribution or {}
    n_signal, n_noise = labels.get("1"), labels.get("0")
    n_rows = (n_signal or 0) + (n_noise or 0) if labels else metrics.get("n_samples")
    cv = cv or {}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind, "model_version": model_version or "", "feature_set": feature_set or "",
        "model_file": Path(model_path).name, "dataset": _names(dataset),
        "n_rows": n_rows, "n_signal": n_signal, "n_noise": n_noise,
        "n_train": n_train, "n_val": n_val, "n_test": n_test, "threshold": _round(threshold),
        "accuracy": _round(metrics.get("accuracy")), "precision": _round(metrics.get("precision")),
        "recall": _round(metrics.get("recall")), "f1": _round(metrics.get("f1_score")),
        "roc_auc": _round(metrics.get("roc_auc")),
        "cv_f1_mean": _round(cv.get("f1_mean")) if cv.get("performed") else None,
        "cv_f1_std": _round(cv.get("f1_std")) if cv.get("performed") else None,
        "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
        "git_commit": _git_commit(), "notes": notes or "",
    }


def append(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def record_training(result, metadata: dict, dataset, notes=None) -> Optional[Path]:
    """One 'train' row (held-out test metrics) for a finished training run."""
    try:
        path = history_path_for(result.model_path)
        append(path, _row(
            "train", metrics=result.metrics, model_path=result.model_path, dataset=dataset,
            model_version=metadata.get("model_version"), feature_set=metadata.get("feature_set"),
            threshold=result.threshold, label_distribution=metadata.get("label_distribution"),
            n_train=result.n_train, n_val=result.n_val, n_test=result.n_test, cv=result.cv_result,
            notes=notes))
        return path
    except Exception as exc:  # noqa: BLE001 - never break training over the log
        print(f"(could not update training history: {exc})", file=sys.stderr)
        return None


def record_external(model_path, dataset, metrics: dict, model_version=None, feature_set=None,
                    notes=None) -> Optional[Path]:
    """One 'external_test' row: a model scored on data it never trained on."""
    try:
        path = history_path_for(model_path)
        append(path, _row("external_test", metrics=metrics, model_path=model_path, dataset=dataset,
                          model_version=model_version, feature_set=feature_set,
                          threshold=metrics.get("threshold"), notes=notes))
        return path
    except Exception as exc:  # noqa: BLE001
        print(f"(could not update training history: {exc})", file=sys.stderr)
        return None
