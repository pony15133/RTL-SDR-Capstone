"""Evaluation utilities for the Random Forest classifier: metrics,
cross-validation, confusion matrix formatting, and feature importance
reporting.

Nothing here claims a model "works" from accuracy alone - compute_metrics()
always returns the full precision/recall/F1/confusion-matrix set together,
and cross_validate() explicitly refuses (with a stated reason) to run on a
dataset too small for the result to mean anything.
"""

from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

#: Minimum number of examples of the *minority* class required before
#: k-fold cross-validation is attempted. Below this, CV folds would be
#: too small to mean anything, so cross_validate() skips with a clear
#: reason instead of producing numbers that look precise but aren't.
MIN_MINORITY_CLASS_FOR_CV = 5
DEFAULT_CV_FOLDS = 5


def compute_metrics(y_true, y_pred, y_proba: Optional[np.ndarray] = None) -> dict:
    """Accuracy/precision/recall/F1/confusion matrix (+ ROC-AUC where computable).

    Precision/recall/F1 use zero_division=0 so an evaluation set missing
    one class doesn't raise - it reports 0.0 for the undefined metric
    instead. Report the full set together; accuracy alone does not
    establish that a model works, especially under class imbalance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "roc_auc": None,
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def format_confusion_matrix(cm) -> str:
    """Human-readable 2x2 confusion matrix (rows=actual, cols=predicted)."""
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        return str(cm)
    tn, fp, fn, tp = cm.ravel()
    return "\n".join(
        [
            "                 Predicted 0   Predicted 1",
            f"Actual 0 (neg)   {tn:>11}   {fp:>11}",
            f"Actual 1 (pos)   {fn:>11}   {tp:>11}",
        ]
    )


def cross_validate(model, X, y, cv_folds: int = DEFAULT_CV_FOLDS) -> dict:
    """Stratified k-fold cross-validation, or a clear skip reason if the
    dataset is too small for it to be meaningful.

    Returns {"performed": False, "reason": ...} rather than raising, so
    the training script can print the reason and move on.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return {"performed": False, "reason": "only one class present in the dataset"}

    min_class_count = int(counts.min())
    if min_class_count < MIN_MINORITY_CLASS_FOR_CV:
        return {
            "performed": False,
            "reason": (
                f"smallest class has only {min_class_count} example(s); need at least "
                f"{MIN_MINORITY_CLASS_FOR_CV} per class for a meaningful {cv_folds}-fold cross-validation"
            ),
        }

    folds = min(cv_folds, min_class_count)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
    return {
        "performed": True,
        "folds": folds,
        "f1_scores": [float(s) for s in scores],
        "f1_mean": float(np.mean(scores)),
        "f1_std": float(np.std(scores)),
    }


def feature_importance_report(model, feature_names) -> list:
    """[(feature_name, importance), ...] sorted by importance, descending.

    Random Forest impurity-based importance reflects which features the
    trees found useful to split on for THIS dataset - it does not
    establish that a feature causally determines whether a signal is a
    satellite pass, and should be reported/read as such.
    """
    importances = model.feature_importances_
    pairs = list(zip(feature_names, (float(v) for v in importances)))
    return sorted(pairs, key=lambda pair: pair[1], reverse=True)


def format_feature_importance(report) -> str:
    if not report:
        return "(no features)"
    name_width = max(len(name) for name, _ in report)
    lines = [f"{'Feature'.ljust(name_width)}   Importance"]
    for name, importance in report:
        lines.append(f"{name.ljust(name_width)}   {importance:.4f}")
    return "\n".join(lines)


def save_feature_importance_chart(report, output_path) -> None:
    """Optional horizontal bar chart of feature importances (highest at top)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name, _ in report]
    values = [value for _, value in report]

    plt.figure(figsize=(8, max(3, 0.4 * len(names))))
    y_pos = range(len(names))
    # Reverse so the highest-importance feature (first in `report`) plots at the top.
    plt.barh(y_pos, values[::-1])
    plt.yticks(list(y_pos), names[::-1])
    plt.xlabel("Random Forest feature importance (impurity-based)")
    plt.title("Feature importance - does not imply causation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
