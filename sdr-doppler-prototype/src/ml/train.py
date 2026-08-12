"""Train the Random Forest satellite-candidate classifier from a labelled CSV.

    python train_model.py --dataset data/training/features.csv --output models/random_forest.joblib

(or, from src/: python -m ml.train --dataset ... --output ...)

IMPORTANT: this script trains whatever it's given. A model trained (even
partly) on synthetic data is refused unless --allow-synthetic is passed
explicitly, and is always saved with trained_on_synthetic_data=true in its
metadata. Training on synthetic data verifies the pipeline runs - it does
NOT produce a scientifically validated model. See data/training/README.md.
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features.extractor import FEATURE_NAMES
from ml.evaluation import (
    compute_metrics,
    cross_validate,
    feature_importance_report,
    format_confusion_matrix,
    format_feature_importance,
    save_feature_importance_chart,
)
from ml.model import ModelBundle

REQUIRED_COLUMNS = ("capture_id", *FEATURE_NAMES, "label")
SYNTHETIC_COLUMN = "is_synthetic"

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.25
DEFAULT_N_ESTIMATORS = 100
DEFAULT_CV_FOLDS = 5

VALIDATION_DISCLAIMER = (
    "This model's evaluation metrics reflect performance on the held-out portion "
    "of the given dataset only. Until validated against a substantial set of real, "
    "independently-verified satellite-pass captures, treat this model as "
    "IMPLEMENTED BUT NOT VALIDATED."
)


class DatasetValidationError(ValueError):
    """The training CSV is missing required columns or contains invalid values."""


@dataclass
class TrainingResult:
    model_path: Path
    metadata_path: Path
    metrics: dict
    cv_result: dict
    importance_report: list
    n_train: int
    n_test: int
    label_distribution: dict
    trained_on_synthetic_data: bool


def _invalid_value_mask(series: pd.Series) -> pd.Series:
    """True where a value is missing, non-numeric, or +/-infinite after numeric coercion."""
    numeric = pd.to_numeric(series, errors="coerce")
    is_missing = numeric.isna()
    is_inf = numeric.replace([np.inf, -np.inf], np.nan).isna() & ~is_missing
    return is_missing | is_inf


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise DatasetValidationError with a clear, actionable message on any problem."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Dataset is missing required column(s): {', '.join(missing)}")
    if df.empty:
        raise DatasetValidationError("Dataset has no rows")

    problems = []
    for col in (*FEATURE_NAMES, "label"):
        bad_rows = df.index[_invalid_value_mask(df[col])]
        if len(bad_rows) > 0:
            ids = df.loc[bad_rows, "capture_id"].tolist()
            problems.append(f"column '{col}': missing/non-numeric/infinite value(s) at capture_id {ids}")
    if problems:
        raise DatasetValidationError("Invalid values in training dataset:\n  " + "\n  ".join(problems))

    label_numeric = pd.to_numeric(df["label"], errors="coerce")
    invalid_labels = df.index[~label_numeric.isin([0, 1])]
    if len(invalid_labels) > 0:
        ids = df.loc[invalid_labels, "capture_id"].tolist()
        raise DatasetValidationError(f"'label' column must be 0 or 1; invalid at capture_id {ids}")

    duplicate_ids = df["capture_id"][df["capture_id"].duplicated()].tolist()
    if duplicate_ids:
        raise DatasetValidationError(f"Duplicate capture_id value(s): {duplicate_ids}")


def check_synthetic_guard(df: pd.DataFrame, allow_synthetic: bool) -> bool:
    """Returns True if the dataset contains rows marked is_synthetic=1.

    Raises DatasetValidationError if such rows exist and allow_synthetic
    is False - training must not silently proceed on synthetic data.
    """
    if SYNTHETIC_COLUMN not in df.columns:
        return False
    synthetic_mask = pd.to_numeric(df[SYNTHETIC_COLUMN], errors="coerce").fillna(0).astype(bool)
    if not synthetic_mask.any():
        return False
    if not allow_synthetic:
        raise DatasetValidationError(
            f"{int(synthetic_mask.sum())} of {len(df)} row(s) are marked is_synthetic=1. "
            "Training on synthetic data only verifies the pipeline runs - it does NOT "
            "produce a scientifically valid model. Re-run with --allow-synthetic to proceed "
            "anyway; the saved model will be clearly marked as trained on synthetic data."
        )
    return True


def load_dataset(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_dataset(df)
    return df


def split_features_labels(df: pd.DataFrame):
    X = df[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return X, y


def grouped_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by recording group to avoid leakage from multiple rows in the same capture.

    When a `recording_id` column is present, the split is performed at the
    recording-group level, not the row level. This keeps all rows from the same
    recording in either the training or test set, which avoids inflated metrics
    caused by data leakage.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size!r}")

    if "recording_id" not in df.columns:
        indices = df.index.to_numpy()
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
        return train_idx, test_idx

    unique_recordings = df["recording_id"].drop_duplicates().to_numpy()
    if unique_recordings.size < 2:
        raise ValueError("At least two unique recording_id values are required for a train/test split.")

    rng = np.random.default_rng(random_state)
    shuffled = unique_recordings[rng.permutation(unique_recordings.size)]
    n_test_groups = max(1, int(round(shuffled.size * test_size)))
    test_groups = shuffled[:n_test_groups]
    train_groups = shuffled[n_test_groups:]

    train_idx = df.index[df["recording_id"].isin(train_groups)].to_numpy()
    test_idx = df.index[df["recording_id"].isin(test_groups)].to_numpy()

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Grouped train/test split produced an empty train or test set.")

    return train_idx, test_idx


def train_random_forest(X_train, y_train, *, n_estimators, max_depth, class_weight, random_state) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def run_training(args: argparse.Namespace) -> TrainingResult:
    dataset_path = Path(args.dataset)
    print_progress("Loading training dataset", 15, 100)
    df = load_dataset(dataset_path)
    trained_on_synthetic_data = check_synthetic_guard(df, args.allow_synthetic)

    print_progress("Preparing training features", 35, 100)
    X, y = split_features_labels(df)

    classes, class_counts = np.unique(y, return_counts=True)
    can_stratify = len(classes) > 1 and class_counts.min() >= 2
    X_train = X_test = y_train = y_test = None
    if "recording_id" in df.columns:
        try:
            train_idx, test_idx = grouped_train_test_split(
                df,
                test_size=args.test_size,
                random_state=args.random_state,
            )
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        except ValueError as exc:
            print(f"WARNING: grouped split not possible ({exc}) - falling back to row-level split.", file=sys.stderr)
    if X_train is None:
        if can_stratify:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=args.test_size, random_state=args.random_state, stratify=y,
                )
            except ValueError as exc:
                # Stratification needs each class represented in both the train
                # and test splits; a tiny dataset can fail that even when each
                # class individually has >= 2 examples (e.g. 4 rows total).
                print(f"WARNING: stratified split not possible ({exc}) - splitting without stratification.", file=sys.stderr)
        if X_train is None:
            if can_stratify:
                pass  # warning already printed above
            else:
                print(
                    "WARNING: cannot stratify the train/test split (a class has fewer than 2 "
                    "examples, or only one class is present) - splitting without stratification.",
                    file=sys.stderr,
                )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state, stratify=None,
            )

    print_progress("Training Random Forest", 65, 100)
    model = train_random_forest(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight=args.class_weight,
        random_state=args.random_state,
    )

    print_progress("Evaluating model", 85, 100)
    y_pred = model.predict(X_test)
    y_proba = None
    if 1 in model.classes_:
        y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    cv_result = cross_validate(model, X, y, cv_folds=args.cv_folds)
    importance_report = feature_importance_report(model, FEATURE_NAMES)
    label_distribution = {str(k): int(v) for k, v in zip(classes, class_counts)}

    model_version = args.model_version or f"random_forest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    metadata = {
        "model_type": "RandomForestClassifier",
        "model_version": model_version,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_forest_params": model.get_params(),
        "n_training_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "n_total_samples": int(len(X)),
        "label_distribution": label_distribution,
        "evaluation_metrics": metrics,
        "cross_validation": cv_result,
        "feature_importance": [{"feature": name, "importance": value} for name, value in importance_report],
        "trained_on_synthetic_data": trained_on_synthetic_data,
        "dataset_path": str(dataset_path),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "notes": args.notes,
        "disclaimer": VALIDATION_DISCLAIMER,
    }

    bundle = ModelBundle(classifier=model, feature_names=FEATURE_NAMES, metadata=metadata)
    model_path = bundle.save(Path(args.output))
    metadata_path = ModelBundle.metadata_path_for(model_path)

    importance_csv_path = model_path.with_name(model_path.stem + "_feature_importance.csv")
    pd.DataFrame(importance_report, columns=["feature", "importance"]).to_csv(importance_csv_path, index=False)

    if args.save_importance_chart:
        chart_path = model_path.with_name(model_path.stem + "_feature_importance.png")
        save_feature_importance_chart(importance_report, chart_path)

    print_progress("Saving trained model", 100, 100)
    return TrainingResult(
        model_path=model_path,
        metadata_path=metadata_path,
        metrics=metrics,
        cv_result=cv_result,
        importance_report=importance_report,
        n_train=len(X_train),
        n_test=len(X_test),
        label_distribution=label_distribution,
        trained_on_synthetic_data=trained_on_synthetic_data,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Random Forest satellite-candidate classifier.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to labelled training CSV")
    parser.add_argument("--output", required=True, type=Path, help="Path to save the trained model (.joblib)")
    parser.add_argument("--model-version", default=None, help="Explicit model version string (default: timestamp-based)")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--class-weight", default=None, help="e.g. 'balanced' for imbalanced datasets")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Fixed by default for reproducible experiments")
    parser.add_argument("--allow-synthetic", action="store_true", help="Required if the dataset contains is_synthetic=1 rows")
    parser.add_argument("--save-importance-chart", action="store_true", help="Also save a feature-importance bar chart PNG")
    parser.add_argument("--notes", default=None, help="Free-text note stored in the model's metadata")
    return parser


def print_progress(label: str, current: int, total: int) -> None:
    if total <= 0:
        total = 1
    percent = min(100, max(0, int((current / total) * 100)))
    width = 20
    filled = int(width * percent / 100)
    bar = "#" * filled + "-" * (width - filled)
    print(f"{label}: {percent:3d}% |{bar}| {current}/{total}")


def _print_report(result: TrainingResult) -> None:
    if result.trained_on_synthetic_data:
        print("=" * 78)
        print("WARNING: this model was trained (at least partly) on SYNTHETIC data.")
        print("Its evaluation metrics do NOT represent real-world performance.")
        print("Status: IMPLEMENTED BUT NOT VALIDATED.")
        print("=" * 78)

    metadata = {
        "model_version": result.model_path.stem,
        "n_training_samples": result.n_train,
        "n_test_samples": result.n_test,
        "evaluation_metrics": result.metrics,
        "cross_validation": result.cv_result,
        "feature_importance": [{"feature": name, "importance": importance} for name, importance in result.importance_report],
    }

    print()
    print("Model training summary")
    print("=====================")
    print(f"Model path: {result.model_path}")
    print(f"Metadata path: {result.metadata_path}")
    print(f"Training samples: {result.n_train}")
    print(f"Test samples: {result.n_test}")
    print(f"Label distribution: {result.label_distribution}")
    print(f"Accuracy: {result.metrics['accuracy']:.3f}")
    print(f"Precision: {result.metrics['precision']:.3f}")
    print(f"Recall: {result.metrics['recall']:.3f}")
    print(f"F1 score: {result.metrics['f1_score']:.3f}")
    if result.metrics.get("roc_auc") is not None:
        print(f"ROC AUC: {result.metrics['roc_auc']:.3f}")
    print()
    print("Confusion matrix:")
    print(format_confusion_matrix(result.metrics["confusion_matrix"]))
    print()
    if result.cv_result.get("performed"):
        print(f"Cross-validation F1: {result.cv_result['f1_mean']:.3f} +/- {result.cv_result['f1_std']:.3f}")
    else:
        print(f"Cross-validation: {result.cv_result.get('reason')}")
    print()
    print("Top feature importances:")
    for name, importance in result.importance_report[:5]:
        print(f"- {name}: {importance:.4f}")
    print()
    print(VALIDATION_DISCLAIMER)


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_training(args)
    except DatasetValidationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
