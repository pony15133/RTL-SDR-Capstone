"""Train the Random Forest satellite-candidate classifier from a labelled CSV.

    python train_model.py --dataset data/training/features.csv --output models/random_forest.joblib

(or, from src/: python -m ml.train --dataset ... --output ...)

IMPORTANT: this script trains whatever it's given. A model trained (even
partly) on synthetic data is refused unless --allow-synthetic is passed
explicitly, and is always saved with trained_on_synthetic_data=true in its
metadata. Training on synthetic data verifies the pipeline runs - it does
NOT produce a scientifically validated model. See data/training/README.md.

Methodology (train / validation / test):

1. Rows are grouped by recording (``recording_id`` column, or the file part
   of ``source_file`` for chunked windows) so chunks of one recording never
   land in two different splits.
2. Recordings are split into train / validation / test (default 70/15/15),
   stratified by label where possible.
3. Hyperparameters are chosen by F1 on the VALIDATION set only.
4. The chosen configuration is refit on train + validation.
5. The TEST set is scored exactly once, at the end. It is never used for
   tuning or cross-validation, so its metrics are an honest estimate.
"""

import argparse
import itertools
import sys
from dataclasses import dataclass, field
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
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15
DEFAULT_N_ESTIMATORS = 100
DEFAULT_CV_FOLDS = 5

#: Small, deliberately modest search space - the datasets here are small,
#: so a big grid would just overfit the validation set.
DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 3],
}

#: chunk_bin_to_dataset.py writes source_file as "<path>#samples=<start>-<end>";
#: everything before this marker identifies the parent recording.
WINDOW_SUFFIX_MARKER = "#samples="

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = "train", "validation", "test"

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
    #: Metrics on the held-out TEST set (scored once, after tuning).
    metrics: dict
    cv_result: dict
    importance_report: list
    n_train: int
    n_test: int
    label_distribution: dict
    trained_on_synthetic_data: bool
    n_val: int = 0
    #: Metrics of the selected configuration on the VALIDATION set
    #: (model fit on the train split only).
    val_metrics: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    search_results: list = field(default_factory=list)
    group_source: str = "row"
    splits_path: Optional[Path] = None


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


def resolve_groups(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """One group id per row, identifying which recording the row came from.

    Priority:
      1. ``recording_id`` column, if present and filled in.
      2. ``source_file`` with any ``#samples=<a>-<b>`` window suffix removed,
         so every chunk cut from one .bin/.iq file shares a group.
      3. ``capture_id`` (each row is its own group - no leakage protection
         possible, reported as such).

    Returns (groups, source) where source is "recording_id",
    "source_file" or "row".
    """
    if "recording_id" in df.columns and df["recording_id"].notna().all():
        return df["recording_id"].astype(str).to_numpy(), "recording_id"
    if "source_file" in df.columns and df["source_file"].notna().all():
        stripped = df["source_file"].astype(str).str.split(WINDOW_SUFFIX_MARKER, n=1, regex=False).str[0]
        return stripped.to_numpy(), "source_file"
    return df["capture_id"].astype(str).to_numpy(), "row"


def _group_labels(groups: np.ndarray, y: np.ndarray) -> dict:
    """Majority label of each group (a recording is normally all 0 or all 1,
    but chunked recordings can be mixed - majority is good enough for
    stratifying the split)."""
    labels = {}
    for group in np.unique(groups):
        values = y[groups == group]
        labels[group] = int(np.round(values.mean()))
    return labels


def grouped_train_val_test_split(
    groups: np.ndarray,
    y: np.ndarray,
    *,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Assign every group (recording) to exactly one of train/validation/test.

    Groups are stratified by their majority label, so each split gets a
    share of both classes when there are enough recordings. Each split is
    guaranteed at least one group; train keeps at least one group of each
    class whenever that class has any.

    Returns {"train": row_idx, "validation": row_idx, "test": row_idx}.
    Raises ValueError if there are fewer than 3 groups.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1 or val_size + test_size >= 1:
        raise ValueError(
            f"val_size and test_size must each be in (0, 1) and sum to < 1; got {val_size!r}, {test_size!r}"
        )
    groups = np.asarray(groups)
    y = np.asarray(y)
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        raise ValueError(
            f"Need at least 3 distinct recordings for a train/validation/test split, got {unique_groups.size}."
        )

    rng = np.random.default_rng(random_state)
    labels = _group_labels(groups, y)
    assignment: dict = {}
    per_class_train: dict = {}

    for cls in sorted(set(labels.values())):
        members = np.array(sorted(g for g, lbl in labels.items() if lbl == cls))
        members = members[rng.permutation(members.size)]
        n = members.size
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        # Always leave at least one recording of this class for training.
        while n_test + n_val > n - 1 and (n_test > 0 or n_val > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            else:
                n_test -= 1
        for g in members[:n_test]:
            assignment[g] = SPLIT_TEST
        for g in members[n_test:n_test + n_val]:
            assignment[g] = SPLIT_VAL
        train_members = list(members[n_test + n_val:])
        for g in train_members:
            assignment[g] = SPLIT_TRAIN
        per_class_train[cls] = train_members

    # Rounding can leave validation or test empty on small datasets - move
    # one training recording across, taken from the class with the most
    # training recordings so train keeps both classes where possible.
    for target in (SPLIT_TEST, SPLIT_VAL):
        if any(split == target for split in assignment.values()):
            continue
        donor_cls = max(per_class_train, key=lambda c: len(per_class_train[c]))
        if len(per_class_train[donor_cls]) <= 1:
            donor_cls = next((c for c in per_class_train if per_class_train[c]), None)
        if donor_cls is None or not per_class_train[donor_cls]:
            raise ValueError("Not enough recordings to fill train, validation and test splits.")
        moved = per_class_train[donor_cls].pop()
        assignment[moved] = target

    if not any(split == SPLIT_TRAIN for split in assignment.values()):
        raise ValueError("Train split ended up empty.")

    row_splits = np.array([assignment[g] for g in groups])
    return {name: np.flatnonzero(row_splits == name) for name in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)}


def _param_grid(args: argparse.Namespace) -> list[dict]:
    """The configurations to try. --no-tune collapses it to the single
    configuration given on the command line."""
    if getattr(args, "no_tune", False):
        return [{
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": 1,
        }]
    keys = list(DEFAULT_PARAM_GRID)
    return [dict(zip(keys, values)) for values in itertools.product(*(DEFAULT_PARAM_GRID[k] for k in keys))]


def _selection_score(metrics: dict, y_val: np.ndarray) -> tuple:
    """Higher is better. F1 when the validation set has both classes,
    otherwise accuracy (F1 is undefined/meaningless with one class);
    ROC-AUC breaks ties."""
    primary = metrics["f1_score"] if len(np.unique(y_val)) == 2 else metrics["accuracy"]
    auc = metrics.get("roc_auc")
    return (primary, auc if auc is not None else -1.0)


def _positive_proba(model: RandomForestClassifier, X: np.ndarray) -> Optional[np.ndarray]:
    if 1 not in model.classes_:
        return None
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def tune_on_validation(X_train, y_train, X_val, y_val, *, grid, class_weight, random_state) -> tuple[dict, dict, list]:
    """Fit each configuration on TRAIN, score it on VALIDATION, keep the best.

    Returns (best_params, best_val_metrics, all_results). Ties are broken
    in favour of the simpler model (fewer trees, shallower, bigger leaves)
    because grid order is simplest-first after sorting.
    """
    def simplicity(params):
        depth = params["max_depth"] if params["max_depth"] is not None else 10**6
        return (params["n_estimators"], depth, -params["min_samples_leaf"])

    results = []
    best = None
    for params in sorted(grid, key=simplicity):
        model = train_random_forest(
            X_train, y_train,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=class_weight,
            random_state=random_state,
        )
        metrics = compute_metrics(y_val, model.predict(X_val), _positive_proba(model, X_val))
        score = _selection_score(metrics, y_val)
        results.append({"params": params, "validation_f1": metrics["f1_score"],
                        "validation_accuracy": metrics["accuracy"], "validation_roc_auc": metrics["roc_auc"]})
        if best is None or score > best[0]:
            best = (score, params, metrics)
    return best[1], best[2], results


def train_random_forest(X_train, y_train, *, n_estimators, max_depth, class_weight, random_state,
                        min_samples_leaf: int = 1) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def _label_counts(y: np.ndarray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(classes, counts)}


def run_training(args: argparse.Namespace) -> TrainingResult:
    dataset_path = Path(args.dataset)
    print_progress("Loading training dataset", 10, 100)
    df = load_dataset(dataset_path).reset_index(drop=True)
    trained_on_synthetic_data = check_synthetic_guard(df, args.allow_synthetic)

    print_progress("Splitting train / validation / test", 25, 100)
    X, y = split_features_labels(df)
    groups, group_source = resolve_groups(df)
    if group_source != "row" and np.unique(groups).size < 3:
        print(
            f"WARNING: grouping by {group_source} gives only {np.unique(groups).size} recording(s) - "
            "too few for a grouped train/validation/test split. Falling back to one group per row; "
            "if rows share a recording, the test metrics WILL be optimistic.",
            file=sys.stderr,
        )
        groups, group_source = df["capture_id"].astype(str).to_numpy(), "row"
    elif group_source == "row":
        print(
            "WARNING: no recording_id or source_file column - every row is treated as its own "
            "recording. If several rows come from one capture, the split cannot prevent leakage.",
            file=sys.stderr,
        )
    try:
        split_idx = grouped_train_val_test_split(
            groups, y, val_size=args.val_size, test_size=args.test_size, random_state=args.random_state,
        )
    except ValueError as exc:
        raise DatasetValidationError(f"Cannot split dataset: {exc}") from exc
    train_idx, val_idx, test_idx = split_idx[SPLIT_TRAIN], split_idx[SPLIT_VAL], split_idx[SPLIT_TEST]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print_progress("Tuning hyperparameters on validation set", 45, 100)
    best_params, val_metrics, search_results = tune_on_validation(
        X_train, y_train, X_val, y_val,
        grid=_param_grid(args), class_weight=args.class_weight, random_state=args.random_state,
    )

    print_progress("Refitting best model on train + validation", 65, 100)
    dev_idx = np.concatenate([train_idx, val_idx])
    X_dev, y_dev, groups_dev = X[dev_idx], y[dev_idx], groups[dev_idx]
    model = train_random_forest(
        X_dev, y_dev,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        class_weight=args.class_weight,
        random_state=args.random_state,
    )

    print_progress("Evaluating once on held-out test set", 80, 100)
    metrics = compute_metrics(y_test, model.predict(X_test), _positive_proba(model, X_test))

    # Cross-validation on the development set only - the test set stays untouched.
    cv_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        class_weight=args.class_weight,
        random_state=args.random_state,
    )
    cv_result = cross_validate(
        cv_model, X_dev, y_dev, cv_folds=args.cv_folds,
        groups=None if group_source == "row" else groups_dev,
    )
    cv_result["data"] = "train+validation (test set excluded)"

    importance_report = feature_importance_report(model, FEATURE_NAMES)
    label_distribution = _label_counts(y)
    split_summary = {
        name: {
            "n_samples": int(idx.size),
            "n_recordings": int(np.unique(groups[idx]).size),
            "label_distribution": _label_counts(y[idx]),
        }
        for name, idx in split_idx.items()
    }

    model_version = args.model_version or f"random_forest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    metadata = {
        "model_type": "RandomForestClassifier",
        "model_version": model_version,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_forest_params": model.get_params(),
        "n_training_samples": int(train_idx.size),
        "n_validation_samples": int(val_idx.size),
        "n_test_samples": int(test_idx.size),
        "n_total_samples": int(len(X)),
        "label_distribution": label_distribution,
        "split_strategy": "grouped train/validation/test; tuned on validation; refit on train+validation; test scored once",
        "group_source": group_source,
        "splits": split_summary,
        "selected_hyperparameters": best_params,
        "hyperparameter_search": search_results,
        "validation_metrics": val_metrics,
        "evaluation_set": "test",
        "evaluation_metrics": metrics,
        "final_model_trained_on": "train+validation",
        "cross_validation": cv_result,
        "feature_importance": [{"feature": name, "importance": value} for name, value in importance_report],
        "trained_on_synthetic_data": trained_on_synthetic_data,
        "dataset_path": str(dataset_path),
        "random_state": args.random_state,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "notes": args.notes,
        "disclaimer": VALIDATION_DISCLAIMER,
    }

    bundle = ModelBundle(classifier=model, feature_names=FEATURE_NAMES, metadata=metadata)
    model_path = bundle.save(Path(args.output))
    metadata_path = ModelBundle.metadata_path_for(model_path)

    importance_csv_path = model_path.with_name(model_path.stem + "_feature_importance.csv")
    pd.DataFrame(importance_report, columns=["feature", "importance"]).to_csv(importance_csv_path, index=False)

    # Which row went where - lets anyone re-check that no recording was split
    # across sets, and re-score the exact same test set later.
    row_split = np.empty(len(df), dtype=object)
    for name, idx in split_idx.items():
        row_split[idx] = name
    splits_path = model_path.with_name(model_path.stem + "_splits.csv")
    pd.DataFrame({
        "capture_id": df["capture_id"], "recording_group": groups, "split": row_split, "label": y,
    }).to_csv(splits_path, index=False)

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
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
        label_distribution=label_distribution,
        trained_on_synthetic_data=trained_on_synthetic_data,
        n_val=int(val_idx.size),
        val_metrics=val_metrics,
        best_params=best_params,
        search_results=search_results,
        group_source=group_source,
        splits_path=splits_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Random Forest satellite-candidate classifier.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to labelled training CSV")
    parser.add_argument("--output", required=True, type=Path, help="Path to save the trained model (.joblib)")
    parser.add_argument("--model-version", default=None, help="Explicit model version string (default: timestamp-based)")
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE,
                        help="Fraction of recordings held out for hyperparameter selection (default 0.15)")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                        help="Fraction of recordings held out for the final, one-time evaluation (default 0.15)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip the validation grid search and use --n-estimators/--max-depth as given")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS, help="Only used with --no-tune")
    parser.add_argument("--max-depth", type=int, default=None, help="Only used with --no-tune")
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


def _print_metrics_block(title: str, metrics: dict) -> None:
    print(title)
    print("-" * len(title))
    print(f"Samples: {metrics.get('n_samples', 'N/A')}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 score: {metrics['f1_score']:.3f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")


def _print_report(result: TrainingResult) -> None:
    if result.trained_on_synthetic_data:
        print("=" * 78)
        print("WARNING: this model was trained (at least partly) on SYNTHETIC data.")
        print("Its evaluation metrics do NOT represent real-world performance.")
        print("Status: IMPLEMENTED BUT NOT VALIDATED.")
        print("=" * 78)

    print()
    print("Model training summary")
    print("=====================")
    print(f"Model path: {result.model_path}")
    print(f"Metadata path: {result.metadata_path}")
    if result.splits_path:
        print(f"Split assignments: {result.splits_path}")
    print(f"Label distribution (all data): {result.label_distribution}")
    print(f"Grouped by: {result.group_source}")
    print(f"Train / validation / test samples: {result.n_train} / {result.n_val} / {result.n_test}")
    print(f"Selected hyperparameters (by validation F1): {result.best_params}")
    print()
    _print_metrics_block("Validation set (used to pick hyperparameters)", result.val_metrics)
    print()
    _print_metrics_block("Test set (held out, scored once - report THESE numbers)", result.metrics)
    print()
    print("Test confusion matrix:")
    print(format_confusion_matrix(result.metrics["confusion_matrix"]))
    print()
    if result.cv_result.get("performed"):
        grouped = " (grouped by recording)" if result.cv_result.get("grouped_by_recording") else ""
        print(
            f"Cross-validation F1 on train+validation{grouped}: "
            f"{result.cv_result['f1_mean']:.3f} +/- {result.cv_result['f1_std']:.3f} "
            f"over {result.cv_result['folds']} folds"
        )
    else:
        print(f"Cross-validation: skipped - {result.cv_result.get('reason')}")
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
