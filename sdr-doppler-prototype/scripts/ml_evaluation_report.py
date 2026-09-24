#!/usr/bin/env python3
"""Reproducible ML evaluation evidence for the Random Forest IQ-feature model.

    python scripts/ml_evaluation_report.py                      # RSP-03 real dataset, defaults
    python scripts/ml_evaluation_report.py --dataset data/training/features.csv --output-dir ../evidence/ml

Runs the repository's own training methodology (src/ml/train.py
run_training(): recordings grouped, split train / validation / test,
hyperparameters picked on validation, refit on train + validation, test
scored once) and then reports, separately and clearly labelled:

  * training performance   - the final model scored on the rows it was fit on
                             (train + validation). Resubstitution: optimistic
                             by construction, shown only to expose overfitting.
  * validation performance - selected configuration fit on train only, scored
                             on validation. The split was used to choose
                             hyperparameters, so it is also optimistic.
  * held-out test          - the headline number; never used for fitting,
                             tuning or cross-validation.
  * external test          - only with --external-model/--external-dataset
                             (a model scored on a dataset it never saw).
  * leave-one-recording-out (supplementary) - each recording held out once,
                             fixed default hyperparameters, to show how much the
                             result depends on WHICH recording is in the test set.

Every block carries accuracy, precision, recall, F1, the confusion matrix,
false-positive rate and false-negative rate, and the number of independent
recording groups behind it. Writes <output-dir>/ml_evaluation.json and
<output-dir>/ML_EVALUATION.md. The trained model itself goes to --model
(default models/, which git ignores) - it is a run-time artifact, not evidence.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parents[1]
REPO = PROTO.parent
sys.path.insert(0, str(PROTO / "src"))

from ml.evaluation import compute_metrics, format_confusion_matrix  # noqa: E402
from ml.model import ModelBundle  # noqa: E402
from ml.train import (  # noqa: E402
    DEFAULT_N_ESTIMATORS,
    FEATURE_SETS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_arg_parser,
    load_dataset,
    resolve_groups,
    run_training,
    split_features_labels,
    train_random_forest,
)

DEFAULT_DATASET = PROTO / "data" / "training" / "rsp03_camras_features.csv"
DEFAULT_MODEL = PROTO / "models" / "rsp03_iq_rf_evaluation.joblib"
DEFAULT_OUTPUT_DIR = REPO / "evidence" / "ml"


def with_error_rates(metrics: dict) -> dict:
    """Add false-positive / false-negative rates (None when undefined:
    no actual negatives / positives in the set)."""
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    out = dict(metrics)
    out["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else None
    out["false_negative_rate"] = fn / (fn + tp) if (fn + tp) else None
    out["tn_fp_fn_tp"] = [int(tn), int(fp), int(fn), int(tp)]
    return out


def _scores(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, list(model.classes_).index(1)] if 1 in model.classes_ else None
    return with_error_rates(compute_metrics(y, model.predict(X), proba))


def _block(name: str, metrics: dict, groups, note: str) -> dict:
    groups = sorted(set(map(str, groups)))
    return {"set": name, "n_recordings": len(groups), "recordings": groups, "note": note, **metrics}


def leave_one_recording_out(X, y, groups, random_state: int) -> list[dict]:
    """Supplementary: hold each recording out once, train on the rest with
    fixed default hyperparameters (no tuning, so no selection leakage)."""
    folds = []
    for g in sorted(set(groups)):
        test = groups == g
        model = train_random_forest(X[~test], y[~test], n_estimators=DEFAULT_N_ESTIMATORS, max_depth=None,
                                    class_weight=None, random_state=random_state)
        folds.append(_block(f"LORO: {g} held out", _scores(model, X[test], y[test]), [g],
                            f"trained on {len(set(groups[~test]))} other recording(s), default hyperparameters"))
    return folds


def external_test(model_path: Path, dataset_path: Path) -> dict:
    bundle = ModelBundle.load(model_path)
    df = pd.read_csv(dataset_path)
    X = df[list(bundle.feature_names)].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups, _ = resolve_groups(df)
    return _block("external test", _scores(bundle.classifier, X, y), groups,
                  f"model {model_path.name} scored on {dataset_path.name}, which it was not trained on")


def environment() -> dict:
    import sklearn

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "git_commit": sha,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def to_markdown(report: dict) -> str:
    d = report["dataset"]
    lines = [
        "# ML evaluation - Random Forest (IQ features)",
        "",
        f"Generated by `sdr-doppler-prototype/scripts/ml_evaluation_report.py` at {report['environment']['generated_utc']}"
        f" on commit `{report['environment']['git_commit']}`.",
        "",
        "## Dataset",
        "",
        f"- File: `{d['path']}` ({d['n_rows']} rows, {d['n_recordings']} independent recordings, "
        f"labels {d['label_distribution']}, synthetic rows: {d['n_synthetic_rows']})",
        f"- Grouped by: `{d['group_source']}` - every row of a recording stays in one split "
        f"(verified: {'no recording appears in two splits' if report['leakage_check']['ok'] else 'LEAK FOUND'})",
        f"- Rows per recording: {d['rows_per_recording']}",
        "- Independent recording groups per split: " + ", ".join(
            f"{name} {len(recs)} ({', '.join(recs)})" for name, recs in report["split_assignment"].items()),
        "",
        "## Results",
        "",
        "| Set | Recordings | Rows | Accuracy | Precision | Recall | F1 | FPR | FNR | TN FP FN TP |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b in report["results"]:
        lines.append(f"| {b['set']} | {b['n_recordings']} ({', '.join(b['recordings'])}) | {b['n_samples']} | "
                     f"{_fmt(b['accuracy'])} | {_fmt(b['precision'])} | {_fmt(b['recall'])} | {_fmt(b['f1_score'])} | "
                     f"{_fmt(b['false_positive_rate'])} | {_fmt(b['false_negative_rate'])} | "
                     f"{' '.join(map(str, b['tn_fp_fn_tp']))} |")
    lines += ["", "What each row means:", ""]
    lines += [f"- **{b['set']}**: {b['note']}" for b in report["results"]]
    lines += ["", "### Held-out test confusion matrix (rows = actual, columns = predicted)", "", "```",
              format_confusion_matrix(report["results"][2]["confusion_matrix"]), "```", ""]
    cv = report["cross_validation"]
    lines.append("Cross-validation on train + validation only: " + (
        f"F1 {cv['f1_mean']:.3f} +/- {cv['f1_std']:.3f} over {cv['folds']} grouped folds, scores {cv['f1_scores']}"
        if cv.get("performed") else f"skipped - {cv.get('reason')}"))
    lines += ["", f"Selected hyperparameters (by validation F1): `{report['selected_hyperparameters']}`", ""]
    loro = report["leave_one_recording_out"]
    if loro:
        f1s = [b["f1_score"] for b in loro]
        lines += ["## Supplementary: leave-one-recording-out", "",
                  "Each recording held out once; default hyperparameters (100 trees, no depth limit), no tuning. "
                  "Shows how strongly the result depends on which recording is tested.", "",
                  "| Held out | Rows | Accuracy | Precision | Recall | F1 | FPR | FNR |", "|---|---|---|---|---|---|---|---|"]
        for b in loro:
            lines.append(f"| {b['recordings'][0]} | {b['n_samples']} | {_fmt(b['accuracy'])} | {_fmt(b['precision'])} | "
                         f"{_fmt(b['recall'])} | {_fmt(b['f1_score'])} | {_fmt(b['false_positive_rate'])} | "
                         f"{_fmt(b['false_negative_rate'])} |")
        lines += ["", f"F1 across held-out recordings: min {min(f1s):.3f}, max {max(f1s):.3f}, "
                      f"mean {np.mean(f1s):.3f}.", ""]
    if report.get("external_test") is None:
        lines += ["## External test", "", report["external_test_note"], ""]
    lines += ["## What these numbers do and do not support", ""] + [f"- {c}" for c in report["caveats"]]
    env = report["environment"]
    lines += ["", "## Environment", "",
              f"Python {env['python']}, {env['platform']}, numpy {env['numpy']}, pandas {env['pandas']}, "
              f"scikit-learn {env['scikit_learn']}. Random state {report['random_state']}.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Grouped train/validation/test evaluation evidence for the RF model.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Where the evaluated model is saved (git-ignored)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="iq")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--external-model", type=Path, help="Trained model to score on --external-dataset")
    p.add_argument("--external-dataset", type=Path, help="Labelled dataset the external model never saw")
    args = p.parse_args(argv)

    feature_names = FEATURE_SETS[args.feature_set]
    train_args = build_arg_parser().parse_args([
        "--dataset", str(args.dataset), "--output", str(args.model), "--feature-set", args.feature_set,
        "--random-state", str(args.random_state), "--model-version", "evaluation_" + args.dataset.stem,
    ])
    result = run_training(train_args)

    df = load_dataset(args.dataset, feature_names).reset_index(drop=True)
    X, y = split_features_labels(df, feature_names)
    groups, group_source = resolve_groups(df)
    splits = pd.read_csv(result.splits_path)
    idx = {name: splits.index[splits["split"] == name].to_numpy() for name in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)}
    leak = splits.groupby("recording_group")["split"].nunique()
    leakage_ok = bool((leak == 1).all())

    final_model = ModelBundle.load(result.model_path).classifier
    dev = np.concatenate([idx[SPLIT_TRAIN], idx[SPLIT_VAL]])
    results = [
        _block("training (train + validation, resubstitution)", _scores(final_model, X[dev], y[dev]), groups[dev],
               "final model scored on the rows it was fit on - optimistic by construction; a big gap to the test "
               "row means overfitting"),
        _block("validation", with_error_rates(result.val_metrics), groups[idx[SPLIT_VAL]],
               "selected configuration fit on the train split only, scored on validation; this split chose the "
               "hyperparameters, so it is optimistic too"),
        _block("held-out test", with_error_rates(result.metrics), groups[idx[SPLIT_TEST]],
               "final model (refit on train + validation) scored once on recordings never used for fitting, "
               "tuning or cross-validation - the number to report"),
    ]

    external = None
    external_note = ("Not run. No labelled real IQ-feature dataset independent of RSP-03 exists in the repository. "
                     "The SatNOGS-trained waterfall model has an external test path "
                     "(`scripts/evaluate_model.py` on `rsp03_waterfall_features.csv`), but it needs "
                     "`fetch_satnogs_dataset.py` to download training data from network.satnogs.org first.")
    if args.external_model and args.external_dataset:
        external = external_test(args.external_model, args.external_dataset)
        results.append(external)
        external_note = None

    n_rec = len(set(groups))
    caveats = [
        f"All {len(df)} rows come from {n_rec} recordings of ONE satellite pass, received by ONE station "
        "(CAMRAS Dwingeloo 25 m dish, 436.95 MHz, ci16 snapshots) - not from an RTL-SDR, and not from the "
        "METEOR/ISS targets the station is configured for. Nothing here shows the model generalises to other "
        "satellites, stations, receivers or the RTL-SDR deployment.",
        f"The held-out test set is {len(set(groups[idx[SPLIT_TEST]]))} recording(s) "
        f"({len(idx[SPLIT_TEST])} one-second windows). Windows from one recording are strongly correlated, so the "
        "effective sample size is closer to the number of recordings than the number of rows.",
        "Labels come from `scripts/label_known_carrier.py` (SNR around the known carrier, borderline windows "
        "dropped), checked by eye against waterfalls per data/training/README.md - they are not independent "
        "ground truth.",
        "About 75% of the ci16 samples in these snapshots are clipped (data/training/README.md), which affects "
        "the power-based features.",
        "The leave-one-recording-out table is supplementary: it varies which recording is tested, it is not a "
        "second independent validation.",
    ]
    cv = result.cv_result
    if cv.get("performed") and min(cv["f1_scores"]) < 0.8:
        caveats.append(
            f"A perfect or near-perfect test score here is NOT evidence the model is reliable: grouped "
            f"cross-validation on the development recordings ranged from F1 {min(cv['f1_scores']):.2f} to "
            f"{max(cv['f1_scores']):.2f}, i.e. the result swings with which recordings the model is trained on."
        )
    caveats += [
        "Status stays IMPLEMENTED BUT NOT VALIDATED for real-world RTL-SDR use.",
    ]
    report = {
        "dataset": {
            "path": str(args.dataset.relative_to(REPO) if args.dataset.is_relative_to(REPO) else args.dataset),
            "n_rows": int(len(df)), "n_recordings": n_rec, "group_source": group_source,
            "label_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "n_synthetic_rows": int(pd.to_numeric(df.get("is_synthetic", 0), errors="coerce").fillna(0).sum()),
            "rows_per_recording": {str(g): {"rows": int((groups == g).sum()), "positives": int(y[groups == g].sum())}
                                   for g in sorted(set(groups))},
        },
        "methodology": "src/ml/train.py run_training(): grouped train/validation/test; tuned on validation; "
                       "refit on train+validation; test scored once",
        "split_assignment": {name: sorted(set(map(str, groups[i]))) for name, i in idx.items()},
        "leakage_check": {"ok": leakage_ok, "splits_per_recording": {str(k): int(v) for k, v in leak.items()}},
        "selected_hyperparameters": result.best_params,
        "cross_validation": result.cv_result,
        "results": results,
        "external_test": external,
        "external_test_note": external_note,
        "leave_one_recording_out": leave_one_recording_out(X, y, groups, args.random_state),
        "feature_importance": [{"feature": n, "importance": v} for n, v in result.importance_report],
        "caveats": caveats,
        "random_state": args.random_state,
        "model_path_not_committed": str(result.model_path),
        "environment": environment(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ml_evaluation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "ML_EVALUATION.md").write_text(to_markdown(report), encoding="utf-8")
    # Which row went to which split - lets anyone re-check that no recording was split across sets.
    splits.to_csv(args.output_dir / "split_assignment.csv", index=False)
    print(to_markdown(report))
    print(f"\nWrote {args.output_dir / 'ml_evaluation.json'} and {args.output_dir / 'ML_EVALUATION.md'}")
    return 0 if leakage_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
