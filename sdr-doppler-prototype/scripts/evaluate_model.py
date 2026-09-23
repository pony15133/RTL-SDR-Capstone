#!/usr/bin/env python3
"""Score a trained model on a labelled dataset it was NOT trained on (external test).

    python scripts/evaluate_model.py --model models/waterfall_rf.joblib \\
        --dataset data/training/rsp03_waterfall_features.csv

The RSP-03 waterfall set is from a different satellite, station (a 25 m
dish), and data source (raw IQ, not SatNOGS images) than the SatNOGS
training data - so this is the honest "does it generalise?" number.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml.evaluation import compute_metrics, format_confusion_matrix  # noqa: E402
from ml.model import ModelBundle  # noqa: E402


def evaluate(model_path, dataset_path) -> dict:
    bundle = ModelBundle.load(Path(model_path))
    df = pd.read_csv(dataset_path)
    missing = [n for n in bundle.feature_names if n not in df.columns]
    if missing:
        raise SystemExit(f"FAILED: dataset lacks the model's features {missing}")
    X = df[list(bundle.feature_names)].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    clf = bundle.classifier
    proba = clf.predict_proba(X)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(y))
    return compute_metrics(y, clf.predict(X), proba)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="External evaluation of a trained model.")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    args = p.parse_args(argv)
    m = evaluate(args.model, args.dataset)
    print(f"External test on {args.dataset} ({m['n_samples']} rows)")
    for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        if m.get(k) is not None:
            print(f"  {k}: {m[k]:.3f}")
    print(format_confusion_matrix(m["confusion_matrix"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
