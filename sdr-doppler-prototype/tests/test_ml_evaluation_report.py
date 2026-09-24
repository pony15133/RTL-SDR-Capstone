"""scripts/ml_evaluation_report.py: error rates, grouped split without
leakage, and the evidence files it writes (run on the real RSP-03 dataset)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTO / "scripts"))
sys.path.insert(0, str(PROTO / "src"))

import ml_evaluation_report as mer  # noqa: E402


def test_error_rates_from_confusion_matrix():
    m = mer.with_error_rates({"confusion_matrix": [[8, 2], [1, 9]]})
    assert m["false_positive_rate"] == pytest.approx(0.2)
    assert m["false_negative_rate"] == pytest.approx(0.1)
    assert m["tn_fp_fn_tp"] == [8, 2, 1, 9]


def test_error_rates_undefined_without_that_class():
    m = mer.with_error_rates({"confusion_matrix": [[0, 0], [3, 7]]})
    assert m["false_positive_rate"] is None and m["false_negative_rate"] == pytest.approx(0.3)


def test_report_on_real_dataset_keeps_recordings_in_one_split(tmp_path):
    code = mer.main(["--output-dir", str(tmp_path / "ev"), "--model", str(tmp_path / "m.joblib")])
    assert code == 0
    report = json.loads((tmp_path / "ev" / "ml_evaluation.json").read_text())
    assert report["leakage_check"]["ok"]
    assert report["dataset"]["n_synthetic_rows"] == 0
    sets = [b["set"] for b in report["results"]]
    assert sets[:3] == ["training (train + validation, resubstitution)", "validation", "held-out test"]
    for block in report["results"]:
        assert {"accuracy", "precision", "recall", "f1_score", "confusion_matrix",
                "false_positive_rate", "false_negative_rate", "n_recordings"} <= set(block)
    assigned = [r for recs in report["split_assignment"].values() for r in recs]
    assert len(assigned) == len(set(assigned)) == report["dataset"]["n_recordings"]
    splits = pd.read_csv(tmp_path / "ev" / "split_assignment.csv")
    assert (splits.groupby("recording_group")["split"].nunique() == 1).all()
    assert "IMPLEMENTED BUT NOT VALIDATED" in (tmp_path / "ev" / "ML_EVALUATION.md").read_text()
