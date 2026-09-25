"""Every training run and external test leaves a row in models/ml_history.csv."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ml.history import COLUMNS, HISTORY_NAME  # noqa: E402
from ml.train import main as train_main  # noqa: E402

RSP = ROOT / "data" / "training" / "rsp03_camras_features.csv"


def test_training_and_external_test_are_recorded(tmp_path):
    import pytest
    if not RSP.exists():
        pytest.skip("RSP-03 dataset not in this checkout")
    model = tmp_path / "models" / "m.joblib"
    assert train_main(["--dataset", str(RSP), "--output", str(model), "--model-version", "v1",
                       "--no-tune", "--notes", "first run"]) == 0
    assert train_main(["--dataset", str(RSP), "--output", str(model), "--model-version", "v2", "--no-tune"]) == 0
    import evaluate_model
    assert evaluate_model.main(["--model", str(model), "--dataset", str(RSP), "--notes", "self-check"]) == 0
    h = pd.read_csv(tmp_path / "models" / HISTORY_NAME)
    assert list(h.columns) == COLUMNS
    assert h.kind.tolist() == ["train", "train", "external_test"]
    assert h.model_version.tolist() == ["v1", "v2", "v2"]
    first = h.iloc[0]
    assert first.n_rows == len(pd.read_csv(RSP)) and first.n_train + first.n_val + first.n_test == first.n_rows
    assert first.notes == "first run" and 0 <= first.roc_auc <= 1
    assert h.iloc[2].tn + h.iloc[2].fp + h.iloc[2].fn + h.iloc[2].tp == first.n_rows


def test_history_failure_never_breaks_training(tmp_path, monkeypatch, capsys):
    from ml import history
    monkeypatch.setattr(history, "append", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert history.record_external(tmp_path / "m.joblib", "x.csv", {"accuracy": 1.0}) is None
    assert "could not update training history" in capsys.readouterr().err
