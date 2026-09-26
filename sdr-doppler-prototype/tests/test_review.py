"""Three-level decision follow-up: reviewing uncertain captures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import review  # noqa: E402
from database import get_result, insert_result, list_results  # noqa: E402
from history import captures_table  # noqa: E402

BASE = {"timestamp_utc": "2026-09-25T00:00:00Z", "detection_result": 0, "confidence_score": 0.0,
        "valid_signal_ratio": 0.0, "frequency_drift_hz": 0.0, "smoothness_score": 0.0}


def _capture(tmp_path, name, verdict="uncertain", folder="uncertain", with_matrix=True):
    rec = tmp_path / "recordings" / folder if folder else tmp_path / "recordings"
    rec.mkdir(parents=True, exist_ok=True)
    iq = rec / f"{name}.iq"
    iq.write_bytes(b"\x80" * 64)
    iq.with_suffix(".json").write_text("{}")
    img = tmp_path / "results" / f"{name}_waterfall.png"
    img.parent.mkdir(exist_ok=True)
    img.write_bytes(b"png")
    if with_matrix:
        m = np.random.default_rng(0).normal(0, 1, (128, 128))
        m[:, 62:66] += 6
        np.save(img.with_suffix(".npy"), m.astype(np.float16))
    db = tmp_path / "c.sqlite3"
    cid = insert_result(db, {**BASE, "input_file": str(iq), "raw_iq_file_path": str(iq), "satellite_name": "METEOR-M2-3",
                             "waterfall_image_path": str(img), "detection_verdict": verdict,
                             "review_status": "pending" if verdict == "uncertain" else None})
    return db, cid, iq


def test_signal_moves_iq_out_of_uncertain_and_adds_training_row(tmp_path):
    db, cid, iq = _capture(tmp_path, "a")
    csv = tmp_path / "station.csv"
    r = review.resolve(cid, 1, db_path=db, reviewer="khoa", policy="delete-negatives", training_csv=csv)
    row = get_result(db, cid)
    kept = tmp_path / "recordings" / "a.iq"
    assert kept.exists() and not iq.exists() and kept.with_suffix(".json").exists()
    assert row["raw_iq_file_path"] == str(kept) and row["review_status"] == "signal" and row["human_label"] == 1
    assert row["reviewed_by"] == "khoa" and r["training_row_added"]
    assert pd.read_csv(csv).label.tolist() == [1]


@pytest.mark.parametrize("policy, gone, folder", [("delete-negatives", True, None), ("archive-negatives", False, "rejected"),
                                                  ("keep-all", False, "uncertain")])
def test_noise_follows_the_policy(tmp_path, policy, gone, folder):
    db, cid, iq = _capture(tmp_path, "b")
    review.resolve(cid, 0, db_path=db, policy=policy, training_csv=tmp_path / "s.csv")
    row = get_result(db, cid)
    assert row["review_status"] == "noise" and row["human_label"] == 0
    if gone:
        assert row["raw_iq_file_path"] is None and not iq.exists()
    else:
        assert Path(row["raw_iq_file_path"]).parent.name == folder and Path(row["raw_iq_file_path"]).exists()


def test_label_is_final_and_missing_matrix_is_fine(tmp_path):
    db, cid, _ = _capture(tmp_path, "c", with_matrix=False)
    r = review.resolve(cid, 1, db_path=db, training_csv=tmp_path / "s.csv")
    assert r["training_row_added"] is False and get_result(db, cid)["review_status"] == "signal"
    with pytest.raises(ValueError):
        review.resolve(cid, 2, db_path=db)
    with pytest.raises(KeyError):
        review.resolve(999, 1, db_path=db)


def test_review_queue_filter_and_history_columns(tmp_path):
    db, cid, _ = _capture(tmp_path, "d")
    _capture(tmp_path, "e", verdict="detected", folder=None)
    pending = list_results(db, 10, review_status="pending")
    assert [r["id"] for r in pending] == [cid]
    table = captures_table(list_results(db, 10))
    assert "verdict" in table and "uncertain" in table and "detected" in table


# --------------------------------------------------------------------------- IQ-model training rows

import json  # noqa: E402

from features.extractor import FEATURE_NAMES  # noqa: E402

STORED = json.dumps({"features": {n: float(i) for i, n in enumerate(FEATURE_NAMES)},
                     "sample_rate_hz": 1_024_000.0, "nperseg": 1024, "noverlap": 512})


def _with_features(db, cid, value=STORED):
    from database import update_result
    update_result(db, cid, {"iq_features": value})


def test_review_adds_an_iq_training_row_from_saved_features(tmp_path):
    db, cid, _ = _capture(tmp_path, "f")
    _with_features(db, cid)
    iq_csv = tmp_path / "iq.csv"
    r = review.resolve(cid, 1, db_path=db, training_csv=tmp_path / "s.csv", iq_training_csv=iq_csv, reviewer="anh")
    df = pd.read_csv(iq_csv)
    assert r["iq_training_row_added"] and r["training_row_added"]
    assert list(df.columns[:len(review.IQ_COLUMNS)]) == review.IQ_COLUMNS
    assert df.label.tolist() == [1] and df.recording_id.tolist() == [f"station_{cid}"]
    assert df.snr_db.tolist() == [0.0] and df.signal_duration_seconds.tolist() == [8.0]
    assert df.labelled_by.tolist() == ["anh"] and df.is_synthetic.tolist() == [0]
    # a label is final: labelling again adds nothing
    assert review.add_iq_training_row(get_result(db, cid), 0, "x", iq_csv) is False


def test_iq_row_survives_the_delete_policy(tmp_path):
    db, cid, iq = _capture(tmp_path, "g")
    _with_features(db, cid)
    iq_csv = tmp_path / "iq.csv"
    r = review.resolve(cid, 0, db_path=db, policy="delete-negatives", training_csv=tmp_path / "s.csv",
                       iq_training_csv=iq_csv)
    assert not iq.exists() and r["iq_training_row_added"]
    assert pd.read_csv(iq_csv).label.tolist() == [0]


def test_iq_features_recomputed_from_the_file_for_older_captures(tmp_path):
    fs = 256_000
    t = np.arange(fs) / fs
    z = np.exp(2j * np.pi * 20_000 * t) * 0.5 + ([1, 1j] @ np.random.default_rng(1).normal(0, 0.05, (2, fs)))
    raw = np.empty(2 * fs, dtype=np.uint8)
    raw[0::2] = np.clip(z.real * 127.5 + 127.5, 0, 255)
    raw[1::2] = np.clip(z.imag * 127.5 + 127.5, 0, 255)
    db, cid, iq = _capture(tmp_path, "h")
    raw.tofile(iq)
    from database import update_result
    update_result(db, cid, {"sample_rate": fs, "frequency_hz": 137_900_000})
    iq_csv = tmp_path / "iq.csv"
    r = review.resolve(cid, 1, db_path=db, training_csv=tmp_path / "s.csv", iq_training_csv=iq_csv)
    df = pd.read_csv(iq_csv)
    assert r["iq_training_row_added"] and df.sample_rate_hz.tolist() == [fs]
    assert np.isfinite(df[list(FEATURE_NAMES)].to_numpy(dtype=float)).all()


def test_no_features_and_no_file_means_no_iq_row(tmp_path):
    db, cid, _ = _capture(tmp_path, "i")
    _with_features(db, cid, "not json")
    r = review.resolve(cid, 1, db_path=db, training_csv=tmp_path / "s.csv", iq_training_csv=tmp_path / "iq.csv")
    assert r["iq_training_row_added"] is False and get_result(db, cid)["review_status"] == "signal"
    assert not (tmp_path / "iq.csv").exists()


def test_station_iq_rows_train_together_with_rsp03(tmp_path):
    """The station CSV has the same layout as the RSP-03 set, so train_model takes both."""
    from ml.train import load_dataset
    rsp = ROOT / "data" / "training" / "rsp03_camras_features.csv"
    if not rsp.exists():
        pytest.skip("RSP-03 dataset not in this checkout")
    iq_csv = tmp_path / "iq.csv"
    for name, label in (("j", 1), ("k", 0)):
        db, cid, _ = _capture(tmp_path, name)
        _with_features(db, cid)
        review.resolve(cid, label, db_path=db, training_csv=tmp_path / "s.csv", iq_training_csv=iq_csv)
    df = load_dataset([rsp, iq_csv], FEATURE_NAMES)
    assert len(df) == len(pd.read_csv(rsp)) + 2 and df.recording_id.notna().all()
