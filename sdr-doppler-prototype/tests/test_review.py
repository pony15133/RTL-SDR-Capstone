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
