"""Real-data training workflow: waterfall-vetting labels, dataset migration,
tuned decision threshold, combined datasets, label review, labelling our
own station's captures and the RSP-03 external-set builder."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.waterfall_features import ADDED_FEATURES, WATERFALL_FEATURE_NAMES, waterfall_features  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fetch = _load_script("fetch_satnogs_dataset")
review = _load_script("review_labels")
station_labels = _load_script("label_station_captures")
rsp03 = _load_script("build_rsp03_waterfall_set")


def _waterfall(signal: bool, seed: int, strength: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 1, (128, 128))
    if signal:
        m[10:118, 62:66] += strength
    return m.astype(np.float32)


def _dataset(tmp_path, n_per_class=60, n_stations=20, noisy_labels=0):
    """A SatNOGS-style dataset (CSV + saved arrays), grouped by station."""
    arrays = tmp_path / "waterfalls"
    arrays.mkdir()
    rows = []
    for i in range(2 * n_per_class):
        label = int(i < n_per_class)
        m = _waterfall(bool(label), seed=i, strength=float(np.random.default_rng(i).uniform(1.0, 6.0)))
        np.save(arrays / f"{5000 + i}.npy", m.astype(np.float16))
        shown = 1 - label if i < noisy_labels else label
        rows.append({"capture_id": f"satnogs_{5000 + i}", **waterfall_features(m), "label": shown, "is_synthetic": 0,
                     "recording_id": f"station_{i % n_stations}", "source_file": "x", "satnogs_id": 5000 + i,
                     "norad_cat_id": 25544, "status": "good" if shown else "bad", "transmitter_mode": "FM"})
    csv = tmp_path / "satnogs.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, arrays


# --------------------------------------------------------------------------- labels

@pytest.mark.parametrize("obs, source, expected", [
    ({"status": "bad", "waterfall_status": "with-signal"}, "waterfall", (1, "waterfall")),
    ({"status": "good", "waterfall_status": "without-signal"}, "waterfall", (0, "waterfall")),
    ({"status": "good", "waterfall_status": True}, "waterfall", (1, "waterfall")),
    ({"status": "good", "waterfall_status": None}, "waterfall", (1, "status")),
    ({"status": "good", "waterfall_status": "unknown"}, "waterfall-only", (None, "waterfall not vetted")),
    ({"status": "bad", "waterfall_status": "with-signal"}, "status", (0, "status")),
    ({"status": "failed"}, "waterfall", (None, "status 'failed'")),
])
def test_label_for_prefers_waterfall_vetting(obs, source, expected):
    assert fetch.label_for(obs, source) == expected


def test_old_dataset_is_migrated_and_features_rebuilt(tmp_path):
    csv, arrays = _dataset(tmp_path, n_per_class=5, n_stations=3)
    old = pd.read_csv(csv).drop(columns=list(ADDED_FEATURES))
    old.to_csv(csv, index=False)
    assert fetch.needs_rebuild(csv) and fetch.needs_migration(csv)
    assert fetch.main(["--output", str(csv), "--arrays-dir", str(arrays), "--rebuild-features"]) == 0
    new = pd.read_csv(csv)
    assert not fetch.needs_rebuild(csv) and not fetch.needs_migration(csv)
    assert len(new) == 10 and list(new.columns[:len(fetch.CSV_COLUMNS)]) == fetch.CSV_COLUMNS
    m = np.load(arrays / "5000.npy").astype(np.float64)
    assert new.loc[0, "wf_vert_coherence"] == pytest.approx(waterfall_features(m)["wf_vert_coherence"])
    # appending after migration keeps columns aligned
    fetch.append_row(csv, {**new.iloc[0].to_dict(), "capture_id": "satnogs_9", "satnogs_id": 9})
    again = pd.read_csv(csv)
    assert again.iloc[-1]["satnogs_id"] == 9 and again.iloc[-1]["wf_peak_z"] == pytest.approx(new.loc[0, "wf_peak_z"])


def test_refresh_labels_relabels_from_waterfall_vetting(tmp_path, monkeypatch):
    csv, arrays = _dataset(tmp_path, n_per_class=3, n_stations=3)
    vetting = {5000: "with-signal", 5001: "without-signal", 5002: None,  # 5001 was "good" -> now 0
               5003: "with-signal", 5004: "unknown", 5005: "without-signal"}  # 5003 was "bad" -> now 1

    def http_get(url, **kw):
        sid = int(url.split("id=")[1].split("&")[0])
        return json.dumps([{"id": sid, "status": "good" if sid < 5003 else "bad",
                            "waterfall_status": vetting[sid]}]).encode(), {}

    monkeypatch.setattr(fetch, "http_get", http_get)
    r = fetch.refresh_labels(csv, "waterfall", page_delay=0)
    df = pd.read_csv(csv).set_index("satnogs_id")
    assert r == {"checked": 6, "relabelled": 2, "dropped": 0, "failed": 0}
    assert df.loc[5001, "label"] == 0 and df.loc[5003, "label"] == 1
    assert df.loc[5001, "label_source"] == "waterfall" and df.loc[5002, "label_source"] == "status"
    # waterfall-only drops the unvetted ones
    r2 = fetch.refresh_labels(csv, "waterfall-only", page_delay=0)
    assert r2["dropped"] == 2 and len(pd.read_csv(csv)) == 4


def test_windows_are_shuffled_over_the_whole_period():
    ordered = list(fetch.iter_windows(10, hours=6))
    shuffled = list(fetch.iter_windows(10, hours=6, shuffle_seed=1))
    assert sorted(shuffled) == sorted(ordered) and shuffled != ordered and len(ordered) == 40
    assert shuffled == list(fetch.iter_windows(10, hours=6, shuffle_seed=1))  # reproducible


# --------------------------------------------------------------------------- training

def test_training_combines_datasets_and_tunes_threshold(tmp_path):
    from detection.waterfall_detector import predict_waterfall
    from ml.train import build_arg_parser, run_training

    csv, arrays = _dataset(tmp_path, n_per_class=60, n_stations=20)
    df = pd.read_csv(csv)
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    df.iloc[::2].to_csv(first, index=False)
    df.iloc[1::2].drop(columns=["norad_cat_id"]).to_csv(second, index=False)   # different extra columns: fine
    res = run_training(build_arg_parser().parse_args([
        "--feature-set", "waterfall", "--dataset", str(first), str(second),
        "--output", str(tmp_path / "wf.joblib"), "--no-tune", "--cv-folds", "3"]))
    meta = json.loads(res.metadata_path.read_text())
    assert meta["n_total_samples"] == 120
    assert meta["threshold_selection"]["method"].startswith("grouped out-of-fold")
    assert 0.25 <= meta["decision_threshold"] <= 0.75 and res.metrics["threshold"] == meta["decision_threshold"]
    assert meta["feature_set"] == "waterfall" and "wf_vert_coherence" in [f["feature"] for f in meta["feature_importance"]]
    strong = predict_waterfall(res.model_path, _waterfall(True, 999))
    empty = predict_waterfall(res.model_path, _waterfall(False, 998))
    assert strong.ml_detection_result is True and empty.ml_detection_result is False


def test_small_dataset_keeps_threshold_half_and_fixed_threshold_is_used(tmp_path):
    from detection.waterfall_detector import predict_waterfall
    from ml.train import build_arg_parser, run_training

    csv, _ = _dataset(tmp_path, n_per_class=20, n_stations=6)
    res = run_training(build_arg_parser().parse_args([
        "--feature-set", "waterfall", "--dataset", str(csv), "--output", str(tmp_path / "s.joblib"), "--no-tune"]))
    assert res.threshold == 0.5 and res.threshold_selection["method"] == "default"

    res = run_training(build_arg_parser().parse_args([
        "--feature-set", "waterfall", "--dataset", str(csv), "--output", str(tmp_path / "f.joblib"),
        "--no-tune", "--threshold", "0.99"]))
    assert res.threshold == 0.99
    # a confident-but-below-0.99 detection is now "no"
    pred = predict_waterfall(res.model_path, _waterfall(True, 5))
    assert pred.ml_detection_result == (pred.ml_confidence_score >= 0.99)


def test_training_explains_how_to_upgrade_an_old_dataset(tmp_path):
    from ml.train import DatasetValidationError, build_arg_parser, run_training

    csv, _ = _dataset(tmp_path, n_per_class=10, n_stations=4)
    pd.read_csv(csv).drop(columns=list(ADDED_FEATURES)).to_csv(csv, index=False)
    with pytest.raises(DatasetValidationError, match="--rebuild-features"):
        run_training(build_arg_parser().parse_args(
            ["--feature-set", "waterfall", "--dataset", str(csv), "--output", str(tmp_path / "m.joblib")]))


def test_retention_follows_model_threshold_when_no_override():
    from retention import decide

    said_yes = SimpleNamespace(ml_confidence_score=0.35, ml_detection_result=True)   # model threshold 0.3
    said_no = SimpleNamespace(ml_confidence_score=0.55, ml_detection_result=False)   # model threshold 0.6
    assert decide(ml_detection=said_yes).keep is True
    assert decide(ml_detection=said_no).keep is False
    assert decide(ml_detection=said_yes, threshold=0.5).keep is False               # explicit override wins


# --------------------------------------------------------------------------- label review

def test_review_finds_flipped_labels_and_applies_fixes(tmp_path):
    csv, arrays = _dataset(tmp_path, n_per_class=40, n_stations=16, noisy_labels=6)
    out = tmp_path / "review"
    path = review.find_suspects(csv, arrays, out, top=12)
    sheet = pd.read_csv(path)
    assert (out / "sheet_01.png").exists() and len(sheet) == 12
    flipped = {5000 + i for i in range(6)}                    # true signal, labelled 0
    assert len(flipped & set(sheet["satnogs_id"])) >= 4     # most mislabelled rows are among the suspects
    sheet["your_label"] = ""
    sheet.loc[sheet.satnogs_id.isin(flipped), "your_label"] = "1"
    other = sheet[~sheet.satnogs_id.isin(flipped)].satnogs_id.iloc[0]
    sheet.loc[sheet.satnogs_id == other, "your_label"] = "x"
    sheet.to_csv(path, index=False)
    r = review.apply_review(csv, path)
    df = pd.read_csv(csv).set_index("satnogs_id")
    fixed = flipped & set(sheet["satnogs_id"])
    assert r["changed"] == len(fixed) and r["dropped"] == 1 and other not in df.index
    assert all(df.loc[s, "label"] == 1 and df.loc[s, "label_original"] == 0 and df.loc[s, "label_source"] == "manual"
               for s in fixed)
    # reviewed rows are not offered again
    again = pd.read_csv(review.find_suspects(csv, arrays, out, top=100))
    assert not (set(again.satnogs_id) & fixed)


# --------------------------------------------------------------------------- our own captures

def test_label_station_captures_appends_rows_for_training(tmp_path):
    from database import insert_result

    db = tmp_path / "c.sqlite3"
    ids = []
    for k, signal in enumerate([True, False, True]):
        img = tmp_path / f"pass{k}_waterfall.png"
        img.write_bytes(b"png")
        np.save(img.with_suffix(".npy"), _waterfall(signal, 100 + k).astype(np.float16))
        ids.append(insert_result(db, {"input_file": f"pass{k}.iq", "timestamp_utc": "2026-09-25T00:00:00Z",
                                      "detection_result": 0, "confidence_score": 0.0, "valid_signal_ratio": 0.0,
                                      "frequency_drift_hz": 0.0, "smoothness_score": 0.0,
                                      "satellite_name": "METEOR-M2-3", "norad_id": 57166,
                                      "waterfall_image_path": str(img), "wf_ml_confidence_score": 0.4}))
    out = tmp_path / "station.csv"
    answers = iter(["1", "0", "s"])
    assert station_labels.main(["--db", str(db), "--output", str(out), "--no-open"],
                               input_fn=lambda _: next(answers)) == 0
    df = pd.read_csv(out)
    assert list(df.label) == [1, 0] and set(df.capture_id) == {f"station_{ids[0]}", f"station_{ids[1]}"}
    assert all(c in df.columns for c in WATERFALL_FEATURE_NAMES) and (df.recording_id == df.capture_id).all()
    # already-labelled captures aren't offered again; the skipped one is
    todo = station_labels.pending(db, out, limit=100, include_simulated=False)
    assert [r["id"] for r in todo] == [ids[2]]


# --------------------------------------------------------------------------- RSP-03 builder

def test_rsp03_builder_makes_standard_waterfall_rows(tmp_path):
    fs = 1_000_000
    n = fs  # one 1-second window
    t = np.arange(2 * n) / fs
    x = 0.3 * np.exp(2j * np.pi * 2000 * t) + 0.05 * (np.random.default_rng(0).standard_normal(2 * n)
                                                       + 1j * np.random.default_rng(1).standard_normal(2 * n))
    raw = np.empty(4 * n, dtype="<i2")
    raw[0::2] = np.clip(x.real * 32767, -32768, 32767)
    raw[1::2] = np.clip(x.imag * 32767, -32768, 32767)
    (tmp_path / "Data").mkdir()
    raw.tofile(tmp_path / "Data" / "cap.raw")
    iq = tmp_path / "iq.csv"
    pd.DataFrame([{"capture_id": "w0", "label": 1, "recording_id": "r1",
                   "source_file": f"Data/cap.raw#samples=0-{n}"}]).to_csv(iq, index=False)
    df = rsp03.build(iq, tmp_path)
    assert len(df) == 1 and df.loc[0, "label"] == 1
    assert df.loc[0, "wf_centre_active_frac"] > 0.8 and df.loc[0, "wf_vert_coherence"] > 0.1
