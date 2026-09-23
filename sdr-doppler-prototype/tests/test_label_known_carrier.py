"""Tests for scripts/label_known_carrier.py using a small SYNTHETIC ci16
capture: a carrier that is on for alternating seconds, in noise."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("label_known_carrier", ROOT / "scripts" / "label_known_carrier.py")
lkc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lkc)

FS = 100_000


def _write_ci16_burst_file(path: Path, seconds: int = 6, carrier_hz: float = 2_000.0, seed: int = 0) -> np.ndarray:
    """Carrier ON in odd seconds, OFF in even seconds. Returns the truth per second."""
    rng = np.random.default_rng(seed)
    n = seconds * FS
    t = np.arange(n) / FS
    x = 0.02 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    truth = np.array([s % 2 for s in range(seconds)])
    gate = np.repeat(truth, FS).astype(float)
    x = x + gate * 0.3 * np.exp(2j * np.pi * carrier_hz * t)
    iq = np.empty(2 * n, dtype="<i2")
    iq[0::2] = np.clip(x.real * 32767, -32768, 32767)
    iq[1::2] = np.clip(x.imag * 32767, -32768, 32767)
    iq.tofile(path)
    return truth


def _run(tmp_path, capture, recording_id="rec_a", dataset=None):
    dataset = dataset or tmp_path / "features.csv"
    args = lkc.build_parser().parse_args([
        "--input", str(capture), "--dtype", "ci16", "--sample-rate", str(FS), "--center-freq", "437000000",
        "--recording-id", recording_id, "--dataset", str(dataset), "--carrier-offset-hz", "2000",
        "--search-khz", "3", "--nperseg", "256", "--noverlap", "128", "--label-nfft", "256",
    ])
    assert lkc.run(args) == 0
    return pd.read_csv(dataset)


def test_labels_match_where_the_carrier_is_on(tmp_path):
    capture = tmp_path / "burst.raw"
    truth = _write_ci16_burst_file(capture)
    df = _run(tmp_path, capture)

    assert len(df) == len(truth)
    assert df["label"].tolist() == truth.tolist()
    assert set(df["recording_id"]) == {"rec_a"}
    assert (df["is_synthetic"] == 0).all()
    assert df["source_file"].str.contains("#samples=").all()


def test_review_image_written_and_rerun_does_not_duplicate(tmp_path):
    capture = tmp_path / "burst.raw"
    _write_ci16_burst_file(capture)
    first = _run(tmp_path, capture)
    second = _run(tmp_path, capture)

    assert len(second) == len(first)
    assert list((tmp_path / "label_review").glob("*_labels.png"))


def test_output_is_trainable_with_recording_groups(tmp_path):
    from ml.train import resolve_groups

    dataset = tmp_path / "features.csv"
    for i in range(3):
        capture = tmp_path / f"burst{i}.raw"
        _write_ci16_burst_file(capture, seed=i)
        _run(tmp_path, capture, recording_id=f"rec_{i}", dataset=dataset)
    df = pd.read_csv(dataset)
    groups, source = resolve_groups(df)
    assert source == "recording_id"
    assert len(set(groups)) == 3


def test_load_iq_scales_each_dtype(tmp_path):
    ci16 = tmp_path / "a.raw"
    np.array([32767, -32768, 0, 16384], dtype="<i2").tofile(ci16)
    np.testing.assert_allclose(lkc.load_iq(ci16, "ci16"), [1 - 1j, 0.5j], atol=1e-3)

    cu8 = tmp_path / "b.iq"
    np.array([255, 0, 128, 127], dtype=np.uint8).tofile(cu8)
    np.testing.assert_allclose(lkc.load_iq(cu8, "cu8"), [1 - 1j, 0], atol=1e-2)
