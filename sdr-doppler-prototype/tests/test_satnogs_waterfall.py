"""Doppler correction, SatNOGS waterfall parsing, waterfall features and
the SatNOGS dataset downloader (network mocked)."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doppler import DopplerCurve, correct_iq, resize, standard_waterfall  # noqa: E402
from features.waterfall_features import WATERFALL_FEATURE_NAMES, waterfall_feature_array, waterfall_features  # noqa: E402
from waterfall_png import WaterfallParseError, png_to_waterfall  # noqa: E402

_spec = importlib.util.spec_from_file_location("fetch_satnogs", ROOT / "scripts" / "fetch_satnogs_dataset.py")
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def satnogs_like_png(signal: bool, seed: int = 0) -> bytes:
    """A waterfall PNG in the SatNOGS layout: tall, time upwards, viridis, colour bar right."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    wf = rng.normal(-80, 2.5, (500, 300))
    wf[:, :25] -= 12
    wf[:, -25:] -= 12                                       # filter roll-off at the edges
    if signal:
        for start in range(80, 420, 60):                   # bursty FM at the centre
            wf[start:start + 35, 140:160] += rng.uniform(12, 20)
    fig, ax = plt.subplots(figsize=(8.3, 16))
    im = ax.imshow(wf, aspect="auto", origin="lower", cmap="viridis", extent=[-24, 24, 0, 600])
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Time (seconds)")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- Doppler

def _drifting_tone(fs=200_000, dur=12.0, offset=20_000.0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    curve = DopplerCurve(np.linspace(0, dur, 121), 8000 * np.tanh(-(np.linspace(0, dur, 121) - dur / 2) / 2.5))
    phase = 2 * np.pi * np.cumsum(offset + curve(t)) / fs
    noise = 0.05 * (np.random.default_rng(0).standard_normal(n) + 1j * np.random.default_rng(1).standard_normal(n))
    return (0.3 * np.exp(1j * phase) + noise).astype(np.complex64), fs, offset, curve


def test_doppler_correction_centres_a_passing_signal():
    iq, fs, offset, curve = _drifting_tone()
    raw = standard_waterfall(iq, fs, offset_hz=offset)
    fixed = standard_waterfall(iq, fs, offset_hz=offset, doppler=curve)
    assert raw.shape == fixed.shape == (128, 128)
    assert np.argmax(raw, axis=1).std() > 10                    # drifts across the window
    assert np.all(np.abs(np.argmax(fixed, axis=1) - 64) <= 2)   # sits on the centre column


def test_correct_iq_continuous_phase_matches():
    iq, fs, offset, curve = _drifting_tone(dur=2.0)
    base = correct_iq(iq, fs, curve, offset_hz=offset)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(base[:65536])))
    freqs = np.fft.fftshift(np.fft.fftfreq(65536, 1 / fs))
    assert abs(freqs[np.argmax(spectrum)]) < 50                  # tone now at DC


def test_doppler_curve_interpolates():
    c = DopplerCurve(np.array([0.0, 10.0]), np.array([1000.0, -1000.0]))
    assert c(np.array([5.0]))[0] == pytest.approx(0.0)
    assert c.max_abs_hz == 1000.0


def test_resize_down_and_up():
    m = np.arange(64, dtype=float).reshape(8, 8)
    assert resize(m, 4, 4).shape == (4, 4)
    assert resize(m, 16, 16).shape == (16, 16)
    assert resize(m, 4, 4).mean() == pytest.approx(m.mean())


def test_too_low_sample_rate_rejected():
    with pytest.raises(ValueError):
        standard_waterfall(np.zeros(100_000, np.complex64), 30_000)


# --------------------------------------------------------------------------- PNG + features

def test_png_parser_recovers_signal_location():
    from matplotlib import image as mpimg

    good = png_to_waterfall(mpimg.imread(io.BytesIO(satnogs_like_png(True)), format="png"))
    bad = png_to_waterfall(mpimg.imread(io.BytesIO(satnogs_like_png(False)), format="png"))
    assert good.shape == bad.shape == (128, 128)
    profile = good.mean(axis=0)
    assert abs(int(np.argmax(profile[15:113])) + 15 - 64) <= 4   # burst column at the centre
    fg, fb = waterfall_features(good), waterfall_features(bad)
    assert fg["wf_centre_excess_z"] > fb["wf_centre_excess_z"] + 1
    assert fg["wf_centre_active_frac"] > fb["wf_centre_active_frac"]


def test_png_parser_rejects_non_waterfall():
    with pytest.raises(WaterfallParseError):
        png_to_waterfall(np.ones((200, 200, 3), dtype=np.float32))


def test_feature_array_order_and_finiteness():
    arr = waterfall_feature_array(np.random.default_rng(0).normal(size=(128, 128)))
    assert arr.shape == (len(WATERFALL_FEATURE_NAMES),) and np.all(np.isfinite(arr))


def test_our_corrected_recording_looks_like_a_good_satnogs_waterfall():
    """Features of a Doppler-corrected recording behave like SatNOGS 'good'."""
    iq, fs, offset, curve = _drifting_tone()
    fixed = waterfall_features(standard_waterfall(iq, fs, offset_hz=offset, doppler=curve))
    raw = waterfall_features(standard_waterfall(iq, fs, offset_hz=offset))
    assert fixed["wf_centre_active_frac"] > 0.9
    assert fixed["wf_centre_excess_z"] > raw["wf_centre_excess_z"]


# --------------------------------------------------------------------------- downloader (mocked network)

def _fake_api(n_good=6, n_bad=6):
    obs = []
    for i in range(n_good + n_bad):
        status = "good" if i < n_good else "bad"
        obs.append({"id": 1000 + i, "status": status, "norad_cat_id": 25544, "transmitter_mode": "FM",
                    "observation_frequency": 437800000, "max_altitude": 40.0, "station_name": f"ST{i % 4}",
                    "ground_station": i % 4, "start": "2026-09-23T10:00:00Z",
                    "waterfall": f"https://img.example/{1000 + i}.png"})
    pngs = {o["waterfall"]: satnogs_like_png(o["status"] == "good", seed=o["id"]) for o in obs}

    def http_get(url, **kw):
        if url.startswith(fetch.API):
            status = "good" if "status=good" in url else "bad"
            page = 1 if "page=" not in url else int(url.split("page=")[1].split("&")[0])
            items = [o for o in obs if o["status"] == status]
            chunk = items[(page - 1) * 4: page * 4]              # 4 per page, no Link header
            return json.dumps(chunk).encode(), {}
        return pngs[url], {}

    return http_get


def test_downloader_builds_trainable_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "http_get", _fake_api())
    out = tmp_path / "satnogs.csv"
    args = ["--output", str(out), "--arrays-dir", str(tmp_path / "wf"), "--per-class", "6",
            "--delay", "0", "--page-delay", "0", "--parse-checks", "2"]
    assert fetch.main(args) == 0

    df = pd.read_csv(out)
    assert len(df) == 12 and set(df["label"]) == {0, 1}
    assert df["recording_id"].str.startswith("station_").all()
    assert len(list((tmp_path / "wf").glob("*.npy"))) == 12
    assert len(list((tmp_path / "parse_check").glob("*.png"))) == 2
    good, bad = df[df.label == 1], df[df.label == 0]
    assert good["wf_centre_excess_z"].mean() > bad["wf_centre_excess_z"].mean()

    # resumable: a second run adds nothing
    assert fetch.main(args) == 0
    assert len(pd.read_csv(out)) == 12

    # and it trains with the waterfall feature set
    from ml.train import build_arg_parser, run_training

    res = run_training(build_arg_parser().parse_args([
        "--feature-set", "waterfall", "--dataset", str(out), "--output", str(tmp_path / "wf.joblib"),
        "--no-tune", "--cv-folds", "2"]))
    assert res.group_source == "recording_id" and res.model_path.exists()

    from detection.waterfall_detector import predict_waterfall

    m = np.load(next((tmp_path / "wf").glob("1000.npy"))).astype(np.float32)
    pred = predict_waterfall(res.model_path, m)
    assert pred.status == "AVAILABLE" and 0.0 <= pred.ml_confidence_score <= 1.0


def test_waterfall_model_missing_is_graceful(tmp_path):
    from detection.waterfall_detector import predict_waterfall

    assert predict_waterfall(tmp_path / "none.joblib", np.zeros((128, 128))).status == "MODEL_NOT_AVAILABLE"


def test_external_evaluation_on_committed_rsp03_waterfall_set(tmp_path):
    """The committed real RSP-03 waterfall features train/evaluate end to end."""
    from ml.train import build_arg_parser, run_training

    data = ROOT / "data" / "training" / "rsp03_waterfall_features.csv"
    df = pd.read_csv(data)
    assert len(df) == 111 and set(df["label"]) == {0, 1}
    res = run_training(build_arg_parser().parse_args([
        "--feature-set", "waterfall", "--dataset", str(data), "--output", str(tmp_path / "m.joblib"),
        "--no-tune", "--cv-folds", "2"]))
    spec = importlib.util.spec_from_file_location("evaluate_model", ROOT / "scripts" / "evaluate_model.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    m = ev.evaluate(res.model_path, data)
    assert m["n_samples"] == 111 and m["accuracy"] > 0.9
