"""METEOR decoding: baseband preparation and the SatDump hand-off (SatDump mocked)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import decode  # noqa: E402
from database import get_result, insert_pass_positions, insert_result  # noqa: E402

FS = 1_024_000


@pytest.fixture(autouse=True)
def _no_real_training_file(tmp_path, monkeypatch):
    import review

    monkeypatch.setattr(review, "STATION_DATASET", tmp_path / "station.csv")
OFFSET = 150_000.0


def _cu8_tone(path: Path, seconds=0.5, offset=OFFSET, drift_hz=0.0):
    n = int(FS * seconds)
    t = np.arange(n) / FS
    phase = 2 * np.pi * np.cumsum(offset + drift_hz * t / seconds) / FS
    x = 0.4 * np.exp(1j * phase) + 0.02 * (np.random.default_rng(0).standard_normal(n)
                                         + 1j * np.random.default_rng(1).standard_normal(n))
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(x.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(x.imag * 127.5 + 127.5), 0, 255)
    raw.tofile(path)
    return path


def _peak_hz(cs16: Path, rate: float) -> float:
    a = np.fromfile(cs16, dtype=np.int16).astype(np.float64)
    z = a[0::2] + 1j * a[1::2]
    spec = np.abs(np.fft.fftshift(np.fft.fft(z[: 1 << 16])))
    freqs = np.fft.fftshift(np.fft.fftfreq(1 << 16, 1 / rate))
    return float(freqs[int(np.argmax(spec))])


def test_baseband_puts_the_downlink_at_zero_and_decimates(tmp_path):
    iq = _cu8_tone(tmp_path / "pass.iq")
    rate = decode.prepare_baseband(iq, FS, tmp_path / "bb.cs16", offset_hz=OFFSET)
    assert rate == FS / 4
    n_out = (tmp_path / "bb.cs16").stat().st_size // 4
    assert abs(n_out - FS * 0.5 / 4) <= 2              # decimation phase carried across blocks
    assert abs(_peak_hz(tmp_path / "bb.cs16", rate)) < 50


def test_baseband_removes_doppler(tmp_path):
    iq = _cu8_tone(tmp_path / "pass.iq", seconds=1.0, drift_hz=3000.0)   # 150 kHz -> 153 kHz over 1 s
    curve = lambda t: 3000.0 * t / 1.0  # noqa: E731
    rate = decode.prepare_baseband(iq, FS, tmp_path / "bb.cs16", offset_hz=OFFSET, doppler=curve)
    a = np.fromfile(tmp_path / "bb.cs16", dtype=np.int16).astype(np.float64)
    z = a[0::2] + 1j * a[1::2]
    tail = z[-(1 << 15):]                                # end of the pass, where drift is largest
    spec = np.abs(np.fft.fftshift(np.fft.fft(tail)))
    freqs = np.fft.fftshift(np.fft.fftfreq(tail.size, 1 / rate))
    assert abs(freqs[int(np.argmax(spec))]) < 60


def _row(tmp_path, name="METEOR-M2-3", review="pending"):
    folder = tmp_path / "rec" / ("uncertain" if review else "")
    folder.mkdir(parents=True, exist_ok=True)
    iq = _cu8_tone(folder / "m.iq")
    db = tmp_path / "c.sqlite3"
    cid = insert_result(db, {"input_file": str(iq), "timestamp_utc": "2026-09-25T00:00:00Z", "detection_result": 0,
                             "confidence_score": 0.0, "valid_signal_ratio": 0.0, "frequency_drift_hz": 0.0,
                             "smoothness_score": 0.0, "satellite_name": name, "raw_iq_file_path": str(iq),
                             "sample_rate": FS, "frequency_hz": 137_750_000, "target_frequency_hz": 137_900_000,
                             "actual_recording_start": "2026-09-25T00:00:00+00:00",
                             "detection_verdict": "uncertain" if review else "detected", "review_status": review})
    insert_pass_positions(db, cid, [{"timestamp_utc": f"2026-09-25T00:00:{s:02d}+00:00", "doppler_hz": 3000 - 100 * s}
                                    for s in range(0, 30, 10)])
    return db, cid


def _fake_satdump(tmp_path, works_on="meteor_m2-x_lrpt_80k"):
    exe = tmp_path / "satdump.exe"
    exe.write_text("fake")
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        assert cmd[1:4] == ["pipeline", cmd[2], "baseband"] and "--baseband_format" in cmd
        out = Path(cmd[5])
        if cmd[2] == works_on:
            (out / "MSU-MR").mkdir(parents=True, exist_ok=True)
            (out / "MSU-MR" / "msu_mr_rgb_221.png").write_bytes(b"png")
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    return str(exe), runner, calls


def test_decode_tries_72k_then_80k_and_confirms_an_uncertain_capture(tmp_path):
    db, cid = _row(tmp_path)
    exe, runner, calls = _fake_satdump(tmp_path)
    r = decode.decode_capture(cid, db_path=db, results_dir=tmp_path / "res", satdump_path=exe, runner=runner)
    assert [c[2] for c in calls] == ["meteor_m2-x_lrpt", "meteor_m2-x_lrpt_80k"]
    assert r["status"] == "decoded: 1 image(s)" and r["images"][0].endswith("msu_mr_rgb_221.png")
    row = get_result(db, cid)
    assert row["decode_status"] == "decoded: 1 image(s)" and Path(row["decoded_image_dir"]).is_dir()
    assert row["review_status"] == "signal" and row["reviewed_by"] == "satdump-decode"
    assert not list((tmp_path / "res").rglob("*.cs16"))            # temporary baseband removed
    assert (Path(row["decoded_image_dir"]) / "decode.log").exists()


def test_decode_statuses_without_satdump_or_images(tmp_path, monkeypatch):
    db, cid = _row(tmp_path)
    monkeypatch.setattr(decode, "find_satdump", lambda explicit=None: None)
    assert decode.decode_capture(cid, db_path=db)["status"] == decode.STATUS_NO_SATDUMP
    monkeypatch.undo()
    exe, runner, _ = _fake_satdump(tmp_path, works_on="nothing")
    r = decode.decode_capture(cid, db_path=db, results_dir=tmp_path / "res", satdump_path=exe, runner=runner)
    assert r["status"].startswith("no images") and get_result(db, cid)["review_status"] == "pending"


def test_non_meteor_is_skipped(tmp_path):
    db, cid = _row(tmp_path, name="ISS")
    assert decode.decode_capture(cid, db_path=db)["status"] == decode.STATUS_NOT_METEOR


def test_doppler_from_track():
    track = [{"timestamp_utc": "2026-09-25T00:00:10+00:00", "doppler_hz": 1000.0},
             {"timestamp_utc": "2026-09-25T00:00:20+00:00", "doppler_hz": -1000.0}]
    curve = decode.doppler_from_track(track, "2026-09-25T00:00:00+00:00")
    assert curve(np.array([15.0]))[0] == pytest.approx(0.0)
    assert decode.doppler_from_track(track[:1], "2026-09-25T00:00:00+00:00") is None
