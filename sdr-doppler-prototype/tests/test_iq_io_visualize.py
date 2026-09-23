"""Tests for src/iq_io.py (format decoding + metadata discovery),
src/visualize.py (standalone waterfall) and the GUI's Visualise IQ tab."""

import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iq_io import IQReader, normalise_format, parse_filename_metadata, probe  # noqa: E402
from load_data import load_input  # noqa: E402
from visualize import compute_waterfall  # noqa: E402

FS = 50_000


def _tone(n=FS, freq=5_000.0, amp=0.5):
    t = np.arange(n) / FS
    return (amp * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


def _write(path: Path, x: np.ndarray, fmt: str) -> Path:
    if fmt == "complex64":
        x.astype(np.complex64).tofile(path)
    elif fmt == "ci16":
        out = np.empty(2 * x.size, dtype="<i2")
        out[0::2] = np.round(x.real * 32767)
        out[1::2] = np.round(x.imag * 32767)
        out.tofile(path)
    elif fmt == "cu8":
        out = np.empty(2 * x.size, dtype=np.uint8)
        out[0::2] = np.clip(np.round(x.real * 127.5 + 127.5), 0, 255)
        out[1::2] = np.clip(np.round(x.imag * 127.5 + 127.5), 0, 255)
        out.tofile(path)
    elif fmt == "wav":
        from scipy.io import wavfile

        stereo = np.stack([x.real, x.imag], axis=1)
        wavfile.write(path, FS, np.round(stereo * 32767).astype(np.int16))
    return path


@pytest.mark.parametrize("fmt,suffix", [("complex64", ".bin"), ("ci16", ".raw"), ("cu8", ".iq"), ("wav", ".wav")])
def test_every_format_round_trips(tmp_path, fmt, suffix):
    x = _tone()
    path = _write(tmp_path / f"capture{suffix}", x, fmt)
    reader = IQReader(path, fmt)
    assert len(reader) == x.size
    got = reader[100:200]
    tol = 0.02 if fmt == "cu8" else 1e-3
    np.testing.assert_allclose(got, x[100:200], atol=tol)


def test_extension_defaults(tmp_path):
    assert probe(tmp_path / "a.iq").iq_format == "cu8"
    assert probe(tmp_path / "a.raw").iq_format == "ci16"
    assert probe(tmp_path / "a.bin").iq_format == "complex64"
    assert probe(tmp_path / "a.bin", iq_format="int16").iq_format == "ci16"


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError):
        normalise_format("complex128")


def test_filename_metadata():
    meta = parse_filename_metadata("rsp03_2026_02_13_09_39_30_436.950MHz_1.00Msps_ci16_le.chan1.sigmf-data")
    assert meta == {"sample_rate_hz": 1e6, "center_freq_hz": 436.95e6, "iq_format": "ci16"}
    assert parse_filename_metadata("SDRSharp_20200728_093524Z_1544500000Hz_NOAA-15.wav")["center_freq_hz"] == 1544.5e6


def test_sigmf_meta_wins_over_extension(tmp_path):
    data = _write(tmp_path / "pass.sigmf-data", _tone(), "ci16")
    (tmp_path / "pass.sigmf-meta").write_text(json.dumps({
        "global": {"core:datatype": "ci16_le", "core:sample_rate": FS},
        "captures": [{"core:sample_start": 0, "core:frequency": 437_000_000}],
    }))
    info = probe(data)
    assert (info.iq_format, info.sample_rate_hz, info.center_freq_hz) == ("ci16", FS, 437e6)
    assert info.source.startswith("sigmf:")


def test_recorder_sidecar_is_used_for_iq_files(tmp_path):
    data = _write(tmp_path / "METEOR_20260901_137900000Hz.iq", _tone(), "cu8")
    (tmp_path / "METEOR_20260901_137900000Hz.json").write_text(json.dumps({
        "satellite_name": "METEOR", "frequency_hz": 137_900_000, "sample_rate": FS,
    }))
    info = probe(data)
    assert (info.iq_format, info.sample_rate_hz, info.center_freq_hz) == ("cu8", FS, 137.9e6)


def test_load_input_decodes_rtl_sdr_iq_correctly(tmp_path):
    """Regression: .iq used to be read as complex64 regardless of content."""
    x = _tone()
    path = _write(tmp_path / "rtl.iq", x, "cu8")
    loaded = load_input(path)  # default "auto"
    assert loaded.kind == "iq" and loaded.values.size == x.size
    np.testing.assert_allclose(loaded.values[:50], x[:50], atol=0.02)


def test_waterfall_peak_is_at_the_tone(tmp_path):
    path = _write(tmp_path / "tone.raw", _tone(n=4 * FS, freq=5_000.0), "ci16")
    wf = compute_waterfall(IQReader(path, "ci16"), FS, center_freq_hz=437e6, nfft=512, rows=20)
    assert wf.power_db.shape == (20, 512)
    peak_hz = wf.freqs_hz[np.argmax(wf.power_db.mean(axis=0))]
    assert abs(peak_hz - (437e6 + 5_000)) < FS / 512 * 1.5
    assert wf.start_s == 0.0 and wf.end_s == pytest.approx(4.0, abs=0.02)


def test_waterfall_start_and_duration(tmp_path):
    path = _write(tmp_path / "tone.raw", _tone(n=4 * FS), "ci16")
    wf = compute_waterfall(IQReader(path, "ci16"), FS, nfft=256, rows=10, start_seconds=1.0, duration_seconds=2.0)
    assert wf.start_s == pytest.approx(1.0)
    assert wf.end_s == pytest.approx(3.0, abs=0.02)


def test_waterfall_span_too_short_raises(tmp_path):
    path = _write(tmp_path / "tiny.raw", _tone(n=100), "ci16")
    with pytest.raises(ValueError, match="fewer than"):
        compute_waterfall(IQReader(path, "ci16"), FS, nfft=1024)


def test_visualize_cli_writes_png(tmp_path):
    path = _write(tmp_path / "capture_437.000MHz_0.05Msps.raw", _tone(n=2 * FS), "ci16")
    out = tmp_path / "wf.png"
    proc = subprocess.run([sys.executable, str(ROOT / "src" / "visualize.py"), "--input", str(path),
                           "--output", str(out), "--nfft", "256"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 1000
    assert "center_freq=437000000.0" in proc.stdout  # picked up from the file name


def test_visualize_cli_refuses_unknown_sample_rate(tmp_path):
    path = _write(tmp_path / "mystery.bin", _tone(), "complex64")
    proc = subprocess.run([sys.executable, str(ROOT / "src" / "visualize.py"), "--input", str(path)],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "sample rate unknown" in (proc.stdout + proc.stderr)


def _import_gui_with_stub_tk(monkeypatch):
    """gui_app imports tkinter at module level; CI images often lack it."""
    class _Var:
        def __init__(self, value=None):
            self._v = value

        def get(self):
            return self._v

        def set(self, v):
            self._v = v

    tk = types.ModuleType("tkinter")
    tk.StringVar = tk.BooleanVar = _Var
    tk.Tk = tk.Toplevel = object
    tk.filedialog = types.SimpleNamespace(askopenfilename=lambda **k: "")
    tk.messagebox = types.SimpleNamespace(showerror=lambda *a, **k: None)
    tk.ttk = types.SimpleNamespace()
    scrolled = types.ModuleType("tkinter.scrolledtext")
    scrolled.ScrolledText = object
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.scrolledtext", scrolled)
    sys.path.insert(0, str(ROOT))
    sys.modules.pop("gui_app", None)
    import gui_app

    return gui_app, _Var


def test_gui_visualise_tab_builds_the_right_command(tmp_path, monkeypatch):
    gui_app, Var = _import_gui_with_stub_tk(monkeypatch)
    capture = _write(tmp_path / "rec.raw", _tone(), "ci16")
    captured = {}
    fake = SimpleNamespace(
        viz_input_var=Var(str(capture)), viz_format_var=Var("ci16"), viz_nfft_var=Var("2048"),
        viz_start_var=Var("1.5"), viz_sample_rate_var=Var("1000000"), viz_center_freq_var=Var(""),
        viz_duration_var=Var("10"), output_dir_var=Var(str(tmp_path / "results")),
        _run_command=lambda cmd, title: captured.update(cmd=cmd, title=title),
    )
    gui_app.SDRDopplerGUI._visualise_iq(fake)

    cmd = captured["cmd"]
    assert cmd[1] == "src/visualize.py"
    assert cmd[cmd.index("--input") + 1] == str(capture)
    assert cmd[cmd.index("--format") + 1] == "ci16"
    assert cmd[cmd.index("--sample-rate") + 1] == "1000000"
    assert cmd[cmd.index("--duration-seconds") + 1] == "10"
    assert "--center-freq" not in cmd  # blank = auto
    out = Path(cmd[cmd.index("--output") + 1])
    assert out.parent.name.startswith("waterfall_") and out.parent.is_dir()
