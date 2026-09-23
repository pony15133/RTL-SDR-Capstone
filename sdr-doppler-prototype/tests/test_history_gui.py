"""Capture-history CLI (src/history.py), the new database tables, and the
GUI's Capture History / Satellite Passes tabs (tkinter stubbed)."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import history  # noqa: E402
from database import (  # noqa: E402
    get_pass_positions,
    init_db,
    insert_pass_positions,
    insert_result,
    list_status,
    log_status,
)


def _row(**over):
    row = {"input_file": "x.iq", "timestamp_utc": "2026-09-24T12:00:00+00:00", "detection_result": 1,
           "confidence_score": 0.9, "valid_signal_ratio": 0.8, "frequency_drift_hz": 2000.0, "smoothness_score": 10.0,
           "satellite_name": "ISS", "frequency_hz": 145_800_000, "recording_status": "SUCCESS",
           "rule_detection_result": 1, "ml_detection_result": 1, "ml_confidence_score": 0.87, "iq_retention": "kept"}
    row.update(over)
    return row


def test_new_tables_and_helpers(tmp_path):
    db = tmp_path / "c.sqlite3"
    init_db(db)
    init_db(db)  # idempotent
    cid = insert_result(db, _row())
    n = insert_pass_positions(db, cid, [
        {"timestamp_utc": "2026-09-24T12:00:00", "azimuth_deg": 10, "elevation_deg": 5, "range_km": 2000, "doppler_hz": 3000},
        {"timestamp_utc": "2026-09-24T12:00:10", "azimuth_deg": 12, "elevation_deg": 7, "range_km": 1900, "doppler_hz": 2900},
    ])
    assert n == 2 and len(get_pass_positions(db, cid)) == 2
    log_status(db, "recorder", "RECORDING", "ISS")
    assert list_status(db)[0]["state"] == "RECORDING"


def test_history_cli_tables(tmp_path, capsys):
    db = tmp_path / "c.sqlite3"
    cid = insert_result(db, _row())
    insert_pass_positions(db, cid, [{"timestamp_utc": "2026-09-24T12:00:00", "azimuth_deg": 10, "elevation_deg": 5,
                                     "range_km": 2000, "doppler_hz": None}])
    log_status(db, "scheduler", "WAITING", "ISS pass at 12:00")
    assert history.main(["--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "ISS" in out and "145.800" in out and "WAITING" in out and "0.87" in out
    assert history.main(["--db", str(db), "--capture", str(cid)]) == 0
    detail = capsys.readouterr().out
    assert "Position history (1 points)" in detail and "satellite_name: ISS" in detail


def test_history_on_empty_database(tmp_path, capsys):
    history.main(["--db", str(tmp_path / "empty.sqlite3")])
    out = capsys.readouterr().out
    assert "(no captures yet)" in out and "(no status events yet)" in out


def _gui(monkeypatch):
    class Var:
        def __init__(self, value=None):
            self._v = value

        def get(self):
            return self._v

        def set(self, v):
            self._v = v

    tk = types.ModuleType("tkinter")
    tk.StringVar = tk.BooleanVar = Var
    tk.Tk = tk.Toplevel = object
    tk.filedialog = SimpleNamespace(askopenfilename=lambda **k: "")
    tk.messagebox = SimpleNamespace(showerror=lambda *a, **k: None)
    tk.ttk = SimpleNamespace()
    st = types.ModuleType("tkinter.scrolledtext")
    st.ScrolledText = object
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.scrolledtext", st)
    sys.path.insert(0, str(ROOT))
    sys.modules.pop("gui_app", None)
    import gui_app

    return gui_app, Var


def _fake_app(Var, tmp_path, calls, **vars_):
    base = dict(db_path_var=Var(str(tmp_path / "c.sqlite3")), history_capture_var=Var(""),
                passes_config_var=Var(str(ROOT.parent / "capture_config.example.json")),
                passes_hours_var=Var("12"), output_dir_var=Var(str(tmp_path / "results")))
    base.update(vars_)
    return SimpleNamespace(**base, _run_command=lambda cmd, title, **kw: calls.append((cmd, kw)))


def test_gui_capture_history_tab(tmp_path, monkeypatch):
    gui_app, Var = _gui(monkeypatch)
    calls = []
    app = _fake_app(Var, tmp_path, calls, history_capture_var=Var("7"))
    gui_app.SDRDopplerGUI._run_selected_function(app, "Capture History")
    cmd, kw = calls[0]
    assert cmd[1] == "src/history.py" and cmd[-2:] == ["--capture", "7"]
    assert kw["open_results"] is False


def test_gui_satellite_passes_tab(tmp_path, monkeypatch):
    gui_app, Var = _gui(monkeypatch)
    calls = []
    app = _fake_app(Var, tmp_path, calls)
    gui_app.SDRDopplerGUI._run_selected_function(app, "List Passes")
    gui_app.SDRDopplerGUI._run_selected_function(app, "Demo Capture")
    (list_cmd, list_kw), (demo_cmd, demo_kw) = calls
    assert list_cmd[1].endswith("auto_capture.py") and "--list-only" in list_cmd and "12" in list_cmd
    assert "--demo" in demo_cmd and demo_kw["open_results"] is True
    assert Path(list_kw["cwd"]) == ROOT.parent  # relative paths in the config resolve from the repo root
