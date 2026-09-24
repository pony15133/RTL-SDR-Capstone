"""dashboard.py: status snapshot, heartbeat handling, image path safety, HTTP endpoints."""

import json
import sys
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dashboard  # noqa: E402
from database import insert_result, log_status  # noqa: E402


def _state(tmp_path):
    cfg = {"station": {"name": "Test"}, "recording": {"output_dir": str(tmp_path / "rec")}}
    (tmp_path / "rec").mkdir()
    return dashboard.DashboardState(cfg, tmp_path / "c.sqlite3", ROOT)


def _heartbeat(tmp_path, **over):
    data = {"updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "phase": "waiting for pass",
            "recorder_state": "WAITING", "simulate": True, "current": None, "upcoming": [],
            "station": {"name": "Test", "lat_deg": 1.35, "lon_deg": 103.8, "alt_m": 15}}
    data.update(over)
    (tmp_path / "rec" / "live_status.json").write_text(json.dumps(data))


def test_snapshot_on_empty_database(tmp_path):
    snap = _state(tmp_path).snapshot()
    assert snap["captures"] == [] and snap["live"]["phase"] == "not running"
    assert snap["storage"]["free_gb"] > 0
    assert snap["models"]["waterfall"]["available"] in (True, False)


def test_snapshot_with_data_and_live_recording(tmp_path):
    st = _state(tmp_path)
    img = tmp_path / "rec" / "wf.png"
    img.write_bytes(b"\x89PNG fake")
    insert_result(st.db, {"input_file": "a.iq", "timestamp_utc": "2026-09-24T10:00:00+00:00", "detection_result": 0,
                          "confidence_score": 0.1, "valid_signal_ratio": 0.1, "frequency_drift_hz": 0.0,
                          "smoothness_score": 1.0, "satellite_name": "ISS", "recording_status": "SUCCESS",
                          "wf_ml_detection_result": 1, "wf_ml_confidence_score": 0.9, "iq_retention": "kept",
                          "waterfall_image_path": str(img)})
    log_status(st.db, "scheduler", "WAITING", "ISS pass")
    _heartbeat(tmp_path, recorder_state="RECORDING")
    snap = st.snapshot()
    assert snap["live"]["phase"] == "recording"
    assert snap["totals"]["n"] == 1 and snap["totals"]["detected"] == 1 and snap["totals"]["kept"] == 1
    assert snap["latest_image"] == str(img)
    assert snap["status_log"][0]["state"] == "WAITING"


def test_stale_heartbeat_is_flagged(tmp_path):
    st = _state(tmp_path)
    _heartbeat(tmp_path, updated_utc=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(timespec="seconds"))
    assert st.live()["phase"] == "stalled?"


def test_image_endpoint_only_serves_pngs_inside_project(tmp_path):
    st = _state(tmp_path)
    inside = tmp_path / "rec" / "ok.png"
    inside.write_bytes(b"x")
    outside = Path("/etc/passwd")
    assert st.image_allowed(inside)
    assert not st.image_allowed(outside)
    assert not st.image_allowed(tmp_path / "rec" / "missing.png")


def test_http_endpoints(tmp_path):
    st = _state(tmp_path)
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(st))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "Station Dashboard" in page
        status = json.loads(urllib.request.urlopen(base + "/api/status").read())
        assert "live" in status and "captures" in status
        try:
            urllib.request.urlopen(base + "/image?path=/etc/passwd")
            raise AssertionError("should be refused")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()


def test_auto_capture_writes_heartbeat(tmp_path):
    import auto_capture
    from rtl_recorder.recorder import RTLSDRRecorder
    from rtl_recorder.config import RecorderConfig

    cfg = json.loads((ROOT / "capture_config.example.json").read_text())
    cfg["recording"]["output_dir"] = str(tmp_path / "rec")
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    s = auto_capture.load_settings(auto_capture.build_parser().parse_args(["--config", str(tmp_path / "cfg.json"),
                                                                           "--simulate"]))
    live = auto_capture.LiveStatus(s, RTLSDRRecorder(RecorderConfig(simulate=True)), interval=0.05).start()
    live.set("planned")
    data = json.loads((tmp_path / "rec" / "live_status.json").read_text())
    live.stop()
    assert data["phase"] == "planned" and data["station"]["name"] == "Singapore campus"


def test_partially_migrated_database_still_shows_rows_and_totals(tmp_path):
    """A pre-ML database (original capture_results only, no status_log /
    pass_positions) must not break the read-only dashboard."""
    import sqlite3

    from database import SCHEMA

    st = _state(tmp_path)
    with sqlite3.connect(st.db) as conn:
        conn.execute(SCHEMA)
        conn.execute("INSERT INTO capture_results (input_file, timestamp_utc, detection_result, confidence_score, "
                     "valid_signal_ratio, frequency_drift_hz, smoothness_score) VALUES ('a.iq', '2026-01-01', 1, "
                     "0.9, 0.5, 2000, 100)")
    snap = st.snapshot()
    assert [c["input_file"] for c in snap["captures"]] == ["a.iq"]
    assert snap["totals"]["n"] == 1 and snap["totals"]["detected"] == 1
    assert snap["status_log"] == [] and snap["sky"]["points"] == []


def test_corrupt_database_file_does_not_crash(tmp_path):
    st = _state(tmp_path)
    st.db.write_bytes(b"not a database" * 100)
    snap = st.snapshot()
    assert snap["captures"] == [] and snap["totals"] == {}


def test_malformed_heartbeat_does_not_crash(tmp_path):
    st = _state(tmp_path)
    hb = tmp_path / "rec" / "live_status.json"
    for content in ('{"phase": "planned"}', '{"updated_utc": "yesterday"}', "[1, 2]", "", '{"updated_utc": "20'):
        hb.write_text(content)
        assert st.snapshot()["live"]["phase"] == "unknown"


def test_naive_heartbeat_timestamp_is_read_as_utc(tmp_path):
    st = _state(tmp_path)
    _heartbeat(tmp_path, updated_utc=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"))
    live = st.live()
    assert live["phase"] == "waiting for pass" and live["heartbeat_age_s"] < 60
