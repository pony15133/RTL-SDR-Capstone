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
        assert "Satellite Station" in page and "__STATION_TOKEN__" not in page
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


# --------------------------------------------------------------------------- web interface API

def _serve(state):
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _req(url, body=None, token=None, host=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Station-Token"] = token
    if host:
        headers["Host"] = host
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=headers,
                                 method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw.decode()


def _uncertain_capture(st, tmp_path):
    iq = tmp_path / "rec" / "uncertain" / "m.iq"
    iq.parent.mkdir(parents=True)
    iq.write_bytes(b"\x80" * 32)
    return insert_result(st.db, {"input_file": str(iq), "timestamp_utc": "2026-09-24T10:00:00+00:00",
                                 "detection_result": 0, "confidence_score": 0.1, "valid_signal_ratio": 0.1,
                                 "frequency_drift_hz": 0.0, "smoothness_score": 1.0, "satellite_name": "METEOR-M2-3",
                                 "recording_status": "SUCCESS", "detection_verdict": "uncertain",
                                 "review_status": "pending", "raw_iq_file_path": str(iq)})


def test_review_and_label_through_the_api(tmp_path, monkeypatch):
    import review

    monkeypatch.setattr(review, "STATION_DATASET", tmp_path / "station.csv")
    st = _state(tmp_path)
    cid = _uncertain_capture(st, tmp_path)
    server, base = _serve(st)
    try:
        code, d = _req(base + "/api/review")
        assert code == 200 and [c["id"] for c in d["pending"]] == [cid]
        code, d = _req(base + f"/api/capture/{cid}")
        assert d["capture"]["detection_verdict"] == "uncertain" and d["iq_exists"]
        code, d = _req(base + f"/api/capture/{cid}/label", {"label": 1})      # no token -> refused
        assert code == 403
        code, d = _req(base + f"/api/capture/{cid}/label", {"label": 1, "reviewer": "anh"}, token=st.token)
        assert code == 200 and d["iq_action"].startswith("kept")
        assert (tmp_path / "rec" / "m.iq").exists()
        assert _req(base + "/api/review")[1]["pending"] == []
        assert st.totals()["pending_review"] == 0
    finally:
        server.shutdown()


def test_foreign_host_is_refused(tmp_path):
    st = _state(tmp_path)
    server, base = _serve(st)
    try:
        assert _req(base + "/api/status", host="evil.example")[0] == 403
        assert _req(base + "/api/status", host="localhost:8050")[0] == 200
    finally:
        server.shutdown()


def test_settings_are_validated_and_saved_with_backup(tmp_path):
    cfg_path = tmp_path / "capture_config.json"
    cfg = json.loads((ROOT / "capture_config.example.json").read_text())
    cfg_path.write_text(json.dumps(cfg))
    (tmp_path / "rec").mkdir()
    st = dashboard.DashboardState(cfg, tmp_path / "c.sqlite3", ROOT, cfg_path)
    server, base = _serve(st)
    try:
        bad = json.loads(json.dumps(cfg))
        bad["station"]["lat_deg"] = 123
        bad["recording"]["uncertain_band"] = [0.8, 0.2]
        bad["satellites"][0]["frequency_hz"] = 5e9
        code, d = _req(base + "/api/config", {"config": bad}, token=st.token)
        assert code == 400 and len(d["problems"]) == 3
        good = json.loads(json.dumps(cfg))
        good["recording"]["retention"] = "delete-negatives"
        good["recording"]["uncertain_band"] = [0.35, 0.65]
        code, d = _req(base + "/api/config", {"config": good}, token=st.token)
        assert code == 200 and d["ok"]
        saved = json.loads(cfg_path.read_text())
        assert saved["recording"]["retention"] == "delete-negatives" and saved["_readme"] == cfg["_readme"]
        assert json.loads(cfg_path.with_suffix(".json.bak").read_text())["recording"]["retention"] == "archive-negatives"
    finally:
        server.shutdown()


def test_files_are_served_only_from_project_folders(tmp_path):
    st = _state(tmp_path)
    iq = tmp_path / "rec" / "a.iq"
    iq.write_bytes(b"abc")
    server, base = _serve(st)
    try:
        from urllib.parse import quote

        assert urllib.request.urlopen(base + "/file?path=" + quote(str(iq))).read() == b"abc"
        assert _req(base + "/file?path=/etc/passwd")[0] == 404
        assert _req(base + "/image?path=" + quote(str(iq)))[0] == 404          # not an image type
    finally:
        server.shutdown()


def test_health_and_pages_render(tmp_path):
    st = _state(tmp_path)
    h = st.health()
    names = [c["name"] for c in h["checks"]]
    assert "rtl_sdr" in names and "SatDump (weather-satellite images)" in names and "Disk space" in names
    server, base = _serve(st)
    try:
        for path in ("/app.js", "/style.css", "/api/passes", "/api/images", "/api/captures?verdict=detected", "/api/health"):
            assert urllib.request.urlopen(base + path).status == 200
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- robustness (handover round)

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
    assert snap["captures"] == [] and not snap["totals"].get("n") and snap["totals"]["pending_review"] == 0


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
