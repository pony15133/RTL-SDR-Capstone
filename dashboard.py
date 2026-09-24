#!/usr/bin/env python3
"""Station web interface - open http://localhost:8050 (run_station starts it).

    python dashboard.py --config capture_config.json        # same config as auto_capture
    python dashboard.py --config capture_config.json --port 9000

Pages (web/index.html + web/app.js, no internet or extra packages needed):
  Overview   what the station is doing now, next pass, results, warnings
  Passes     upcoming passes with countdowns and the sky track
  Captures   every recording with its verdict; open one for all details,
             waterfall, spectrogram, pass track, Doppler and decoded images
  Review     the "uncertain" queue: look at each waterfall, press Signal / Noise
  Images     decoded METEOR pictures
  Settings   station location, satellites, retention, uncertain band, decoding
  Health     checks (dongle tools, SatDump, models, disk), status log, log file

Actions (labelling, decoding, saving settings) only work from this computer:
the server listens on 127.0.0.1, checks the Host header and requires a
per-run token that only the page itself can read.
Standard library only (no Flask).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import shutil
import sqlite3
import sys
import threading
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parent
WEB = REPO / "web"
sys.path.insert(0, str(REPO / "sdr-doppler-prototype" / "src"))
sys.path.insert(0, str(REPO / "iq-recorder"))

from config import DB_PATH, MODELS_DIR  # noqa: E402

IMAGE_TYPES = {".png", ".jpg", ".jpeg"}
VERDICT_ORDER = ("detected", "uncertain", "not_detected")


def _rows(db: Path, sql: str, params=()) -> list:
    if not db.exists():
        return []
    try:
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params)]
    except sqlite3.OperationalError:
        return []  # table/column not created yet


def _folder_size(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


def _model_info(path: Path) -> dict:
    meta = path.with_suffix(".json")
    if not path.exists():
        return {"available": False, "path": str(path)}
    info = {"available": True, "path": str(path)}
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
        info.update(version=m.get("model_version"), trained=m.get("training_timestamp"),
                    test_f1=(m.get("evaluation_metrics") or {}).get("f1_score"),
                    test_auc=(m.get("evaluation_metrics") or {}).get("roc_auc"),
                    threshold=m.get("decision_threshold"),
                    n_samples=m.get("n_total_samples"), synthetic=m.get("trained_on_synthetic_data"))
    except (OSError, ValueError):
        pass
    return info


class DashboardState:
    def __init__(self, config: dict, db: Path, config_dir: Path, config_path=None):
        rec = config.get("recording", {})
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self.db = db
        base = config_dir
        self.base = base
        self.output_dir = (base / rec.get("output_dir", "recordings")).resolve()
        self.ml_model = (base / rec.get("ml_model", str(MODELS_DIR / "random_forest.joblib"))).resolve()
        self.wf_model = (base / rec.get("waterfall_model", str(MODELS_DIR / "waterfall_rf.joblib"))).resolve()
        self.tle_cache = (base / config.get("tle_cache_dir", rec.get("tle_cache_dir", "tle_cache"))).resolve()
        self.log_dir = (base / rec.get("log_dir", "logs")).resolve()
        self.allowed_roots = [REPO.resolve(), self.output_dir, db.parent.resolve()]
        self.token = secrets.token_urlsafe(24)
        self.decoding: set = set()

    # ------------------------------------------------------------------ status
    def live(self) -> dict:
        path = self.output_dir / "live_status.json"
        if not path.exists():
            return {"phase": "not running", "note": "The capture loop isn't running - start run_station."}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"phase": "unknown"}
        updated = datetime.fromisoformat(data["updated_utc"])
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        data["heartbeat_age_s"] = round(age)
        if data.get("recorder_state") == "RECORDING":
            data["phase"] = "recording"
        if age > 60 and data.get("phase") != "stopped":
            data["phase"] = "stalled?"
            data["note"] = f"No heartbeat for {age:.0f} s - is the capture loop still running?"
        return data

    def sky_track(self, live: dict) -> dict:
        """Track of the pass being waited for / recorded, else the last recorded pass."""
        cur = (live.get("current") or (live.get("upcoming") or [None])[0]) if isinstance(live, dict) else None
        if cur and cur.get("norad_id"):
            pts = self.predicted_track(cur, live.get("station"))
            if pts:
                return {"label": f"{'Now' if live.get('current') else 'Next'}: {cur['satellite']}", "points": pts}
        last = _rows(self.db, "SELECT capture_id, satellite_name FROM pass_positions ORDER BY id DESC LIMIT 1")
        if last:
            pts = _rows(self.db, "SELECT azimuth_deg, elevation_deg FROM pass_positions WHERE capture_id=? "
                                 "ORDER BY timestamp_utc", (last[0]["capture_id"],))
            return {"label": f"Last recorded: {last[0]['satellite_name']}",
                    "points": [[p["azimuth_deg"], p["elevation_deg"]] for p in pts]}
        return {"label": "no pass track yet", "points": []}

    def predicted_track(self, p: dict, station: dict = None) -> list:
        station = station or self.config.get("station") or {}
        tle_file = self.tle_cache / f"{p.get('norad_id')}.tle"
        if not tle_file.exists() or "lat_deg" not in station:
            return []
        try:
            from rtl_recorder.passes import GroundStation, load_tle_file, make_propagator, pass_track

            track = pass_track(make_propagator(load_tle_file(tle_file, p["norad_id"])),
                               GroundStation(station["lat_deg"], station["lon_deg"], station.get("alt_m", 0)),
                               datetime.fromisoformat(p["aos"]), datetime.fromisoformat(p["los"]), step_seconds=15)
            return [[q["azimuth_deg"], q["elevation_deg"]] for q in track]
        except Exception:  # never break the page over a plot
            return []

    def totals(self) -> dict:
        t = (_rows(self.db, """SELECT COUNT(*) AS n,
                 SUM(CASE WHEN recording_status='SUCCESS' THEN 1 ELSE 0 END) AS ok,
                 SUM(CASE WHEN recording_status IS NOT NULL AND recording_status<>'SUCCESS' THEN 1 ELSE 0 END) AS failed,
                 SUM(CASE WHEN COALESCE(wf_ml_detection_result, ml_detection_result, rule_detection_result)=1
                          THEN 1 ELSE 0 END) AS detected_any,
                 SUM(CASE WHEN iq_retention LIKE 'kept%' THEN 1 ELSE 0 END) AS kept,
                 SUM(CASE WHEN iq_retention LIKE 'archived%' THEN 1 ELSE 0 END) AS archived,
                 SUM(CASE WHEN iq_retention LIKE 'deleted%' THEN 1 ELSE 0 END) AS deleted
                 FROM capture_results""") or [{}])[0]
        t["detected"] = t.pop("detected_any", 0)
        verdicts = {r["v"]: r["n"] for r in _rows(
            self.db, "SELECT detection_verdict AS v, COUNT(*) AS n FROM capture_results "
                     "WHERE detection_verdict IS NOT NULL GROUP BY 1")}
        t["verdicts"] = {v: verdicts.get(v, 0) for v in VERDICT_ORDER}
        t["pending_review"] = (_rows(self.db, "SELECT COUNT(*) AS n FROM capture_results "
                                              "WHERE review_status='pending'") or [{"n": 0}])[0]["n"]
        t["decoded"] = (_rows(self.db, "SELECT COUNT(*) AS n FROM capture_results "
                                       "WHERE decode_status LIKE 'decoded%'") or [{"n": 0}])[0]["n"]
        return t

    def snapshot(self) -> dict:
        live = self.live()
        captures = self.captures(limit=15)
        status_log = _rows(self.db, "SELECT timestamp_utc, component, state, message FROM status_log "
                                    "ORDER BY id DESC LIMIT 12")
        disk_dir = self.output_dir if self.output_dir.exists() else REPO
        usage = shutil.disk_usage(disk_dir)
        latest_wf = next((c.get("waterfall_image_path") or c.get("spectrogram_image_path")
                          for c in captures if c.get("waterfall_image_path") or c.get("spectrogram_image_path")), None)
        return {
            "now_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "live": live,
            "sky": self.sky_track(live),
            "captures": captures,
            "totals": self.totals(),
            "status_log": status_log,
            "latest_image": latest_wf,
            "storage": {"free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1),
                        "recordings_gb": round(_folder_size(self.output_dir) / 1e9, 3),
                        "uncertain_gb": round(_folder_size(self.output_dir / "uncertain") / 1e9, 3),
                        "archived_gb": round(_folder_size(self.output_dir / "rejected") / 1e9, 3)},
            "models": {"iq": _model_info(self.ml_model), "waterfall": _model_info(self.wf_model)},
            "warnings": self.warnings(live, usage.free),
            "db": str(self.db),
        }

    def warnings(self, live: dict, free_bytes: int) -> list:
        w = []
        min_free = float(self.config.get("recording", {}).get("min_free_gb", 2))
        if free_bytes / 1e9 < max(10.0, 2 * min_free):
            w.append({"level": "bad" if free_bytes / 1e9 < min_free else "warn",
                      "text": f"Low disk space: {free_bytes / 1e9:.1f} GB free.", "link": "#/health"})
        if live.get("phase") == "stalled?":
            w.append({"level": "bad", "text": live.get("note"), "link": "#/health"})
        elif live.get("phase") == "not running":
            w.append({"level": "warn", "text": "The capture loop isn't running. Start run_station.", "link": "#/health"})
        if not self.wf_model.exists():
            w.append({"level": "warn", "text": "The waterfall model isn't trained yet - run setup --fetch-satnogs.",
                      "link": "#/health"})
        pending = self.totals().get("pending_review", 0)
        if pending:
            w.append({"level": "info", "text": f"{pending} capture(s) need your review.", "link": "#/review"})
        return w

    # ------------------------------------------------------------------ captures
    def captures(self, *, limit=50, offset=0, verdict=None, satellite=None, review=None) -> list:
        where, args = [], []
        for col, val in (("detection_verdict", verdict), ("satellite_name", satellite), ("review_status", review)):
            if val:
                where.append(f"{col} = ?")
                args.append(val)
        sql = ("SELECT * FROM capture_results" + (" WHERE " + " AND ".join(where) if where else "")
               + " ORDER BY id DESC LIMIT ? OFFSET ?")
        return _rows(self.db, sql, (*args, int(limit), int(offset)))

    def satellites(self) -> list:
        return [r["s"] for r in _rows(self.db, "SELECT DISTINCT satellite_name AS s FROM capture_results "
                                               "WHERE satellite_name IS NOT NULL ORDER BY 1")]

    def capture(self, cid: int) -> dict:
        rows = _rows(self.db, "SELECT * FROM capture_results WHERE id = ?", (int(cid),))
        if not rows:
            return {}
        row = rows[0]
        track = _rows(self.db, "SELECT timestamp_utc, azimuth_deg, elevation_deg, range_km, doppler_hz "
                               "FROM pass_positions WHERE capture_id = ? ORDER BY timestamp_utc", (int(cid),))
        images = []
        if row.get("decoded_image_dir") and Path(row["decoded_image_dir"]).is_dir():
            images = [str(p) for p in sorted(Path(row["decoded_image_dir"]).rglob("*"))
                      if p.suffix.lower() in IMAGE_TYPES][:60]
        iq = row.get("raw_iq_file_path")
        return {"capture": row, "track": track, "decoded_images": images,
                "iq_exists": bool(iq and Path(iq).exists()),
                "iq_size": Path(iq).stat().st_size if iq and Path(iq).exists() else None,
                "decoding": int(cid) in self.decoding}

    def gallery(self, limit=60) -> list:
        out = []
        for r in _rows(self.db, "SELECT id, satellite_name, timestamp_utc, decoded_image_dir FROM capture_results "
                                "WHERE decode_status LIKE 'decoded%' ORDER BY id DESC LIMIT ?", (int(limit),)):
            d = Path(r["decoded_image_dir"] or "")
            imgs = [str(p) for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_TYPES] if d.is_dir() else []
            if imgs:
                out.append({**r, "images": imgs})
        return out

    # ------------------------------------------------------------------ actions
    def retention_policy(self) -> str:
        return self.config.get("recording", {}).get("retention", "archive-negatives")

    def label(self, cid: int, label: int, reviewer: str) -> dict:
        import review

        return review.resolve(int(cid), int(label), db_path=self.db, reviewer=reviewer or "web",
                              policy=self.retention_policy())

    def decode(self, cid: int) -> dict:
        import decode

        if int(cid) in self.decoding:
            return {"status": "already decoding"}
        sat = self.config.get("recording", {}).get("satdump_path")
        if decode.find_satdump(sat) is None:
            return {"status": decode.STATUS_NO_SATDUMP}
        self.decoding.add(int(cid))

        def work():
            try:
                decode.decode_capture(int(cid), db_path=self.db, satdump_path=sat, force=True)
            finally:
                self.decoding.discard(int(cid))

        threading.Thread(target=work, daemon=True).start()
        return {"status": "decoding started"}

    # ------------------------------------------------------------------ settings / health
    def read_config(self) -> dict:
        if self.config_path and self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return self.config

    def save_config(self, new: dict) -> dict:
        problems = validate_config(new)
        if problems:
            return {"ok": False, "problems": problems}
        if not self.config_path:
            return {"ok": False, "problems": ["The web page was started without --config, so there is no file to save."]}
        if self.config_path.exists():
            shutil.copy(self.config_path, self.config_path.with_suffix(".json.bak"))
        tmp = self.config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(new, indent=2), encoding="utf-8")
        tmp.replace(self.config_path)
        self.config = new
        return {"ok": True, "message": "Saved. Restart run_station for the capture loop to use the new settings.",
                "backup": str(self.config_path.with_suffix(".json.bak"))}

    def health(self) -> dict:
        import decode

        checks = []
        rec = self.config.get("recording", {})
        from rtl_recorder.utils import find_executable

        for tool in ("rtl_sdr", "rtl_test"):
            try:
                path = find_executable(tool, rec.get("rtl_sdr_path") if tool == "rtl_sdr" else None)
                checks.append({"name": tool, "ok": True, "detail": path})
            except Exception:
                checks.append({"name": tool, "ok": False,
                               "detail": "not found - install the RTL-SDR tools (see QUICKSTART.md), "
                                         "or set RTL_SDR_HOME"})
        sd = decode.find_satdump(rec.get("satdump_path"))
        checks.append({"name": "SatDump (METEOR images)", "ok": bool(sd),
                       "detail": sd or "not installed - optional, from https://www.satdump.org"})
        for key, path in (("IQ model", self.ml_model), ("Waterfall model", self.wf_model)):
            info = _model_info(path)
            checks.append({"name": key, "ok": info["available"],
                           "detail": (f"{info.get('version') or ''} - test F1 {info['test_f1']:.2f}"
                                      if info.get("test_f1") is not None else str(path)) if info["available"]
                           else "not trained - run setup"})
        usage = shutil.disk_usage(self.output_dir if self.output_dir.exists() else REPO)
        min_free = float(rec.get("min_free_gb", 2))
        checks.append({"name": "Disk space", "ok": usage.free / 1e9 >= max(10.0, 2 * min_free),
                       "detail": f"{usage.free / 1e9:.1f} GB free of {usage.total / 1e9:.0f} GB"})
        tles = sorted(self.tle_cache.glob("*.tle")) if self.tle_cache.exists() else []
        if tles:
            age_d = (datetime.now().timestamp() - min(p.stat().st_mtime for p in tles)) / 86400
            checks.append({"name": "Orbit data (TLE)", "ok": age_d < 7, "detail": f"oldest {age_d:.1f} days old"})
        live = self.live()
        checks.append({"name": "Capture loop", "ok": live.get("phase") not in ("not running", "stalled?", "unknown"),
                       "detail": live.get("note") or f"{live.get('phase')} (heartbeat {live.get('heartbeat_age_s')} s ago)"})
        return {"checks": checks,
                "status_log": _rows(self.db, "SELECT timestamp_utc, component, state, message FROM status_log "
                                             "ORDER BY id DESC LIMIT 200"),
                "log_tail": self.log_tail()}

    def log_tail(self, lines: int = 200) -> str:
        path = self.log_dir / "station.log"
        if not path.exists():
            return ""
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64_000))
            text = f.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def file_allowed(self, path: Path, types=IMAGE_TYPES) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return (types is None or resolved.suffix.lower() in types) and resolved.is_file() and any(
            root == resolved or root in resolved.parents for root in self.allowed_roots)

    # kept for older callers/tests
    def image_allowed(self, path: Path) -> bool:
        return self.file_allowed(path, {".png"})


def validate_config(cfg: dict) -> list:
    """Human-readable problems with a config the Settings page wants to save."""
    from retention import POLICIES

    p = []
    st = cfg.get("station") or {}
    try:
        lat, lon = float(st.get("lat_deg")), float(st.get("lon_deg"))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            p.append("Latitude must be -90..90 and longitude -180..180.")
    except (TypeError, ValueError):
        p.append("Station latitude and longitude are required numbers.")
    rec = cfg.get("recording") or {}
    if rec.get("retention") not in POLICIES:
        p.append(f"Retention must be one of {', '.join(POLICIES)}.")
    band = rec.get("uncertain_band")
    if band is not None:
        try:
            low, high = float(band[0]), float(band[1])
            if not 0 <= low <= high <= 1:
                p.append("Uncertain band must satisfy 0 <= low <= high <= 1.")
        except (TypeError, ValueError, IndexError):
            p.append("Uncertain band must be two numbers, e.g. [0.3, 0.7].")
    try:
        if float(rec.get("min_elevation_deg", 15)) < 0 or float(rec.get("min_elevation_deg", 15)) > 90:
            p.append("Minimum elevation must be between 0 and 90 degrees.")
    except (TypeError, ValueError):
        p.append("Minimum elevation must be a number.")
    sats = cfg.get("satellites") or []
    if not sats:
        p.append("Add at least one satellite.")
    for i, s in enumerate(sats, 1):
        try:
            int(s["norad_id"])
            f = float(s["frequency_hz"])
            if not 24e6 <= f <= 1.8e9:
                p.append(f"Satellite {i}: frequency must be within the RTL-SDR range (24 MHz - 1.8 GHz).")
        except (KeyError, TypeError, ValueError):
            p.append(f"Satellite {i}: name, NORAD id and frequency are required.")
        if not str(s.get("name", "")).strip():
            p.append(f"Satellite {i}: name is required.")
    return p


LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


def make_handler(state: DashboardState, *, allow_remote: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(code, json.dumps(data, default=str).encode(), "application/json")

        def _host_ok(self) -> bool:
            if allow_remote:
                return True
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0] if not (self.headers.get("Host") or "").startswith("[") \
                else (self.headers.get("Host") or "").split("]")[0] + "]"
            return host in LOCAL_HOSTS

        def _static(self, name: str):
            path = (WEB / name).resolve()
            if WEB.resolve() not in path.parents or not path.is_file():
                return self._send(404, b"not found", "text/plain")
            body = path.read_bytes()
            if name == "index.html":
                body = body.replace(b"__STATION_TOKEN__", state.token.encode())
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self._send(200, body, ctype)

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden host", "text/plain")
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            path = url.path
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path in ("/app.js", "/style.css"):
                return self._static(path[1:])
            if path == "/api/status":
                return self._json(state.snapshot())
            if path == "/api/captures":
                return self._json({"captures": state.captures(limit=min(int(q.get("limit", 50)), 500),
                                                             offset=int(q.get("offset", 0)),
                                                             verdict=q.get("verdict"), satellite=q.get("satellite"),
                                                             review=q.get("review")),
                                   "satellites": state.satellites()})
            if path.startswith("/api/capture/"):
                try:
                    data = state.capture(int(path.rsplit("/", 1)[1]))
                except ValueError:
                    data = {}
                return self._json(data) if data else self._json({"error": "no such capture"}, 404)
            if path == "/api/review":
                return self._json({"pending": state.captures(limit=200, review="pending")})
            if path == "/api/images":
                return self._json({"captures": state.gallery()})
            if path == "/api/passes":
                live = state.live()
                ups = live.get("upcoming") or []
                return self._json({"upcoming": [{**p, "track": state.predicted_track(p, live.get("station"))}
                                                for p in ups[:12]], "station": live.get("station")})
            if path == "/api/config":
                return self._json({"config": state.read_config(), "path": str(state.config_path or "")})
            if path == "/api/health":
                return self._json(state.health())
            if path in ("/image", "/file"):
                p = Path(q.get("path", ""))
                types = IMAGE_TYPES if path == "/image" else None
                if not state.file_allowed(p, types):
                    return self._send(404, b"not found", "text/plain")
                if path == "/image":
                    return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "image/png")
                return self._stream_file(p)
            return self._send(404, b"not found", "text/plain")

        def _stream_file(self, p: Path):
            size = p.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with p.open("rb") as f:
                shutil.copyfileobj(f, self.wfile, 1 << 20)

        def do_POST(self):
            if not self._host_ok():
                return self._send(403, b"forbidden host", "text/plain")
            if self.headers.get("X-Station-Token") != state.token:
                return self._json({"error": "missing or wrong token - reload the page"}, 403)
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1_000_000:
                return self._json({"error": "request too large"}, 413)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json({"error": "invalid JSON"}, 400)
            path = urlparse(self.path).path
            try:
                if path.startswith("/api/capture/") and path.endswith("/label"):
                    cid = int(path.split("/")[3])
                    return self._json(state.label(cid, int(body.get("label")), str(body.get("reviewer") or "web")))
                if path.startswith("/api/capture/") and path.endswith("/decode"):
                    return self._json(state.decode(int(path.split("/")[3])))
                if path == "/api/config":
                    res = state.save_config(body.get("config") or {})
                    return self._json(res, 200 if res["ok"] else 400)
            except (KeyError, ValueError) as exc:
                return self._json({"error": str(exc)}, 400)
            return self._send(404, b"not found", "text/plain")

    return Handler


def build_state(args) -> DashboardState:
    config, config_dir, config_path = {}, REPO, None
    if args.config:
        config_path = Path(args.config).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_dir = REPO  # auto_capture runs from the repo root, so config paths are repo-relative
    rec = config.get("recording", {})
    db = Path(args.db or rec.get("db_path") or config.get("db_path") or DB_PATH)
    if not db.is_absolute():
        db = (config_dir / db).resolve()
    if args.output_dir:
        config.setdefault("recording", {})["output_dir"] = args.output_dir
    return DashboardState(config, db, config_dir, config_path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Web interface for the capture station.")
    p.add_argument("--config", help="Same JSON config as auto_capture.py")
    p.add_argument("--db", help="SQLite database (default: from config, else sdr-doppler-prototype's)")
    p.add_argument("--output-dir", help="Recordings folder (where live_status.json is written)")
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 lets other computers on the LAN open it (they can then also label and change settings)")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args(argv)
    state = build_state(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, allow_remote=args.host != "127.0.0.1"))
    print(f"Station web page on http://localhost:{args.port}  (database {state.db}, recordings {state.output_dir})"
          " - Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
