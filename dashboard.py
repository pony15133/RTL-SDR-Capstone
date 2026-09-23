#!/usr/bin/env python3
"""Live station dashboard - open http://localhost:8050 while auto_capture.py runs.

    python dashboard.py --config capture_config.json        # same config as auto_capture
    python dashboard.py --config capture_config.json --port 9000 --host 0.0.0.0   # reachable on the LAN

Shows, refreshed every 5 s:
  * what the station is doing now (waiting / recording / processing) and the
    recorder state, from the heartbeat auto_capture.py writes;
  * the next passes with countdowns;
  * the sky track (azimuth/elevation) of the next pass or the last recording;
  * recent captures with rule / IQ-model / waterfall-model results and what
    happened to each IQ file, plus the latest Doppler-corrected waterfall;
  * totals, detection rate, disk space and recording-folder size;
  * the station status log.

Standard library only (no Flask), read-only on the database.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "sdr-doppler-prototype" / "src"))
sys.path.insert(0, str(REPO / "iq-recorder"))

from config import DB_PATH, MODELS_DIR  # noqa: E402


def _rows(db: Path, sql: str, params=()) -> list:
    if not db.exists():
        return []
    try:
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params)]
    except sqlite3.OperationalError:
        return []  # table not created yet


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
                    n_samples=m.get("n_total_samples"), synthetic=m.get("trained_on_synthetic_data"))
    except (OSError, ValueError):
        pass
    return info


class DashboardState:
    def __init__(self, config: dict, db: Path, config_dir: Path):
        rec = config.get("recording", {})
        self.config = config
        self.db = db
        base = config_dir
        self.output_dir = (base / rec.get("output_dir", "recordings")).resolve()
        self.ml_model = (base / rec.get("ml_model", str(MODELS_DIR / "random_forest.joblib"))).resolve()
        self.wf_model = (base / rec.get("waterfall_model", str(MODELS_DIR / "waterfall_rf.joblib"))).resolve()
        self.tle_cache = (base / config.get("tle_cache_dir", rec.get("tle_cache_dir", "tle_cache"))).resolve()
        self.allowed_image_roots = [REPO.resolve(), self.output_dir, db.parent.resolve()]

    def live(self) -> dict:
        path = self.output_dir / "live_status.json"
        if not path.exists():
            return {"phase": "not running", "note": f"no heartbeat at {path} - start auto_capture.py"}
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
            data["note"] = f"no heartbeat for {age:.0f} s - is auto_capture.py still running?"
        return data

    def sky_track(self, live: dict) -> dict:
        """Track of the pass being waited for / recorded, else the last recorded pass."""
        cur = (live.get("current") or (live.get("upcoming") or [None])[0]) if isinstance(live, dict) else None
        if cur and cur.get("norad_id"):
            tle_file = self.tle_cache / f"{cur['norad_id']}.tle"
            if tle_file.exists():
                try:
                    from rtl_recorder.passes import GroundStation, load_tle_file, make_propagator, pass_track

                    st = live["station"]
                    track = pass_track(make_propagator(load_tle_file(tle_file, cur["norad_id"])),
                                       GroundStation(st["lat_deg"], st["lon_deg"], st.get("alt_m", 0)),
                                       datetime.fromisoformat(cur["aos"]), datetime.fromisoformat(cur["los"]),
                                       step_seconds=15)
                    return {"label": f"{'Now' if live.get('current') else 'Next'}: {cur['satellite']}",
                            "points": [[p["azimuth_deg"], p["elevation_deg"]] for p in track]}
                except Exception:  # never break the page over a plot
                    pass
        last = _rows(self.db, "SELECT capture_id, satellite_name FROM pass_positions ORDER BY id DESC LIMIT 1")
        if last:
            pts = _rows(self.db, "SELECT azimuth_deg, elevation_deg FROM pass_positions WHERE capture_id=? "
                                 "ORDER BY timestamp_utc", (last[0]["capture_id"],))
            return {"label": f"Last recorded: {last[0]['satellite_name']}",
                    "points": [[p["azimuth_deg"], p["elevation_deg"]] for p in pts]}
        return {"label": "no pass track yet", "points": []}

    def snapshot(self) -> dict:
        live = self.live()
        captures = _rows(self.db, "SELECT id, timestamp_utc, satellite_name, target_frequency_hz, frequency_hz, "
                                  "recording_status, rule_detection_result, ml_detection_result, ml_confidence_score, "
                                  "wf_ml_detection_result, wf_ml_confidence_score, decision_source, iq_retention, "
                                  "doppler_corrected, waterfall_image_path, spectrogram_image_path, "
                                  "recording_duration_seconds, output_file_size "
                                  "FROM capture_results ORDER BY id DESC LIMIT 15")
        if not captures:  # older DB without the newest columns
            captures = _rows(self.db, "SELECT * FROM capture_results ORDER BY id DESC LIMIT 15")
        totals = (_rows(self.db, """SELECT COUNT(*) AS n,
                 SUM(CASE WHEN recording_status='SUCCESS' THEN 1 ELSE 0 END) AS ok,
                 SUM(CASE WHEN COALESCE(wf_ml_detection_result, ml_detection_result, rule_detection_result)=1
                          THEN 1 ELSE 0 END) AS detected,
                 SUM(CASE WHEN iq_retention LIKE 'kept%' THEN 1 ELSE 0 END) AS kept,
                 SUM(CASE WHEN iq_retention='archived' THEN 1 ELSE 0 END) AS archived,
                 SUM(CASE WHEN iq_retention='deleted' THEN 1 ELSE 0 END) AS deleted
                 FROM capture_results""") or [{}])[0]
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
            "totals": totals,
            "status_log": status_log,
            "latest_image": latest_wf,
            "storage": {"free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1),
                        "recordings_gb": round(_folder_size(self.output_dir) / 1e9, 3),
                        "archived_gb": round(_folder_size(self.output_dir / "rejected") / 1e9, 3)},
            "models": {"iq": _model_info(self.ml_model), "waterfall": _model_info(self.wf_model)},
            "db": str(self.db),
        }

    def image_allowed(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return resolved.suffix.lower() == ".png" and resolved.is_file() and any(
            root == resolved or root in resolved.parents for root in self.allowed_image_roots)


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/status":
                self._send(200, json.dumps(state.snapshot(), default=str).encode(), "application/json")
            elif url.path == "/image":
                p = Path(parse_qs(url.query).get("path", [""])[0])
                if state.image_allowed(p):
                    self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "image/png")
                else:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Station Dashboard</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1c2230;--muted:#667085;--line:#e4e7ec;--accent:#2f6fde;--ok:#1f9d55;--warn:#c27c0e;--bad:#c53030;--sky:#eef3fb}
@media (prefers-color-scheme:dark){:root{--bg:#0f131a;--panel:#171c25;--ink:#e6e9ef;--muted:#98a2b3;--line:#273041;--accent:#6fa0ff;--ok:#3fbf7f;--warn:#e0a13a;--bad:#f06a6a;--sky:#1b2330}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:18px;margin:0}h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 10px}
main{display:grid;gap:16px;padding:16px 20px;grid-template-columns:repeat(12,1fr)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.c4{grid-column:span 4}.c5{grid-column:span 5}.c7{grid-column:span 7}.c8{grid-column:span 8}.c12{grid-column:span 12}
@media (max-width:960px){.c4,.c5,.c7,.c8{grid-column:span 12}}
.phase{font-size:22px;font-weight:600}.muted{color:var(--muted)}.small{font-size:12px}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:500}td.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.stat b{display:block;font-size:22px}
img.wf{max-width:100%;max-height:440px;object-fit:contain;border-radius:6px;border:1px solid var(--line)}svg text{fill:var(--muted);font-size:10px}
</style></head><body>
<header><h1 id="title">Ground station</h1><span class="muted small" id="clock"></span></header>
<main>
 <section class="card c4"><h2>Now</h2><div class="phase" id="phase">…</div><div id="nowdetail" class="muted"></div>
   <div style="margin-top:8px" id="recstate"></div><div class="small muted" id="note"></div></section>
 <section class="card c4"><h2>Totals</h2><div class="stats" id="stats"></div></section>
 <section class="card c4"><h2>Storage &amp; models</h2><div id="storage" class="small"></div></section>
 <section class="card c7"><h2>Upcoming passes</h2><div class="scroll"><table id="passes"></table></div></section>
 <section class="card c5"><h2 id="skylabel">Sky track</h2><svg id="sky" viewBox="-110 -110 220 220" width="100%" style="max-height:300px"></svg></section>
 <section class="card c8"><h2>Recent captures</h2><div class="scroll"><table id="captures"></table></div></section>
 <section class="card c4"><h2>Latest waterfall</h2><div id="wf" class="muted small">none yet</div></section>
 <section class="card c12"><h2>Status log</h2><div class="scroll"><table id="log"></table></div></section>
</main>
<script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const t=s=>s?String(s).replace('T',' ').slice(0,19):'–';
const pct=v=>v==null?'–':(100*v).toFixed(0)+'%';
function countdown(iso){const d=(new Date(iso)-Date.now())/1000;if(d<=0)return'now';const h=Math.floor(d/3600),m=Math.floor(d%3600/60),s=Math.floor(d%60);return(h?h+'h ':'')+m+'m '+(h?'':s+'s')}
function yesno(v,c){if(v==null)return'<span class="muted">–</span>';return v?`<span class="ok">yes${c!=null?' '+pct(c):''}</span>`:`<span class="muted">no${c!=null?' '+pct(c):''}</span>`}
function sky(points){const r=e=>100*(90-Math.max(0,e))/90;let g='';for(const e of[0,30,60])g+=`<circle r="${r(e)}" fill="none" stroke="var(--line)"/>`;
 g+=`<line x1="-100" y1="0" x2="100" y2="0" stroke="var(--line)"/><line x1="0" y1="-100" x2="0" y2="100" stroke="var(--line)"/>`;
 g+=`<text x="0" y="-102" text-anchor="middle">N</text><text x="104" y="3">E</text><text x="0" y="109" text-anchor="middle">S</text><text x="-110" y="3">W</text>`;
 const xy=p=>{const a=p[0]*Math.PI/180,rr=r(p[1]);return[rr*Math.sin(a),-rr*Math.cos(a)]};
 const vis=points.filter(p=>p[1]>=0).map(xy);if(vis.length>1){g+=`<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="${vis.map(p=>p.join(',')).join(' ')}"/>`;
 g+=`<circle cx="${vis[0][0]}" cy="${vis[0][1]}" r="3" fill="var(--ok)"/><circle cx="${vis.at(-1)[0]}" cy="${vis.at(-1)[1]}" r="3" fill="var(--bad)"/>`}
 document.getElementById('sky').innerHTML=g}
async function refresh(){
 let d;try{d=await (await fetch('/api/status')).json()}catch(e){document.getElementById('phase').textContent='dashboard offline';return}
 const L=d.live||{};document.getElementById('title').textContent=(L.station?.name||'Ground station')+(L.simulate?' (simulation)':'');
 document.getElementById('clock').textContent='UTC '+t(d.now_utc)+(L.heartbeat_age_s!=null?` · heartbeat ${L.heartbeat_age_s}s ago`:'');
 const ph=document.getElementById('phase');ph.textContent=L.phase||'–';ph.className='phase '+({'stalled?':'warn','not running':'muted'}[L.phase]||'');
 const cur=L.current;document.getElementById('nowdetail').innerHTML=cur?`${esc(cur.satellite)} · ${(cur.frequency_hz/1e6).toFixed(3)} MHz · AOS ${t(cur.aos)} (${countdown(cur.aos)}) · max el ${cur.max_elevation_deg}°`:'';
 document.getElementById('recstate').innerHTML=L.recorder_state?`Recorder <span class="pill">${esc(L.recorder_state)}</span>`:'';
 document.getElementById('note').textContent=L.note||'';
 const T=d.totals||{};document.getElementById('stats').innerHTML=[['Captures',T.n],['Recorded OK',T.ok],['Satellite found',T.detected],['IQ kept',T.kept],['Archived',T.archived],['Deleted',T.deleted]].map(([k,v])=>`<div class="stat"><b>${v??0}</b><span class="muted small">${k}</span></div>`).join('');
 const S=d.storage,M=d.models;const lowDisk=S.free_gb<10;
 document.getElementById('storage').innerHTML=`<div class="${lowDisk?'bad':''}">Disk free: <b>${S.free_gb} GB</b> of ${S.total_gb} GB</div><div>Recordings: ${S.recordings_gb} GB (archived ${S.archived_gb} GB)</div><hr style="border:0;border-top:1px solid var(--line)">`+
  ['iq','waterfall'].map(k=>{const m=M[k];return `<div>${k==='iq'?'IQ model':'Waterfall model'}: ${m.available?`<span class="ok">ready</span> <span class="muted">${esc(m.version||'')}${m.test_f1!=null?' · test F1 '+m.test_f1.toFixed(2):''}${m.synthetic?' · SYNTHETIC':''}</span>`:'<span class="warn">not trained</span>'}</div>`}).join('');
 const up=L.upcoming||[];document.getElementById('passes').innerHTML='<tr><th>Satellite</th><th>AOS (UTC)</th><th>In</th><th class="num">Max el</th><th class="num">Min</th><th>MHz</th><th></th></tr>'+(up.length?up.map(p=>`<tr><td>${esc(p.satellite)}</td><td>${t(p.aos)}</td><td>${countdown(p.aos)}</td><td class="num">${p.max_elevation_deg}°</td><td class="num">${p.duration_min}</td><td>${(p.frequency_hz/1e6).toFixed(3)}</td><td class="small ${p.skipped_reason?'warn':''}">${esc(p.skipped_reason||'')}</td></tr>`).join(''):'<tr><td colspan="7" class="muted">No schedule yet - start auto_capture.py</td></tr>');
 document.getElementById('skylabel').textContent='Sky track · '+(d.sky.label||'');sky(d.sky.points||[]);
 const C=d.captures||[];document.getElementById('captures').innerHTML='<tr><th>#</th><th>Time (UTC)</th><th>Satellite</th><th>Recording</th><th>Rule</th><th>IQ model</th><th>Waterfall model</th><th>Doppler</th><th>IQ file</th></tr>'+(C.length?C.map(c=>`<tr><td>${c.id}</td><td>${t(c.timestamp_utc)}</td><td>${esc(c.satellite_name||'–')}</td><td class="${c.recording_status==='SUCCESS'?'ok':'bad'}">${esc(c.recording_status||'–')}</td><td>${yesno(c.rule_detection_result)}</td><td>${yesno(c.ml_detection_result,c.ml_confidence_score)}</td><td>${yesno(c.wf_ml_detection_result,c.wf_ml_confidence_score)}</td><td>${c.doppler_corrected?'<span class="ok">corrected</span>':'<span class="muted">–</span>'}</td><td>${esc(c.iq_retention||'–')}</td></tr>`).join(''):'<tr><td colspan="9" class="muted">No captures yet</td></tr>');
 document.getElementById('wf').innerHTML=d.latest_image?`<img class="wf" src="/image?path=${encodeURIComponent(d.latest_image)}&t=${Date.now()}" alt="latest waterfall">`:'none yet';
 const G=d.status_log||[];document.getElementById('log').innerHTML='<tr><th>Time (UTC)</th><th>Component</th><th>State</th><th>Message</th></tr>'+(G.length?G.map(e=>`<tr><td>${t(e.timestamp_utc)}</td><td>${esc(e.component)}</td><td>${esc(e.state)}</td><td style="white-space:normal">${esc(e.message)}</td></tr>`).join(''):'<tr><td colspan="4" class="muted">No events yet</td></tr>');
}
refresh();setInterval(refresh,5000);
</script></body></html>"""


def build_state(args) -> DashboardState:
    config, config_dir = {}, REPO
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config_dir = REPO  # auto_capture runs from the repo root, so config paths are repo-relative
    rec = config.get("recording", {})
    db = Path(args.db or rec.get("db_path") or config.get("db_path") or DB_PATH)
    if not db.is_absolute():
        db = (config_dir / db).resolve()
    if args.output_dir:
        config.setdefault("recording", {})["output_dir"] = args.output_dir
    return DashboardState(config, db, config_dir)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Live dashboard for the capture station.")
    p.add_argument("--config", help="Same JSON config as auto_capture.py")
    p.add_argument("--db", help="SQLite database (default: from config, else sdr-doppler-prototype's)")
    p.add_argument("--output-dir", help="Recordings folder (where live_status.json is written)")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other computers on the LAN")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args(argv)
    state = build_state(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Dashboard on http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}  "
          f"(database {state.db}, recordings {state.output_dir}) - Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
