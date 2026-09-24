#!/usr/bin/env python3
"""Verify the automatic pipeline end to end in simulation, and save the evidence.

    python tools/verify_simulated_pipeline.py                  # writes evidence/pipeline/
    python tools/verify_simulated_pipeline.py --tle-file my.tle --norad 25544

Two runs, each in a fresh temporary folder (nothing touches your real
database or recordings):

  A. ``auto_capture.py --demo`` exactly as a user runs it: fake pass 5 s
     from now -> simulated record_pass() -> processing -> rule + ML
     detectors -> SQLite -> retention -> heartbeat for the dashboard.
     The demo pass has no TLE, so Doppler correction and the position
     track are skipped by design.
  B. A pass with a real TLE attached (default: the ISS element set bundled
     below, epoch 2024-01-01) run through auto_capture.run_plan(), so the
     TLE-dependent steps run too: Doppler curve, offset tuning,
     Doppler-corrected waterfall and pass_positions rows. The recording
     window is placed a few seconds from now (a real pass can be hours
     away) - this checks the software path, not pass-time accuracy.

The simulated recorder writes 64 KB of random bytes, not a satellite
signal, so detection results here mean nothing - only that every stage
ran and stored what it should. After each run the script checks the
database rows and output files and fails loudly on anything missing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import auto_capture  # noqa: E402  (also puts both subprojects on sys.path)
import dashboard  # noqa: E402
from rtl_recorder.passes import Pass, load_tle_file  # noqa: E402

# ISS element set (epoch 2024-01-01), the same one tests/test_auto_capture.py uses.
ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537
"""

CAPTURE_FIELDS = ["satellite_name", "norad_id", "frequency_hz", "target_frequency_hz", "sample_rate", "gain",
                  "recording_status", "scheduled_aos", "scheduled_los", "actual_recording_start",
                  "actual_recording_stop", "recording_duration_seconds", "output_file_size", "simulated",
                  "processing_status", "rule_detection_result", "rule_confidence_score", "ml_detection_result",
                  "ml_confidence_score", "model_version", "wf_ml_detection_result", "doppler_corrected",
                  "doppler_max_hz", "decision_source", "decision_score", "iq_retention", "retention_reason",
                  "raw_iq_file_path", "metadata_file_path", "spectrogram_image_path", "waterfall_image_path"]


def _fix_checksum(line: str) -> str:
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return line[:68] + str(total % 10)


def _db(db: Path, sql: str, params=()) -> list:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params)]


def _exists(path) -> bool:
    return bool(path) and Path(path).exists()


def check(results: list, name: str, ok: bool, detail="") -> None:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def run_demo(work: Path, config: Path) -> dict:
    print("\nA. auto_capture.py --demo")
    db, rec, res = work / "captures.sqlite3", work / "recordings", work / "results"
    code = auto_capture.main(["--config", str(config), "--demo", "--db", str(db), "--output-dir", str(rec),
                              "--results-dir", str(res), "--log-level", "WARNING"])
    checks: list = []
    check(checks, "auto_capture --demo exit code 0", code == 0, f"exit {code}")
    rows = _db(db, "SELECT * FROM capture_results")
    check(checks, "one capture_results row", len(rows) == 1, f"{len(rows)} row(s)")
    row = rows[0] if rows else {}
    check(checks, "recording SUCCESS and processed", row.get("recording_status") == "SUCCESS"
          and row.get("processing_status") == "DETECTED", f"{row.get('recording_status')}/{row.get('processing_status')}")
    check(checks, "scheduled AOS/LOS and actual start/stop stored",
          all(row.get(k) for k in ("scheduled_aos", "scheduled_los", "actual_recording_start", "actual_recording_stop")))
    check(checks, "offset tuning: tuned 150 kHz below the downlink",
          row.get("target_frequency_hz", 0) - (row.get("frequency_hz") or 0) == 150_000,
          f"target {row.get('target_frequency_hz')} tuned {row.get('frequency_hz')}")
    check(checks, "rule detector result stored", row.get("rule_detection_result") in (0, 1))
    ml_model = REPO / "sdr-doppler-prototype" / "models" / "random_forest.joblib"
    if ml_model.exists():
        check(checks, "ML detector ran (model present)", row.get("ml_confidence_score") is not None,
              f"model_version {row.get('model_version')}")
    else:
        check(checks, "ML detector reported as unavailable (no model trained)", row.get("ml_detection_result") is None,
              "train one with setup / train_model.py to exercise it")
    check(checks, "retention decision stored", bool(row.get("iq_retention") and row.get("retention_reason")),
          f"{row.get('iq_retention')}: {row.get('retention_reason')}")
    check(checks, "IQ file where the row says it is (or deleted)", row.get("iq_retention") == "deleted"
          or _exists(row.get("raw_iq_file_path")), row.get("raw_iq_file_path"))
    check(checks, "recorder JSON sidecar exists", _exists(row.get("metadata_file_path"))
          or _exists(Path(row.get("raw_iq_file_path") or "x").with_suffix(".json")))
    check(checks, "spectrogram PNG exists", _exists(row.get("spectrogram_image_path")))
    check(checks, "summary JSON exists", bool(list(res.glob("*_summary.json"))))
    check(checks, "no Doppler correction on the TLE-less demo pass (expected)", row.get("doppler_corrected") == 0)
    states = {(e["component"], e["state"]) for e in _db(db, "SELECT component, state FROM status_log")}
    check(checks, "status_log has scheduler WAITING, recorder SUCCESS, pipeline DONE",
          {("scheduler", "WAITING"), ("recorder", "SUCCESS"), ("pipeline", "DONE")} <= states, sorted(states))
    check(checks, "schedule.json written", _exists(rec / "schedule.json"))
    live = json.loads((rec / "live_status.json").read_text(encoding="utf-8")) if _exists(rec / "live_status.json") else {}
    check(checks, "heartbeat live_status.json written, final phase 'stopped'", live.get("phase") == "stopped",
          live.get("phase"))
    snap = dashboard.DashboardState({"recording": {"output_dir": str(rec)}}, db, REPO).snapshot()
    check(checks, "dashboard snapshot shows the capture and totals", len(snap["captures"]) == 1
          and snap["totals"].get("n") == 1, f"totals {snap['totals']}")
    return {"run": "A: auto_capture --demo", "checks": checks,
            "capture_row": {k: row.get(k) for k in CAPTURE_FIELDS}, "status_log_states": sorted(states)}


def run_tle_pass(work: Path, config: Path, tle_path: Path, norad: int) -> dict:
    print("\nB. simulated pass with a TLE (Doppler, position history)")
    db = work / "captures.sqlite3"
    args = auto_capture.build_parser().parse_args(
        ["--config", str(config), "--simulate", "--db", str(db), "--output-dir", str(work / "recordings"),
         "--results-dir", str(work / "results")])
    settings = auto_capture.load_settings(args)
    settings.pre_buffer = settings.post_buffer = 0.5
    tle = load_tle_file(tle_path, norad)
    target = next((t for t in settings.targets if t.norad_id == norad), settings.targets[0])
    now = datetime.now(timezone.utc)
    p = Pass(target.name, target.norad_id, now + timedelta(seconds=2), now + timedelta(seconds=6), 40.0,
             now + timedelta(seconds=4), 90.0, 270.0, "verification")
    summary = auto_capture.run_plan([auto_capture.PlannedPass(target, p, tle=tle)], settings)[0]
    checks: list = []
    check(checks, "recording SUCCESS", summary["recording_status"] == "SUCCESS", summary["recording_status"])
    row = _db(db, "SELECT * FROM capture_results WHERE id=?", (summary["db_row"],))[0]
    check(checks, "Doppler curve computed from the TLE and applied", row["doppler_corrected"] == 1
          and (row["doppler_max_hz"] or 0) > 0, f"max |Doppler| {row['doppler_max_hz']:.0f} Hz")
    check(checks, "Doppler-corrected waterfall PNG exists", _exists(row["waterfall_image_path"]))
    track = _db(db, "SELECT * FROM pass_positions WHERE capture_id=?", (summary["db_row"],))
    check(checks, "pass_positions rows stored (az/el/range/Doppler)", len(track) >= 1 and all(
        t["azimuth_deg"] is not None and t["elevation_deg"] is not None for t in track), f"{len(track)} point(s)")
    return {"run": "B: TLE pass via run_plan", "checks": checks, "tle_epoch": tle.epoch.isoformat(),
            "capture_row": {k: row.get(k) for k in CAPTURE_FIELDS}, "track_points": len(track),
            "first_track_point": track[0] if track else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Simulated end-to-end verification of the automatic pipeline.")
    p.add_argument("--config", type=Path, default=REPO / "capture_config.example.json")
    p.add_argument("--tle-file", type=Path, help="TLE file for run B (default: bundled ISS set)")
    p.add_argument("--norad", type=int, default=25544)
    p.add_argument("--output-dir", type=Path, default=REPO / "evidence" / "pipeline")
    args = p.parse_args(argv)
    os.chdir(REPO)  # the config's model paths are repo-relative, as when auto_capture runs from the repo root

    with tempfile.TemporaryDirectory(prefix="sdr_verify_") as tmp:
        tmp = Path(tmp)
        tle_path = args.tle_file
        if tle_path is None:
            lines = ISS_TLE.strip().splitlines()
            tle_path = tmp / "iss.tle"
            tle_path.write_text("\n".join([lines[0], _fix_checksum(lines[1]), _fix_checksum(lines[2])]) + "\n")
        runs = [run_demo(tmp / "demo", args.config), run_tle_pass(tmp / "tle", args.config, tle_path, args.norad)]

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = None
    all_ok = all(c["ok"] for r in runs for c in r["checks"])
    report = {"all_checks_passed": all_ok, "runs": runs,
              "note": "Simulated recorder output (random bytes): detection outcomes are meaningless; this verifies "
                      "that every stage runs and stores its results. Not live-hardware evidence.",
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "git_commit": sha,
                              "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "simulated_pipeline.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md = ["# Simulated end-to-end pipeline verification", "",
          f"Generated {report['environment']['generated_utc']} on commit `{sha}` by `tools/verify_simulated_pipeline.py` "
          f"(Python {report['environment']['python']}, {report['environment']['platform']}).", "",
          f"**Result: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}**", "", report["note"], ""]
    for r in runs:
        md += [f"## {r['run']}", "", "| Check | Result | Detail |", "|---|---|---|"]
        md += [f"| {c['check']} | {'PASS' if c['ok'] else 'FAIL'} | {str(c['detail']).replace('|', '/')[:160]} |"
               for c in r["checks"]]
        md += ["", "Stored `capture_results` row (selected fields):", "", "```json",
               json.dumps(r["capture_row"], indent=1, default=str), "```", ""]
    (args.output_dir / "SIMULATED_PIPELINE.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} - wrote {args.output_dir}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
