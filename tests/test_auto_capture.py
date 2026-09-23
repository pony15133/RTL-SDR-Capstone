"""End-to-end tests for auto_capture.py: plan passes -> record -> detect ->
database -> retention, all in simulate mode."""

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import auto_capture  # noqa: E402
from rtl_recorder.passes import GroundStation, Pass, load_tle_file  # noqa: E402


def _fix(line):
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return line[:68] + str(total % 10)


@pytest.fixture
def tle_file(tmp_path):
    path = tmp_path / "iss.tle"
    path.write_text("ISS (ZARYA)\n"
                    + _fix("1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005") + "\n"
                    + _fix("2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537") + "\n")
    return path


@pytest.fixture
def config(tmp_path, tle_file):
    cfg = json.loads((ROOT / "capture_config.example.json").read_text())
    cfg["recording"].update({"output_dir": str(tmp_path / "rec"), "ml_model": str(tmp_path / "no_model.joblib")})
    cfg["tle_file"] = str(tle_file)
    path = tmp_path / "capture_config.json"
    path.write_text(json.dumps(cfg))
    return path


def _settings(config, *extra):
    args = auto_capture.build_parser().parse_args(["--config", str(config), *extra])
    return auto_capture.load_settings(args)


def test_config_is_loaded(config):
    s = _settings(config)
    assert s.station.name == "Singapore campus"
    assert [t.norad_id for t in s.targets] == [25544, 57166, 59051]
    assert s.targets[0].sample_rate == 1_024_000
    assert s.retention == "archive-negatives"


def test_cli_overrides_config(config):
    s = _settings(config, "--retention", "keep-all", "--min-elevation", "40", "--lat", "10")
    assert s.retention == "keep-all" and s.min_elevation_deg == 40 and s.station.lat_deg == 10


def test_single_satellite_mode_needs_frequency(config):
    with pytest.raises(SystemExit):
        _settings(config, "--norad", "25544")


def test_plan_uses_tle_file_and_skips_missing_satellites(config, tle_file):
    s = _settings(config)  # only ISS is in the TLE file; the Meteors are logged and skipped
    plan = auto_capture.plan_passes(s, start=load_tle_file(tle_file).epoch)
    assert plan, "expected at least one ISS pass in 24 h"
    assert all(pp.target.name == "ISS" for pp in plan)
    assert [pp.pass_.aos for pp in plan] == sorted(pp.pass_.aos for pp in plan)
    assert all(pp.pass_.max_elevation_deg >= 15 for pp in plan)


def _pp(name, aos, minutes, max_el):
    t = auto_capture.Target(name, 1, 437_000_000)
    return auto_capture.PlannedPass(t, Pass(name, 1, aos, aos + timedelta(minutes=minutes), max_el,
                                            aos + timedelta(minutes=minutes / 2), 0, 180, "test"))


def test_overlapping_passes_keep_the_higher_one():
    t0 = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
    low = _pp("LOW", t0, 10, 20)
    high = _pp("HIGH", t0 + timedelta(minutes=5), 10, 60)
    later = _pp("LATER", t0 + timedelta(hours=2), 10, 30)
    plan = auto_capture.resolve_conflicts([later, high, low], pre=30, post=30)
    by_name = {pp.target.name: pp for pp in plan}
    assert by_name["LOW"].skipped_reason and "HIGH" in by_name["LOW"].skipped_reason
    assert by_name["HIGH"].skipped_reason is None
    assert by_name["LATER"].skipped_reason is None


def test_run_plan_records_detects_and_logs_each_pass(tmp_path, config):
    s = _settings(config, "--simulate", "--db", str(tmp_path / "c.sqlite3"), "--results-dir", str(tmp_path / "res"))
    s.pre_buffer = s.post_buffer = 0.2
    now = datetime.now(timezone.utc)
    plan = [_pp("SAT-A", now + timedelta(seconds=0.5), 1 / 60, 40),
            _pp("SAT-B", now + timedelta(seconds=2.5), 1 / 60, 50)]
    summaries = auto_capture.run_plan(plan, s)

    assert [x["recording_status"] for x in summaries] == ["SUCCESS", "SUCCESS"]
    with sqlite3.connect(tmp_path / "c.sqlite3") as conn:
        rows = conn.execute("SELECT satellite_name, scheduled_aos, iq_retention, processing_status "
                            "FROM capture_results ORDER BY id").fetchall()
    assert [r[0] for r in rows] == ["SAT-A", "SAT-B"]
    assert all(r[1] for r in rows)                        # scheduled AOS recorded
    assert all(r[3] == "DETECTED" for r in rows)
    assert all(r[2] in ("archived", "kept") for r in rows)


def test_skipped_and_past_passes_do_not_record(tmp_path, config):
    s = _settings(config, "--simulate", "--db", str(tmp_path / "c.sqlite3"))
    past = _pp("PAST", datetime.now(timezone.utc) - timedelta(hours=1), 5, 40)
    skipped = _pp("SKIP", datetime.now(timezone.utc) + timedelta(hours=1), 5, 10)
    skipped.skipped_reason = "overlap"
    summaries = auto_capture.run_plan([past, skipped], s)
    assert len(summaries) == 1 and summaries[0]["recording_status"] == "FAILED"
    with sqlite3.connect(tmp_path / "c.sqlite3") as conn:
        status = conn.execute("SELECT processing_status FROM capture_results").fetchone()[0]
    assert status == "NOT_PROCESSED_RECORDING_FAILED"  # the failed attempt is still in the log


def test_list_only_prints_schedule(config, capsys):
    assert auto_capture.main(["--config", str(config), "--list-only", "--hours", "1"]) == 0
    assert "Pass schedule for Singapore campus" in capsys.readouterr().out


def test_demo_runs_the_whole_chain(tmp_path, config, capsys):
    code = auto_capture.main(["--config", str(config), "--demo", "--db", str(tmp_path / "demo.sqlite3"),
                              "--results-dir", str(tmp_path / "res"), "--output-dir", str(tmp_path / "rec")])
    out = capsys.readouterr().out
    assert code == 0
    assert "SUCCESS, db row 1" in out
    assert json.loads((tmp_path / "rec" / "schedule.json").read_text())[0]["satellite"] == "ISS"
    assert list((tmp_path / "res").glob("*_spectrogram.png"))


def test_position_history_and_status_log_are_stored(tmp_path, config, tle_file):
    from database import get_pass_positions, list_status

    s = _settings(config, "--simulate", "--db", str(tmp_path / "c.sqlite3"), "--results-dir", str(tmp_path / "res"))
    s.pre_buffer = s.post_buffer = 0.2
    tle = load_tle_file(tle_file)
    now = datetime.now(timezone.utc)
    pp = _pp("ISS", now + timedelta(seconds=0.5), 1 / 60, 40)
    pp.tle = tle
    summary = auto_capture.run_plan([pp], s)[0]

    track = get_pass_positions(tmp_path / "c.sqlite3", summary["db_row"])
    assert len(track) == summary["track_points"] >= 1
    assert {"azimuth_deg", "elevation_deg", "range_km", "doppler_hz"} <= set(track[0])
    assert track[0]["norad_id"] == pp.target.norad_id
    states = [(e["component"], e["state"]) for e in list_status(tmp_path / "c.sqlite3")]
    assert ("scheduler", "WAITING") in states and ("recorder", "SUCCESS") in states and ("pipeline", "DONE") in states
