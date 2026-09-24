"""Long unattended runs: disk-space guard, retries, resilient loop, daily log."""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import station_guard  # noqa: E402
from tests.test_auto_capture import _pp, _settings, config, tle_file  # noqa: E402,F401  (fixtures)

import auto_capture  # noqa: E402


def _file(path: Path, size: int, age_s: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    t = time.time() - age_s
    os.utime(path, (t, t))
    return path


def test_enough_space_needs_no_pruning(tmp_path):
    r = station_guard.ensure_space(tmp_path, 1_024_000, 600, min_free_gb=0, free_fn=lambda p: 10**12)
    assert r.ok and not r.deleted and r.needed_bytes == int(1_024_000 * 2 * 600 * 1.2)


def test_prunes_oldest_rejected_only(tmp_path):
    old = _file(tmp_path / "rejected" / "old.iq", 600, 300)
    new = _file(tmp_path / "rejected" / "new.iq", 600, 10)
    sidecar = _file(tmp_path / "rejected" / "old.json", 10, 300)
    kept = _file(tmp_path / "kept.iq", 600, 999)
    unsure = _file(tmp_path / "uncertain" / "u.iq", 600, 999)
    state = {"free": 1000}

    def free_fn(_):
        return state["free"] + sum(600 for f in (old, new) if not f.exists())

    r = station_guard.ensure_space(tmp_path, 1, 500, min_free_gb=0, margin=1.0, free_fn=free_fn)
    # needs 1000 bytes (1 x 2 x 500): already enough -> nothing deleted
    assert r.ok and old.exists()
    r = station_guard.ensure_space(tmp_path, 1, 800, min_free_gb=0, margin=1.0, free_fn=free_fn)
    # needs 1600: frees the OLDEST rejected recording only
    assert r.ok and not old.exists() and new.exists() and r.deleted == [str(old)]
    assert kept.exists() and unsure.exists() and sidecar.exists()


def test_not_enough_space_skips_the_pass(tmp_path, config):
    s = _settings(config, "--simulate", "--db", str(tmp_path / "c.sqlite3"))
    s.min_free_gb = 10**6   # a million GB: never enough
    now = datetime.now(timezone.utc)
    out = auto_capture.run_plan([_pp("SAT-A", now + timedelta(seconds=0.5), 1 / 60, 40)], s)
    assert out[0]["recording_status"] == "SKIPPED_DISK_FULL" and "NOT ENOUGH SPACE" in out[0]["error"]
    assert not list(Path(s.output_dir).glob("*.iq"))


class FlakyRecorder:
    """First attempt: dongle busy; second: success (delegates to the simulated recorder)."""

    def __init__(self, real):
        self.real, self.calls = real, 0

    def record_pass(self, **kw):
        self.calls += 1
        if self.calls == 1:
            from rtl_recorder import RecordingResult, RecordingStatus

            return RecordingResult(status=RecordingStatus.DEVICE_BUSY, error_message="usb_claim_interface error -6")
        return self.real.record_pass(**kw)

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_busy_dongle_is_retried_for_the_rest_of_the_pass(tmp_path, config):
    from rtl_recorder import RecorderConfig, RTLSDRRecorder

    s = _settings(config, "--simulate", "--db", str(tmp_path / "c.sqlite3"), "--results-dir", str(tmp_path / "res"))
    s.pre_buffer = s.post_buffer = 0.2
    s.retry_delay_s = 0.1
    s.min_retry_window_s = 0.5
    s.min_free_gb = 0
    rec = FlakyRecorder(RTLSDRRecorder(RecorderConfig(output_dir=s.output_dir, simulate=True)))
    now = datetime.now(timezone.utc)
    out = auto_capture.run_plan([_pp("SAT-A", now + timedelta(seconds=0.3), 0.05, 40)], s, recorder=rec)
    assert rec.calls == 2 and out[0]["recording_status"] == "SUCCESS"


def test_forever_loop_survives_a_planning_error(tmp_path, config, monkeypatch):
    calls = {"n": 0}

    def flaky_plan(settings, start=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("no internet")
        raise KeyboardInterrupt  # end the test after the retry

    monkeypatch.setattr(auto_capture, "plan_passes", flaky_plan)
    monkeypatch.setattr(auto_capture.threading.Event, "wait", lambda self, t=None: False)
    monkeypatch.setattr(station_guard, "setup_daily_log", lambda d: Path(d) / "station.log")
    with pytest.raises(KeyboardInterrupt):
        auto_capture.main(["--config", str(config), "--simulate", "--forever", "--db", str(tmp_path / "c.sqlite3")])
    assert calls["n"] == 2


def test_daily_log_file(tmp_path):
    path = station_guard.setup_daily_log(tmp_path / "logs")
    logging.getLogger("station-test").warning("hello from the station")
    for h in logging.getLogger().handlers:
        h.flush()
    assert path.exists() and "hello from the station" in path.read_text()
    for h in list(logging.getLogger().handlers):
        if getattr(h, "baseFilename", "") == str(path.resolve()):
            logging.getLogger().removeHandler(h)
            h.close()


def test_keep_awake_is_harmless_off_windows():
    if not sys.platform.startswith("win"):
        assert station_guard.keep_awake() is False


def test_kept_meteor_pass_is_queued_for_decoding(tmp_path, config, monkeypatch):
    s = _settings(config, "--simulate", "--db", str(tmp_path / "c.sqlite3"), "--results-dir", str(tmp_path / "res"))
    s.pre_buffer = s.post_buffer = 0.2
    s.min_free_gb = 0
    seen = []
    monkeypatch.setattr(auto_capture.meteor_decode, "decode_capture",
                        lambda cid, **kw: seen.append(cid) or {"status": "decoded: 3 image(s)", "images": [], "image_dir": None})
    worker = auto_capture.DecodeWorker(s, tmp_path / "c.sqlite3")
    now = datetime.now(timezone.utc)
    meteor = _pp("METEOR-M2-3", now + timedelta(seconds=0.3), 1 / 60, 40)
    iss = _pp("ISS", now + timedelta(seconds=2.5), 1 / 60, 40)
    out = auto_capture.run_plan([meteor, iss], s, decoder=worker)
    assert worker.wait(timeout=10)
    assert [o["decode_queued"] for o in out] == [True, False]
    assert seen == [out[0]["db_row"]] and worker.results[0]["status"].startswith("decoded")
