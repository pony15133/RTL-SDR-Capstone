#!/usr/bin/env python3
"""Automatic satellite capture - the whole project in one command.

    predict passes (TLE) -> wait -> record AOS..LOS (+buffers) -> spectrogram
    -> rule + ML detection -> SQLite row -> keep / archive / delete the IQ

Examples (run from the repo root):

    # What's coming up over Singapore in the next 24 h?
    python auto_capture.py --config capture_config.json --list-only

    # Run the schedule for real (Ctrl+C to stop; the current recording is finalised)
    python auto_capture.py --config capture_config.json

    # Single satellite without a config file
    python auto_capture.py --norad 25544 --name ISS --frequency 437800000 \\
        --lat 1.3521 --lon 103.8198 --hours 12

    # 20-second end-to-end demo, no hardware, no waiting
    python auto_capture.py --config capture_config.json --demo

The config file (see capture_config.example.json) holds the ground station,
the recording defaults and the list of satellites. One dongle can only
record one satellite at a time, so overlapping passes are resolved by
taking the higher-elevation pass and logging the one that was skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pipeline  # noqa: F401  (sets up sys.path for both subprojects)
from config import DB_PATH
from database import insert_pass_positions, log_status
from doppler import doppler_curve_from_pass
from pipeline import process_recording
import decode as meteor_decode
from retention import DEFAULT_UNCERTAIN_BAND, POLICIES, parse_band
import station_guard
from rtl_recorder.config import RecorderConfig
from rtl_recorder.passes import TLE, GroundStation, Pass, find_passes, get_tle, load_tle_file, make_propagator, pass_track
from rtl_recorder.recorder import RTLSDRRecorder

logger = logging.getLogger("auto_capture")


@dataclass
class Target:
    name: str
    norad_id: int
    frequency_hz: int
    sample_rate: int = 1_024_000
    gain: object = "auto"
    min_elevation_deg: Optional[float] = None
    #: Width of the Doppler-corrected waterfall around the downlink. 48 kHz suits
    #: narrow FM/FSK beacons; wide signals like METEOR LRPT (~150 kHz) need ~160 kHz,
    #: which is also how SatNOGS draws them (so the waterfall model sees the same view).
    waterfall_span_hz: float = 48_000.0


@dataclass
class Settings:
    station: GroundStation
    targets: List[Target]
    hours: float = 24.0
    min_elevation_deg: float = 15.0
    pre_buffer: float = 30.0
    post_buffer: float = 30.0
    output_dir: str = "recordings"
    results_dir: Optional[str] = None
    db_path: Optional[str] = None
    ml_model: Optional[str] = None
    retention: str = "archive-negatives"
    keep_threshold: Optional[float] = None  # None = the model's tuned threshold (two-level mode only)
    #: Confidence range judged "uncertain": IQ always kept for review. None = two levels.
    uncertain_band: Optional[tuple] = DEFAULT_UNCERTAIN_BAND
    tle_file: Optional[str] = None
    tle_cache_dir: str = "tle_cache"
    save_image: bool = True
    tuning_offset_hz: float = 150_000.0
    waterfall_model: Optional[str] = None
    device_index: int = 0
    rtl_sdr_path: Optional[str] = None
    simulate: bool = False
    # Long unattended runs (station_guard.py)
    min_free_gb: float = station_guard.DEFAULT_MIN_FREE_GB
    prune_rejected: bool = True
    record_retries: int = 2
    retry_delay_s: float = 15.0
    min_retry_window_s: float = 30.0   # don't retry for less than this much of the pass
    log_dir: str = "logs"
    # METEOR LRPT decoding of kept recordings (sdr-doppler-prototype/src/decode.py)
    decode: bool = True
    satdump_path: Optional[str] = None


@dataclass
class PlannedPass:
    target: Target
    pass_: Pass
    skipped_reason: Optional[str] = None
    tle: Optional[TLE] = None

    def as_dict(self) -> dict:
        p = self.pass_
        return {
            "satellite": self.target.name, "norad_id": self.target.norad_id,
            "frequency_hz": self.target.frequency_hz, "aos": p.aos.isoformat(), "los": p.los.isoformat(),
            "max_elevation_deg": round(p.max_elevation_deg, 1), "duration_min": round(p.duration_seconds / 60, 1),
            "propagator": p.propagator, "skipped_reason": self.skipped_reason,
        }


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def load_settings(args) -> Settings:
    cfg: dict = {}
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    st = cfg.get("station", {})
    lat = args.lat if args.lat is not None else st.get("lat_deg")
    lon = args.lon if args.lon is not None else st.get("lon_deg")
    if lat is None or lon is None:
        raise SystemExit("FAILED: ground station latitude/longitude needed (--lat/--lon or 'station' in --config).")
    station = GroundStation(float(lat), float(lon), float(args.alt if args.alt is not None else st.get("alt_m", 0.0)),
                            st.get("name", "station"))

    defaults = cfg.get("recording", {})
    targets = [Target(**{k: v for k, v in t.items() if k in Target.__dataclass_fields__})
               for t in cfg.get("satellites", [])]
    if args.norad:
        if not args.frequency:
            raise SystemExit("FAILED: --frequency is required with --norad.")
        targets = [Target(args.name or f"NORAD {args.norad}", args.norad, int(args.frequency),
                          int(args.sample_rate or defaults.get("sample_rate", 1_024_000)), args.gain or "auto")]
    if not targets:
        raise SystemExit("FAILED: no satellites - add a 'satellites' list to --config or pass --norad/--frequency.")
    for t in targets:
        if args.sample_rate:
            t.sample_rate = int(args.sample_rate)
        elif "sample_rate" in defaults and t.sample_rate == Target.__dataclass_fields__["sample_rate"].default:
            t.sample_rate = int(defaults["sample_rate"])
        if args.gain:
            t.gain = args.gain

    def pick(name, default):
        cli = getattr(args, name, None)
        return cli if cli is not None else defaults.get(name, cfg.get(name, default))

    return Settings(
        station=station, targets=targets,
        hours=float(pick("hours", 24.0)),
        min_elevation_deg=float(pick("min_elevation_deg", 15.0)),
        pre_buffer=float(pick("pre_buffer", 30.0)),
        post_buffer=float(pick("post_buffer", 30.0)),
        output_dir=str(pick("output_dir", "recordings")),
        results_dir=pick("results_dir", None),
        db_path=pick("db_path", None),
        ml_model=pick("ml_model", None),
        retention=str(pick("retention", "archive-negatives")),
        keep_threshold=(None if pick("keep_threshold", None) is None else float(pick("keep_threshold", None))),
        uncertain_band=parse_band(pick("uncertain_band", list(DEFAULT_UNCERTAIN_BAND))),
        tle_file=pick("tle_file", None),
        tle_cache_dir=str(pick("tle_cache_dir", "tle_cache")),
        save_image=not args.no_image and bool(defaults.get("save_image", True)),
        device_index=int(pick("device_index", 0)),
        tuning_offset_hz=float(pick("tuning_offset_hz", 150_000.0)),
        waterfall_model=pick("waterfall_model", None),
        rtl_sdr_path=pick("rtl_sdr_path", None),
        simulate=bool(args.simulate or args.demo),
        min_free_gb=float(pick("min_free_gb", station_guard.DEFAULT_MIN_FREE_GB)),
        prune_rejected=bool(pick("prune_rejected", True)),
        record_retries=int(pick("record_retries", 2)),
        retry_delay_s=float(pick("retry_delay_s", 15.0)),
        log_dir=str(pick("log_dir", "logs")),
        decode=bool(pick("decode", True)),
        satdump_path=pick("satdump_path", None),
    )


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

def plan_passes(settings: Settings, start: Optional[datetime] = None) -> List[PlannedPass]:
    """Predict passes for every target, then resolve overlaps (one dongle)."""
    start = start or datetime.now(timezone.utc)
    planned: List[PlannedPass] = []
    for target in settings.targets:
        try:
            tle = (load_tle_file(settings.tle_file, target.norad_id) if settings.tle_file
                   else get_tle(target.norad_id, cache_dir=Path(settings.tle_cache_dir)))
        except Exception as exc:  # one bad satellite must not stop the others
            logger.error("No TLE for %s (NORAD %s): %s - skipping it", target.name, target.norad_id, exc)
            continue
        age = tle.age_days(start)
        if age > 7:
            logger.warning("TLE for %s is %.0f days old - pass times may be off; refresh it", target.name, age)
        min_el = target.min_elevation_deg if target.min_elevation_deg is not None else settings.min_elevation_deg
        for p in find_passes(tle, settings.station, start=start, hours=settings.hours, min_max_elevation_deg=min_el):
            p.satellite_name = target.name
            planned.append(PlannedPass(target, p, tle=tle))
    return resolve_conflicts(planned, settings.pre_buffer, settings.post_buffer)


def resolve_conflicts(planned: List[PlannedPass], pre: float, post: float) -> List[PlannedPass]:
    """Mark passes whose recording windows overlap a better (higher) pass as skipped."""
    planned.sort(key=lambda pp: pp.pass_.aos)
    kept: List[PlannedPass] = []
    for pp in planned:
        start = pp.pass_.aos - timedelta(seconds=pre)
        clash = next((k for k in kept if k.skipped_reason is None and
                      start < k.pass_.los + timedelta(seconds=post)), None)
        if clash is None:
            kept.append(pp)
        elif pp.pass_.max_elevation_deg > clash.pass_.max_elevation_deg:
            clash.skipped_reason = f"overlaps higher pass of {pp.target.name}"
            kept.append(pp)
        else:
            pp.skipped_reason = f"overlaps higher pass of {clash.target.name}"
            kept.append(pp)
    return kept


def demo_plan(settings: Settings) -> List[PlannedPass]:
    """A fake pass 5 s from now, 10 s long, for the first target - lets the
    whole chain run in ~20 s without hardware or waiting for a real pass."""
    now = datetime.now(timezone.utc)
    t = settings.targets[0]
    fake = Pass(t.name, t.norad_id, now + timedelta(seconds=5), now + timedelta(seconds=15), 45.0,
                now + timedelta(seconds=10), 90.0, 270.0, "demo")
    settings.pre_buffer = settings.post_buffer = 1.0
    return [PlannedPass(t, fake)]


def format_schedule(plan: List[PlannedPass], station: GroundStation) -> str:
    lines = [f"Pass schedule for {station.name} ({station.lat_deg:.4f}, {station.lon_deg:.4f}) - times UTC"]
    if not plan:
        lines.append("  (no passes above the elevation limit in this window)")
    for pp in plan:
        p = pp.pass_
        flag = f"  SKIP: {pp.skipped_reason}" if pp.skipped_reason else ""
        lines.append(f"  {p.aos:%Y-%m-%d %H:%M:%S} -> {p.los:%H:%M:%S}  max el {p.max_elevation_deg:4.1f}  "
                     f"{pp.target.name:<16} {pp.target.frequency_hz / 1e6:.3f} MHz  [{p.propagator}]{flag}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

class LiveStatus:
    """Writes <output_dir>/live_status.json every few seconds for dashboard.py:
    what the station is doing right now and what's next."""

    def __init__(self, settings: Settings, recorder: RTLSDRRecorder, interval: float = 3.0):
        self.path = Path(settings.output_dir) / "live_status.json"
        self.settings, self.recorder, self.interval = settings, recorder, interval
        self.phase = "starting"
        self.current: Optional[PlannedPass] = None
        self.plan: List[PlannedPass] = []
        self.last_result: Optional[dict] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        return self

    def stop(self):
        self.phase = "stopped"
        self.write()
        self._stop.set()

    def set(self, phase: str, current: Optional[PlannedPass] = None):
        self.phase, self.current = phase, current
        self.write()

    def write(self):
        cur = self.current.as_dict() if self.current else None
        data = {
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "phase": self.phase,
            "recorder_state": self.recorder.check_recording_status().value,
            "simulate": self.settings.simulate,
            "current": cur,
            "upcoming": [pp.as_dict() for pp in self.plan if pp.pass_.los > datetime.now(timezone.utc)][:12],
            "last_result": self.last_result,
            "station": {"name": self.settings.station.name, "lat_deg": self.settings.station.lat_deg,
                        "lon_deg": self.settings.station.lon_deg, "alt_m": self.settings.station.alt_m},
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.path)  # atomic, so the dashboard never reads half a file

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.write()
            except OSError:
                pass


class DecodeWorker:
    """Decodes METEOR recordings with SatDump in the background, one at a
    time, so a long decode never delays the next pass."""

    def __init__(self, settings: Settings, db_path: Path):
        import queue

        self.settings, self.db_path = settings, db_path
        self.queue: "queue.Queue" = queue.Queue()
        self.results: List[dict] = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, capture_id: int):
        self.queue.put(capture_id)

    def _loop(self):
        while True:
            cid = self.queue.get()
            if cid is None:
                return
            try:
                r = meteor_decode.decode_capture(
                    cid, db_path=self.db_path, satdump_path=self.settings.satdump_path,
                    results_dir=Path(self.settings.results_dir) if self.settings.results_dir else None)
                self.results.append({"capture_id": cid, **r})
                log_status(self.db_path, "decoder", "DONE", f"capture {cid}: {r['status']}")
            except Exception as exc:  # never take the station down
                logger.warning("Decoding capture %s failed: %s", cid, exc)
            finally:
                self.queue.task_done()

    def wait(self, timeout: float = None):
        """For tests / shutdown: wait until the queue is empty."""
        import time

        deadline = None if timeout is None else time.monotonic() + timeout
        while self.queue.unfinished_tasks:
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True


def run_plan(plan: List[PlannedPass], settings: Settings, *, stop_event: Optional[threading.Event] = None,
             recorder: Optional[RTLSDRRecorder] = None, record_kwargs: Optional[dict] = None,
             live: Optional[LiveStatus] = None, decoder: Optional[DecodeWorker] = None) -> List[dict]:
    """Record and process each non-skipped pass in order. Returns one summary
    dict per attempted pass (also what gets printed / saved)."""
    stop_event = stop_event or threading.Event()
    recorder = recorder or RTLSDRRecorder(RecorderConfig(
        output_dir=settings.output_dir, simulate=settings.simulate,
        device_index=settings.device_index, rtl_sdr_path=settings.rtl_sdr_path,
    ))
    summaries = []
    db_path = Path(settings.db_path) if settings.db_path else DB_PATH
    for pp in plan:
        if stop_event.is_set():
            break
        if pp.skipped_reason:
            log_status(db_path, "scheduler", "SKIPPED", f"{pp.target.name} {pp.pass_.aos.isoformat()}: {pp.skipped_reason}")
            continue
        p, t = pp.pass_, pp.target
        logger.info("Next: %s", p.describe())
        log_status(db_path, "scheduler", "WAITING", p.describe())
        if live:
            live.set("waiting for pass", pp)
        # Offset tuning: record a little below the downlink so the RTL-SDR's DC spike
        # (always at the tuned centre) can't be mistaken for the satellite.
        tuned_hz = int(t.frequency_hz - settings.tuning_offset_hz)

        # Room on disk for this pass? (frees old rejected recordings if needed)
        window_s = p.duration_seconds + settings.pre_buffer + settings.post_buffer
        disk = station_guard.ensure_space(settings.output_dir, t.sample_rate, window_s,
                                          min_free_gb=settings.min_free_gb, prune=settings.prune_rejected)
        if disk.deleted:
            log_status(db_path, "storage", "PRUNED", disk.message)
        if not disk.ok:
            logger.error("Skipping %s: %s", t.name, disk.message)
            log_status(db_path, "storage", "DISK_FULL", f"{t.name}: {disk.message}")
            summaries.append({**pp.as_dict(), "recording_status": "SKIPPED_DISK_FULL", "error": disk.message,
                              "db_row": None, "verdict": None, "iq_retention": None})
            continue

        result = recorder.record_pass(
            satellite_name=t.name, norad_id=t.norad_id, frequency_hz=tuned_hz, sample_rate=t.sample_rate,
            gain=t.gain, aos=p.aos, los=p.los, pre_buffer=settings.pre_buffer, post_buffer=settings.post_buffer,
            **(record_kwargs or {}),
        )
        # Dongle busy / unplugged / rtl_sdr crashed: try again for the rest of the pass.
        attempts = 0
        while (result.status.value in station_guard.RETRYABLE and attempts < settings.record_retries
               and not stop_event.is_set()
               and datetime.now(timezone.utc) < p.los + timedelta(seconds=settings.post_buffer)
               - timedelta(seconds=settings.retry_delay_s + settings.min_retry_window_s)):
            attempts += 1
            logger.warning("Recording %s failed (%s: %s) - retry %d/%d in %.0f s", t.name, result.status.value,
                           result.error_message, attempts, settings.record_retries, settings.retry_delay_s)
            log_status(db_path, "recorder", "RETRY",
                       f"{t.name}: {result.status.value} {result.error_message or ''} - retry {attempts}")
            if stop_event.wait(settings.retry_delay_s):
                break
            result = recorder.record_pass(
                satellite_name=t.name, norad_id=t.norad_id, frequency_hz=tuned_hz, sample_rate=t.sample_rate,
                gain=t.gain, aos=p.aos, los=p.los, pre_buffer=settings.pre_buffer, post_buffer=settings.post_buffer,
                **(record_kwargs or {}),
            )
        if live:
            live.set("processing", pp)
        doppler = None
        if pp.tle is not None and result.success and result.actual_recording_start:
            try:
                start = datetime.fromisoformat(result.actual_recording_start)
                doppler = doppler_curve_from_pass(make_propagator(pp.tle), settings.station, t.frequency_hz, start,
                                                  float(result.recording_duration_seconds or p.duration_seconds))
            except Exception as exc:  # correction is an improvement, never a blocker
                logger.warning("No Doppler curve for %s: %s", t.name, exc)
        pr = process_recording(
            result, frequency_hz=tuned_hz, sample_rate_hz=t.sample_rate,
            target_frequency_hz=t.frequency_hz, doppler=doppler, waterfall_span_hz=t.waterfall_span_hz,
            waterfall_model_path=Path(settings.waterfall_model) if settings.waterfall_model else None,
            db_path=Path(settings.db_path) if settings.db_path else None,
            output_dir=Path(settings.results_dir) if settings.results_dir else None,
            ml_model_path=Path(settings.ml_model) if settings.ml_model else None,
            save_image=settings.save_image, retention_policy=settings.retention,
            keep_threshold=settings.keep_threshold, uncertain_band=settings.uncertain_band, log_failures=True,
        )
        log_status(db_path, "recorder", result.status.value,
                   f"{t.name}: {result.error_message or result.output_file}")
        track_points = 0
        if pp.tle is not None and pr.result_id is not None:
            try:
                track = pass_track(make_propagator(pp.tle), settings.station,
                                   p.aos - timedelta(seconds=settings.pre_buffer),
                                   p.los + timedelta(seconds=settings.post_buffer),
                                   step_seconds=10.0, frequency_hz=t.frequency_hz)
                for point in track:
                    point.update(norad_id=t.norad_id, satellite_name=t.name)
                track_points = insert_pass_positions(db_path, pr.result_id, track)
            except Exception as exc:  # position history is a bonus - never lose the capture over it
                logger.warning("Could not store pass track for %s: %s", t.name, exc)
        wants_decode = (settings.decode and decoder is not None and pr.result_id is not None
                        and meteor_decode.is_meteor(t.name) and pr.verdict in ("detected", "uncertain")
                        and pr.final_iq_path)
        if wants_decode:
            decoder.submit(pr.result_id)
        log_status(db_path, "pipeline", "DONE" if pr.result_id else "ERROR",
                   f"{t.name}: row {pr.result_id}, retention {pr.retention_action}, {track_points} track points")
        summary = {
            **pp.as_dict(), "recording_status": result.status.value, "error": result.error_message,
            "db_row": pr.result_id, "rule_detected": pr.detected, "ml_status": pr.ml_status,
            "ml_confidence": pr.ml_confidence_score, "iq_retention": pr.retention_action, "verdict": pr.verdict,
            "iq_path": pr.final_iq_path, "spectrogram": pr.spectrogram_image, "track_points": track_points,
            "doppler_corrected": pr.doppler_corrected, "waterfall_ml": pr.wf_status,
            "waterfall_confidence": pr.wf_confidence_score, "waterfall_image": pr.waterfall_image,
            "decode_queued": bool(wants_decode),
        }
        summaries.append(summary)
        if live:
            live.last_result = summary
            live.set("idle")
        logger.info("Done: %s status=%s db_row=%s retention=%s", t.name, result.status.value, pr.result_id,
                    pr.retention_action)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict satellite passes and capture + classify them automatically.")
    p.add_argument("--config", type=Path, help="JSON config (station, recording defaults, satellites)")
    p.add_argument("--norad", type=int, help="Single satellite NORAD id (instead of the config's list)")
    p.add_argument("--name", help="Name for --norad")
    p.add_argument("--frequency", type=float, help="Downlink frequency in Hz for --norad")
    p.add_argument("--sample-rate", type=float)
    p.add_argument("--gain")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--alt", type=float)
    p.add_argument("--hours", type=float, help="How far ahead to plan (default 24)")
    p.add_argument("--min-elevation", dest="min_elevation_deg", type=float, help="Skip passes peaking below this (default 15)")
    p.add_argument("--pre-buffer", type=float)
    p.add_argument("--post-buffer", type=float)
    p.add_argument("--output-dir", help="Where .iq/.json recordings go")
    p.add_argument("--results-dir", help="Where summaries/spectrograms go")
    p.add_argument("--db", dest="db_path", help="SQLite database (default: sdr-doppler-prototype/data/results/captures.sqlite3)")
    p.add_argument("--ml-model", help="Trained model .joblib")
    p.add_argument("--retention", choices=POLICIES, help="Default archive-negatives")
    p.add_argument("--keep-threshold", type=float)
    p.add_argument("--uncertain-band", help="Confidence range judged 'uncertain' (IQ kept for review), e.g. 0.3,0.7; 'off' = two levels")
    p.add_argument("--tle-file", help="Use this TLE file instead of downloading from CelesTrak")
    p.add_argument("--no-image", action="store_true", help="Don't save spectrogram PNGs")
    p.add_argument("--simulate", action="store_true", help="No hardware: the recorder writes placeholder files")
    p.add_argument("--demo", action="store_true", help="Simulate one fake pass starting in 5 s (end-to-end demo)")
    p.add_argument("--list-only", action="store_true", help="Print the pass schedule and exit")
    p.add_argument("--forever", action="store_true", help="Re-plan and keep going after each planning window")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(args)

    stop_event = threading.Event()

    def _stop(*_):
        logger.warning("Stop requested - finishing the current step")
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    recorder = RTLSDRRecorder(RecorderConfig(output_dir=settings.output_dir, simulate=settings.simulate,
                                             device_index=settings.device_index, rtl_sdr_path=settings.rtl_sdr_path))
    stop_event_watch = threading.Thread(target=lambda: (stop_event.wait(), recorder.cancel_recording()), daemon=True)
    stop_event_watch.start()

    all_summaries = []
    live = None
    if not args.list_only:
        log_file = station_guard.setup_daily_log(settings.log_dir)
        logger.info("Logging to %s", log_file)
        if station_guard.keep_awake():
            logger.info("Keeping the computer awake while the station runs")
    failures_in_a_row = 0
    decoder = None
    if settings.decode and not args.list_only:
        if meteor_decode.find_satdump(settings.satdump_path):
            decoder = DecodeWorker(settings, Path(settings.db_path) if settings.db_path else DB_PATH)
            logger.info("METEOR decoding on (SatDump found)")
        else:
            logger.info("METEOR decoding off: SatDump not installed (https://www.satdump.org)")
    while not stop_event.is_set():
        try:
            plan = demo_plan(settings) if args.demo else plan_passes(settings)
        except Exception as exc:  # e.g. no internet for TLEs - retry rather than die overnight
            if not args.forever:
                raise
            failures_in_a_row += 1
            wait = min(3600, 60 * 2 ** min(failures_in_a_row, 6))
            logger.exception("Planning failed (%s) - trying again in %d s", exc, wait)
            stop_event.wait(wait)
            continue
        print(format_schedule(plan, settings.station))
        if args.list_only:
            return 0
        if live is None:
            live = LiveStatus(settings, recorder).start()
        live.plan = plan
        live.set("planned")
        schedule_file = Path(settings.output_dir) / "schedule.json"
        schedule_file.parent.mkdir(parents=True, exist_ok=True)
        schedule_file.write_text(json.dumps([pp.as_dict() for pp in plan], indent=2), encoding="utf-8")
        try:
            summaries = run_plan(plan, settings, stop_event=stop_event, recorder=recorder, live=live,
                                 decoder=decoder)
            failures_in_a_row = 0
        except Exception as exc:  # one bad pass must not end an unattended run
            if not args.forever:
                raise
            failures_in_a_row += 1
            wait = min(3600, 60 * 2 ** min(failures_in_a_row, 6))
            logger.exception("Station loop error (%s) - continuing in %d s", exc, wait)
            try:
                log_status(Path(settings.db_path) if settings.db_path else DB_PATH, "station", "ERROR",
                           f"{type(exc).__name__}: {exc} - restarting loop in {wait} s")
            except Exception:
                pass
            stop_event.wait(wait)
            continue
        all_summaries += summaries
        for s in summaries:
            if s["recording_status"] == "SKIPPED_DISK_FULL":
                print(f"- {s['satellite']} {s['aos']}: skipped - {s['error']}")
                continue
            print(f"- {s['satellite']} {s['aos']}: {s['recording_status']}, db row {s['db_row']}, "
                  f"rule={s['rule_detected']}, ml={s['ml_status']} {s['ml_confidence']}, "
                  f"waterfall-ml={s['waterfall_ml']} {s['waterfall_confidence']}, "
                  f"doppler={'corrected' if s['doppler_corrected'] else 'n/a'}, verdict {s['verdict']}, "
                  f"IQ {s['iq_retention']}")
        if args.demo or not args.forever:
            break
        if not any(pp.skipped_reason is None for pp in plan):
            stop_event.wait(3600)  # nothing to record in this window - check again in an hour
    if decoder is not None and args.demo:
        decoder.wait(timeout=600)
    if live is not None:
        live.stop()
    if args.forever:
        return 0  # only a deliberate stop (Ctrl+C) ends --forever; errors are caught and retried above
    return 0 if all(s["recording_status"] == "SUCCESS" for s in all_summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
