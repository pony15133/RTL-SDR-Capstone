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
from pipeline import process_recording
from retention import POLICIES
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
    keep_threshold: float = 0.5
    tle_file: Optional[str] = None
    tle_cache_dir: str = "tle_cache"
    save_image: bool = True
    device_index: int = 0
    rtl_sdr_path: Optional[str] = None
    simulate: bool = False


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
        keep_threshold=float(pick("keep_threshold", 0.5)),
        tle_file=pick("tle_file", None),
        tle_cache_dir=str(pick("tle_cache_dir", "tle_cache")),
        save_image=not args.no_image and bool(defaults.get("save_image", True)),
        device_index=int(pick("device_index", 0)),
        rtl_sdr_path=pick("rtl_sdr_path", None),
        simulate=bool(args.simulate or args.demo),
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

def run_plan(plan: List[PlannedPass], settings: Settings, *, stop_event: Optional[threading.Event] = None,
             recorder: Optional[RTLSDRRecorder] = None, record_kwargs: Optional[dict] = None) -> List[dict]:
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
        result = recorder.record_pass(
            satellite_name=t.name, norad_id=t.norad_id, frequency_hz=t.frequency_hz, sample_rate=t.sample_rate,
            gain=t.gain, aos=p.aos, los=p.los, pre_buffer=settings.pre_buffer, post_buffer=settings.post_buffer,
            **(record_kwargs or {}),
        )
        pr = process_recording(
            result, frequency_hz=t.frequency_hz, sample_rate_hz=t.sample_rate,
            db_path=Path(settings.db_path) if settings.db_path else None,
            output_dir=Path(settings.results_dir) if settings.results_dir else None,
            ml_model_path=Path(settings.ml_model) if settings.ml_model else None,
            save_image=settings.save_image, retention_policy=settings.retention,
            keep_threshold=settings.keep_threshold, log_failures=True,
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
        log_status(db_path, "pipeline", "DONE" if pr.result_id else "ERROR",
                   f"{t.name}: row {pr.result_id}, retention {pr.retention_action}, {track_points} track points")
        summary = {
            **pp.as_dict(), "recording_status": result.status.value, "error": result.error_message,
            "db_row": pr.result_id, "rule_detected": pr.detected, "ml_status": pr.ml_status,
            "ml_confidence": pr.ml_confidence_score, "iq_retention": pr.retention_action,
            "iq_path": pr.final_iq_path, "spectrogram": pr.spectrogram_image, "track_points": track_points,
        }
        summaries.append(summary)
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
    while not stop_event.is_set():
        plan = demo_plan(settings) if args.demo else plan_passes(settings)
        print(format_schedule(plan, settings.station))
        if args.list_only:
            return 0
        schedule_file = Path(settings.output_dir) / "schedule.json"
        schedule_file.parent.mkdir(parents=True, exist_ok=True)
        schedule_file.write_text(json.dumps([pp.as_dict() for pp in plan], indent=2), encoding="utf-8")
        summaries = run_plan(plan, settings, stop_event=stop_event, recorder=recorder)
        all_summaries += summaries
        for s in summaries:
            print(f"- {s['satellite']} {s['aos']}: {s['recording_status']}, db row {s['db_row']}, "
                  f"rule={s['rule_detected']}, ml={s['ml_status']} {s['ml_confidence']}, IQ {s['iq_retention']}")
        if args.demo or not args.forever:
            break
        if not any(pp.skipped_reason is None for pp in plan):
            stop_event.wait(3600)  # nothing to record in this window - check again in an hour
    return 0 if all(s["recording_status"] == "SUCCESS" for s in all_summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
